module PowerSystem

using ..BasicTypes
using ..Models
using ..Dian: THREAD_MODES
using PyCall: PyObject
using SparseArrays: SparseMatrixCSC, sparse

import Base: convert
import ..Models: set_v!, collect_g!, add_triplets!

export System
export make_instance, convert
export clear_g!, collect_g!, set_v!, add_triplets!, YMatrix
export sg_update!, pg_update!

Base.@kwdef struct System{T}
    Bus::Bus{T}
    PQ::PQ{T}
    PV::PV{T}
    Slack::Slack{T}
    Line::Line{T}
    Shunt::Shunt{T}

    dae::DAE{T}

    triplets::Triplets{T, Int64}
    triplets_init::Triplets{T, Int64}

    model_instances::Vector{Model{T}} = []
end

function System{T}(ss::PyObject) where {T<:AbstractFloat}

    # add all models first
    models = Dict{Symbol, Any}()

    for (ty, name) in zip(System{T}.types, fieldnames(System))
        if ty <: Model
            models[name] = make_instance(ty, ss[name])
        end
    end
    # Note: IMPORTANT!!
    #   The order of models in `models::Dict` is random!!
    
    model_instances = collect(values(models))
    
    # then merge triplets
    t1 = merge_triplets(model_instances...)
    t2 = merge_triplets(model_instances...)  # same as `t1` on different memory
    models[:triplets] = t1
    models[:triplets_init] = t2

    # add `dae` last
    models[:dae] = make_instance(DAE{T}, ss[:dae])
    models[:model_instances] = model_instances

    System{T}(; models...)
end

System(ss::PyObject) = System{Float64}(ss)

function make_instance(
    ty::Type{T},
    model::PyObject,
) where {T<:Model{N}} where {N<:AbstractFloat}
    objects = Dict(f => model[f] for f in fieldnames(ty))
    objects[:triplets] = alloc_triplets(ty, objects[:n])

    mdl = ty(; objects...)
    store_triplets!(mdl)

    return mdl
end

"""Make a DAE instance."""
function make_instance(
    ty::Type{T},
    model::PyObject,
) where {T<:DAE{N}} where {N<:AbstractFloat}

    objects = Dict{Symbol, Any}()
    for (ftype, f) in zip(ty.types, fieldnames(ty))
        (ftype <: SparseMatrixCSC) ? continue : nothing
        objects[f] = model[f]
    end

    ty(; objects...)
end

"""
Make an empty Triplets instance for System.
"""
function make_instance(
    ty::Type{M},
    model::PyObject,
) where {M<:Triplets{T, N}} where {T<:AbstractFloat, N<:Integer}
    Triplets{T, N}(0)
end

function clear_g!(dae::DAE{T}) where {T<:AbstractFloat}
    dae.g .= 0.0
end

#= type converters =#
function convert(ty::Type{T}, model::PyObject) where {T<:Union{Model,DAE}}
    ty(model)
end

function collect_g!(sys::System{T}) where {T<:AbstractFloat}
    clear_g!(sys.dae)
    collect_g!(sys.PQ, sys.dae)
    collect_g!(sys.PV, sys.dae)
    collect_g!(sys.Slack, sys.dae)
    collect_g!(sys.Line, sys.dae)
    collect_g!(sys.Shunt, sys.dae)
end

function set_v!(sys::System{T}, y::Vector{T}) where {T<:AbstractFloat}
    set_v!(sys.PQ, y)
    set_v!(sys.PV, y)
    set_v!(sys.Slack, y)
    set_v!(sys.Line, y)
    set_v!(sys.Shunt, y)
end

"""
Build admittance matrix for System (Line and Bus).
"""
function Ymatrix(line::Line{T}, bus::Bus{T}) where {T<:AbstractFloat}
    rows = zeros(Int64, 4line.n)
    cols = zeros(Int64, 4line.n)
    vals = zeros(Complex{Float64}, 4line.n)

    y1 = line.gh + line.bh * 1im
    y2 = line.gk + line.bk * 1im
    y12 = line.ghk + line.bhk * 1im
    itap2 = 1 ./ line.tap ./ line.tap
    itap = 1 ./ line.tap .* exp.(1im * line.phi)
    itapconj = conj.(itap)

    Ymatrix!(line, rows, cols, vals, y1, y2, y12, itap, itap2, itapconj)

    sparse(rows, cols, vals, bus.n, bus.n)
end

"""
Allocation-free function for Ymatrix building
"""
Base.@inline function Ymatrix!(
    line::Line{T},
    rows::Vector{Int64},
    cols::Vector{Int64},
    vals::Vector{Complex{Float64}},
    y1,
    y2,
    y12,
    itap,
    itap2,
    itapconj,
) where {T<:AbstractFloat}

    Threads.@threads for i = 1:line.n
        @inbounds rows[i] = line.a1.a[i]
        @inbounds cols[i] = line.a1.a[i]
        @inbounds vals[i] = (y1[i] + y12[i]) * itap2[i]

        @inbounds rows[line.n+i] = line.a1.a[i]
        @inbounds cols[line.n+i] = line.a2.a[i]
        @inbounds vals[line.n+i] = -y12[i] * itapconj[i]

        @inbounds rows[2line.n+i] = line.a2.a[i]
        @inbounds cols[2line.n+i] = line.a1.a[i]
        @inbounds vals[2line.n+i] = -y12[i] * itap[i]

        @inbounds rows[3line.n+i] = line.a2.a[i]
        @inbounds cols[3line.n+i] = line.a2.a[i]
        @inbounds vals[3line.n+i] = y12[i] + y2[i]

    end
end

function calc_Yinj!(
    line::Line{T},
    bus::Bus{T},
    Ymat::SparseMatrixCSC{Complex{T},Int64},
    Vbus::Vector{Complex{T}},
    Sbus::Vector{Complex{T}},
    Pvec::Vector{T},
    Qvec::Vector{T},
) where {T<:AbstractFloat}
    @simd for i = 1:bus.n
        @inbounds Vbus[i] = bus.v[i] * exp(1im * bus.a[i])
    end

    Sbus .= Vbus .* conj.(Ymat * Vbus)

    @simd for i = 1:bus.n
        @inbounds Pvec[i] = real(Sbus[i])
        @inbounds Qvec[1] = imag(Sbus[i])
    end

    nothing
end

"""
Model-serial g_update
"""
function sg_update!(jss::System{T}, tflag::THREAD_MODES) where {T<:AbstractFloat}
    @inbounds g_update!(jss.PQ, tflag)
    @inbounds g_update!(jss.PV, tflag)
    @inbounds g_update!(jss.Slack, tflag)
    @inbounds g_update!(jss.Line, tflag)
    @inbounds g_update!(jss.Shunt, tflag)
end

"""
Model-parallel g_update
"""
function pg_update!(jss::System{T}, tflag::THREAD_MODES) where {T<:AbstractFloat}

    # TODO: replace with `model_instances`
    models = [jss.PQ, jss.PV, jss.Slack, jss.Line, jss.Shunt]

    Threads.@threads for model in models
        @inbounds g_update!(model, tflag)
    end
end

"""
Put values in the `vals` of model triplets
"""
function add_triplets!(jss::System{T}, tflag::THREAD_MODES) where {T<:AbstractFloat}
    @inbounds add_triplets!(jss.PQ, tflag)
    @inbounds add_triplets!(jss.PV, tflag)
    @inbounds add_triplets!(jss.Slack, tflag)
    @inbounds add_triplets!(jss.Line, tflag)
    @inbounds add_triplets!(jss.Shunt, tflag)
    
    # reset values first (include constant jacobians)
    for i in 1:length(jss.triplets.vals)
        @inbounds jss.triplets.vals[i] = jss.triplets_init.vals[i]
    end

    # collect values
    for m in jss.model_instances
        for (i, addr) = enumerate(m.triplets.addr[1]:m.triplets.addr[2])
            jss.triplets.vals[addr] = m.triplets.vals[i]
        end
    end

end

"""Merge model triplets into System."""
function merge_triplets(models...) 
    count::Int64 = 0

    for m in models
        @assert hasproperty(m, :triplets)
        count += length(m.triplets.rows)
    end

    rows = zeros(Int64, count)
    cols = zeros(Int64, count)
    vals = zeros(Float64, count)
    
    pos::Int64 = 1
    for m in models
        tpl = m.triplets
        if tpl.n <= 0
            continue
        end

        tpl.addr[1] = pos
        for (row, col, val) in zip(tpl.rows, tpl.cols, tpl.vals)
            rows[pos] = row
            cols[pos] = col
            vals[pos] = val  # also merges constant jacobian elements
            pos += 1
        end
        tpl.addr[2] = pos - 1
    end

    Triplets{Float64, Int64}(count, rows, cols, vals, [1, pos-1])
end

end  # end of `PowerSystem` module


