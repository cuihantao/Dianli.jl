module PowerSystem

using LoopVectorization
using ..BasicTypes
using ..Models
using ..Dianli: THREAD_MODES
using PyCall: PyObject
using SparseArrays: SparseMatrixCSC, sparse

import Base: convert
import ..Models: set_v!, collect_g!, add_triplets!

export System
export make_instance, convert
export clear_g!, collect_g!, set_v!, Ymatrix
export sg_update!, pg_update!, j_update!
export calc_Yinj!

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
            models[name] = make_instance(ty, getproperty(ss, name))
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
    dae = make_instance(DAE{T}, ss.dae)

    models[:dae] = dae
    models[:model_instances] = model_instances

    System{T}(; models...)
end

System(ss::PyObject) = System{Float64}(ss)

function make_instance(
    ty::Type{T},
    model::PyObject,
) where {T<:Model{N}} where {N<:AbstractFloat}
    objects = Dict(f => getproperty(model, f) for f in fieldnames(ty))
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
        objects[f] = getproperty(model, f)
    end

    n = length(objects[:y])
    objects[:gy] = sparse([], [], [], n, n)

    ty(; objects...)
end

"""r
Make an empty Triplets instance for System.
"""
function make_instance(
    ty::Type{M},
    model::PyObject,
) where {M<:Triplets{T, N}} where {T<:AbstractFloat, N<:Integer}
    Triplets{T, N}(0)
end

"""
Set the `g` array of a DAE object to zeros in place.
"""
function clear_g!(dae::DAE{T}) where {T<:AbstractFloat}
    dae.g .= 0.0
end

#= type converters =#
function convert(ty::Type{T}, model::PyObject) where {T<:Union{Model,DAE}}
    ty(model)
end

"""
Collect residual values into system DAE.
"""
function collect_g!(sys::System{T}) where {T<:AbstractFloat}
    clear_g!(sys.dae)
    collect_g!(sys.PQ, sys.dae)
    collect_g!(sys.PV, sys.dae)
    collect_g!(sys.Slack, sys.dae)
    collect_g!(sys.Line, sys.dae)
    collect_g!(sys.Shunt, sys.dae)
end

"""
Set variable values from the input array `y`.
"""
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
    itapc = line.itap .* exp.(-1im * line.phi)   # 1 / (tap .* exp.(1im * phi)) = itap .* exp.(-1im * phi)
    itapcconj = conj.(itapc)

    Ymatrix!(line, rows, cols, vals, y1, y2, y12, itapc, line.itap2, itapcconj)

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
    itapc,  # itap * e^(-j phi)
    itap2,
    itapcconj,
) where {T<:AbstractFloat}

    Threads.@threads for i = 1:line.n
        @inbounds rows[i] = line.a1.a[i]
        @inbounds cols[i] = line.a1.a[i]
        @inbounds vals[i] = (y1[i] + y12[i]) * itap2[i]

        @inbounds rows[line.n+i] = line.a1.a[i]
        @inbounds cols[line.n+i] = line.a2.a[i]
        @inbounds vals[line.n+i] = -y12[i] * itapcconj[i]

        @inbounds rows[2line.n+i] = line.a2.a[i]
        @inbounds cols[2line.n+i] = line.a1.a[i]
        @inbounds vals[2line.n+i] = -y12[i] * itapc[i]

        @inbounds rows[3line.n+i] = line.a2.a[i]
        @inbounds cols[3line.n+i] = line.a2.a[i]
        @inbounds vals[3line.n+i] = y12[i] + y2[i]

    end
end

"""
Calculate line injections based on admittance matrix.
"""
function calc_Yinj!(
    line::Line{T},
    bus::Bus{T},
    Ymat::SparseMatrixCSC{Complex{T},Int64},
    Vbus::Vector{Complex{T}},
    Sbus::Vector{Complex{T}},
    Pvec::Vector{T},
    Qvec::Vector{T},
) where {T<:AbstractFloat}
    @fastmath @avx for i = 1:bus.n
        Vbus[i] = bus.v[i] * exp(1im * bus.a[i])
    end

    calc_Yinj!(Ymat, Vbus, Sbus)

    @avx for i = 1:bus.n
        Pvec[i] = real(Sbus[i])
        Qvec[1] = imag(Sbus[i])
    end

    nothing
end


"""
Calculate line injections based on admittance matrix.

Given complex voltages are used.
"""
function calc_Yinj!(
    Ymat::SparseMatrixCSC{Complex{T},Int64},
    Vbus::Vector{Complex{T}},
    Sbus::Vector{Complex{T}},
) where T
    Sbus .= Vbus .* conj.(Ymat * Vbus)
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
    Threads.@threads for i in 1:length(jss.model_instances)
        @inbounds g_update!(jss.model_instances[i], tflag)
    end
end

"""
Put values in the `vals` of model triplets and then to that of System.

Warning: This function is type-unstable. Use `add_triplets!` for production.
"""
function j_update_unstable!(jss::System{T}, tflag::THREAD_MODES) where {T<:AbstractFloat}
    @inbounds add_triplets!(jss.PQ, tflag)
    @inbounds add_triplets!(jss.PV, tflag)
    @inbounds add_triplets!(jss.Slack, tflag)
    @inbounds add_triplets!(jss.Line, tflag)
    @inbounds add_triplets!(jss.Shunt, tflag)

    # reset values first (include constant jacobians)
    for i in 1:length(jss.triplets.vals)
        @inbounds jss.triplets.vals[i] = jss.triplets_init.vals[i]
    end

    # collect values into `System.triplets.vals`
    for m in jss.model_instances
        m.triplets.n <= 0 ? continue : nothing
        for (i, addr) = enumerate(m.triplets.addr[1]:m.triplets.addr[2])
            jss.triplets.vals[addr] += m.triplets.vals[i]
        end
    end

    # set values to sparse matrix
    stpl::Triplets{T} = jss.triplets
    for i in 1:length(stpl.rows)
        jss.dae.gy[stpl.rows[i], stpl.cols[i]] += stpl.vals[i]
    end

end

"""
Put values in the `vals` of model triplets and then to that of System.

Type-stable and allocation-free.
"""
function sj_update!(jss::System{T}, tflag::THREAD_MODES) where {T<:AbstractFloat}
    @inbounds add_triplets!(jss.Bus, tflag)
    @inbounds add_triplets!(jss.PQ, tflag)
    @inbounds add_triplets!(jss.PV, tflag)
    @inbounds add_triplets!(jss.Slack, tflag)
    @inbounds add_triplets!(jss.Line, tflag)
    @inbounds add_triplets!(jss.Shunt, tflag)

    # collect values into `System.triplets.vals`
    @inbounds upload_triplets!(jss.Bus, jss.triplets, tflag)
    @inbounds upload_triplets!(jss.PQ, jss.triplets, tflag)
    @inbounds upload_triplets!(jss.PV, jss.triplets, tflag)
    @inbounds upload_triplets!(jss.Slack, jss.triplets, tflag)
    @inbounds upload_triplets!(jss.Line, jss.triplets, tflag)
    @inbounds upload_triplets!(jss.Shunt, jss.triplets, tflag)

    # build sparse matrix from Triplets and update `gy` in-place

    jss.dae.gy .= sparse(jss.triplets.rows,
                         jss.triplets.cols,
                         jss.triplets.vals)
    return jss.dae.gy
end

function pj_update!(jss::System{T}, tflag::THREAD_MODES) where {T<:AbstractFloat}
    @inbounds Threads.@threads for i in 1:length(jss.model_instances)
        add_triplets!(jss.model_instances[i], tflag)
        # collect values into `System.triplets.vals`
    end

    # collect values into `System.triplets.vals`
    @inbounds upload_triplets!(jss.Bus, jss.triplets, tflag)
    @inbounds upload_triplets!(jss.PQ, jss.triplets, tflag)
    @inbounds upload_triplets!(jss.PV, jss.triplets, tflag)
    @inbounds upload_triplets!(jss.Slack, jss.triplets, tflag)
    @inbounds upload_triplets!(jss.Line, jss.triplets, tflag)
    @inbounds upload_triplets!(jss.Shunt, jss.triplets, tflag)

    # build sparse matrix from Triplets and update `gy` in-place
    jss.dae.gy .= sparse(jss.triplets.rows,
                         jss.triplets.cols,
                         jss.triplets.vals)
    return jss.dae.gy
end

j_update!(jss::System{T}, tflag::THREAD_MODES) where {T <: AbstractFloat} = sj_update!(jss, tflag)

function testp!(jss::System{T}, tflag::THREAD_MODES) where {T<:AbstractFloat}
    @inbounds Threads.@threads for i in eachindex(jss.model_instances)
        add_triplets!(jss.model_instances[i], tflag)
        # collect values into `System.triplets.vals`
    end
end

function tests!(jss::System{T}, tflag::THREAD_MODES) where {T<:AbstractFloat}
    @inbounds add_triplets!(jss.Bus, tflag)
    @inbounds add_triplets!(jss.PQ, tflag)
    @inbounds add_triplets!(jss.PV, tflag)
    @inbounds add_triplets!(jss.Slack, tflag)
    @inbounds add_triplets!(jss.Line, tflag)
    @inbounds add_triplets!(jss.Shunt, tflag)
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
