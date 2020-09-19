module PowerSystem

using ..BasicTypes
using ..Models
using ..Dian: THREAD_MODES
using PyCall: PyObject
using SparseArrays: SparseMatrixCSC, sparse

import Base: convert
import ..Models: set_v!, collect_g!

export System
export make_instance, convert
export clear_g!, collect_g!, set_v!, YMatrix
export sg_update!, pg_update!

Base.@kwdef struct System{T}
    Bus::Bus{T}
    PQ::PQ{T}
    PV::PV{T}
    Slack::Slack{T}
    Line::Line{T}
    Shunt::Shunt{T}

    dae::DAE{T}
end

function System{T}(ss::PyObject) where {T<:AbstractFloat}
    models = [
        make_instance(ty, ss[name])
        for (ty, name) in zip(System{T}.types, fieldnames(System))
    ]
    System{T}(models...)
end

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

function make_instance(
    ty::Type{T},
    model::PyObject,
) where {T<:DAE{N}} where {N<:AbstractFloat}
    objects = Dict(f => model[f] for f in fieldnames(ty))
    ty(; objects...)
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

    models = [jss.PQ, jss.PV, jss.Slack, jss.Line, jss.Shunt]

    Threads.@threads for model in models
        @inbounds g_update!(model, tflag)
    end
end

end
