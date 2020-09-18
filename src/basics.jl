module BasicTypes

using PyCall: PyObject

import Base: convert

export Component, VComponent, VAComponent
export Param, IntParam, ExtParam
export Variable, IntVariable, ExtVariable
export IntAlgeb, ExtAlgeb
export Model, DAE
export addval!, setval!
export convert

abstract type Component{T}<:AbstractVector{T} end

abstract type VComponent{T} <: Component{T} end

Base.size(VC::VComponent) = Base.size(VC.v)
Base.iterate(VC::VComponent) = Base.iterate(VC.v)
Base.iterate(VC::VComponent, iter) = Base.iterate(VC.v, iter)
Base.iterate(VC::VComponent, iter, state) = Base.Iterate(VC.v, iter, state)
Base.@propagate_inbounds function Base.getindex(VC::VComponent, i::Int)
   @boundscheck checkbounds(VC, i)
   VC.v[i]
end
Base.@propagate_inbounds function Base.setindex!(VC::VComponent, v::T, i::Int) where T<:AbstractFloat
    @boundscheck checkbounds(VC, i)
    VC.v[i] = v
end

abstract type VAComponent{T} <: VComponent{T} end

#= Parameter =#
abstract type Param{T}<:VComponent{T}  end

Base.@kwdef struct IntParam{T} <: Param{T}
    v::Vector{T}
    vin::Vector{T}
    pu::Symbol  # :device or :system
end

IntParam{T}(p::PyObject) where T = IntParam{T}(vec(p.v), vec(p.v), :system)

convert(::Type{T}, p::PyObject) where T<: IntParam = T(p)

Base.@kwdef struct ExtParam{T} <: Param{T}
    v::Vector{T}
end

#= Variables =#
abstract type Variable{T} <: VAComponent{T} end

abstract type IntVariable{T} <: Variable{T} end

abstract type ExtVariable{T} <: Variable{T} end

Base.@kwdef struct IntAlgeb{T} <: IntVariable{T}
    v::Vector{T} = []
    e::Vector{T} = []
    a::Vector{Int64} = []
end

struct ExtAlgeb{T} <: ExtVariable{T}
    v::Vector{T}
    e::Vector{T}
    a::Vector{Int64}
end

# convert to 1-based index in Julia
ExtAlgeb{T}(var::PyObject) where T = ExtAlgeb{T}(var.v, zeros(size(var.e)), vec(var.a) .+ 1)

convert(::Type{T}, var::PyObject) where T<: ExtAlgeb = T(var)

#= DAE Arrays =#
Base.@kwdef struct DAE{T}
    y::Vector{T}
    g::Vector{T}
end

Base.@inline function addval!(v::ExtAlgeb{T}, dae::DAE{T}) where T <: AbstractFloat
    for i = 1:length(v.a)
        @inbounds dae.g[v.a[i]] += v.e[i]
    end
end

Base.@inline function setval!(v::ExtAlgeb{T}, y::Vector{T}) where T <: AbstractFloat
    for i = 1:length(v.a)
        @inbounds v.v[i] = y[v.a[i]]
    end
end

end