Base.@kwdef struct Shunt{T} <: Model{T}
    n::Int64

    g::IntParam{T}
    b::IntParam{T}

    a::ExtAlgeb{T}
    v::ExtAlgeb{T}
end

Base.@propagate_inbounds function g_update!(shunt::Shunt{T}, ::Type{Val{:serial}}) where T <: AbstractFloat
    @simd for i = 1:shunt.n
        @inbounds shunt.a.e[i] = shunt.v[i] ^2 * shunt.g[i]
        @inbounds shunt.v.e[i] = -shunt.v[i] ^2 * shunt.b[i]
    end
end

Base.@propagate_inbounds function g_update!(shunt::Shunt{T}, ::Type{Val{:threads}}) where T <: AbstractFloat
    Threads.@threads for i = 1:shunt.n
        @inbounds shunt.a.e[i] = shunt.v[i] ^2 * shunt.g[i]
        @inbounds shunt.v.e[i] = -shunt.v[i] ^2 * shunt.b[i]
    end
end

function collect_g!(shunt::Shunt{T}, dae::DAE{T}) where T <: AbstractFloat
    addval!(shunt.a, dae)
    addval!(shunt.v, dae)
    nothing
end

function set_v!(shunt::Shunt{T}, y::Vector{T}) where T <: AbstractFloat
    setval!(shunt.a, y)
    setval!(shunt.v, y)
    nothing
end
