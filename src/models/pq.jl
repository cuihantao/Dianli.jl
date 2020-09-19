Base.@kwdef struct PQ{T} <: Model{T}
    n::Int64
    p0::IntParam{T}
    q0::IntParam{T}

    a::ExtAlgeb{T}
    v::ExtAlgeb{T}

    triplets::Triplets{T,Int64}
end


Base.@propagate_inbounds function g_update!(
    PQ::PQ{T},
    ::Type{Val{:serial}},
) where {T<:AbstractFloat}
    @simd for i = 1:PQ.n
        @inbounds PQ.a.e[i] = PQ.p0[i]
        @inbounds PQ.v.e[i] = PQ.q0[i]
    end
end


Base.@propagate_inbounds function g_update!(
    PQ::PQ{T},
    ::Type{Val{:threaded}},
) where {T<:AbstractFloat}
    Threads.@threads for i = 1:PQ.n
        @inbounds PQ.a.e[i] = PQ.p0[i]
        @inbounds PQ.v.e[i] = PQ.q0[i]
    end
end


function collect_g!(pq::PQ{T}, dae::DAE{T}) where {T<:AbstractFloat}
    addval!(pq.a, dae)
    addval!(pq.v, dae)
    nothing
end


function set_v!(pq::PQ{T}, y::Vector{T}) where {T<:AbstractFloat}
    setval!(pq.a, y)
    setval!(pq.v, y)
    nothing
end


alloc_triplets(::Type{PQ{T}}, n::N) where {T<:AbstractFloat,N<:Integer} = Triplets{T,N}(0)

Base.@inline store_triplets!(::PQ{T}) where {T<:AbstractFloat} = nothing

Base.@inline push_triplets!(::PQ{T}, ::THREAD_MODES) where {T<:AbstractFloat} = nothing
