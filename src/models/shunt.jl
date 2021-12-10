Base.@kwdef struct Shunt{T} <: Model{T}
    n::Int64

    g::IntParam{T}
    b::IntParam{T}

    a::ExtAlgeb{T}
    v::ExtAlgeb{T}

    triplets::Triplets{T,Int64}
end


Base.@propagate_inbounds function g_update!(
    shunt::Shunt{T},
    ::Type{Val{:serial}},
) where {T<:AbstractFloat}
    @avx for i = 1:shunt.n
        @inbounds shunt.a.e[i] = shunt.v[i]^2 * shunt.g[i]
        @inbounds shunt.v.e[i] = -shunt.v[i]^2 * shunt.b[i]
    end
end


Base.@propagate_inbounds function g_update!(
    shunt::Shunt{T},
    ::Type{Val{:threaded}},
) where {T<:AbstractFloat}
    Threads.@threads for i = 1:shunt.n
        @inbounds shunt.a.e[i] = shunt.v[i]^2 * shunt.g[i]
        @inbounds shunt.v.e[i] = -shunt.v[i]^2 * shunt.b[i]
    end
end


function collect_g!(shunt::Shunt{T}, dae::DAE{T}) where {T<:AbstractFloat}
    addval!(shunt.a, dae)
    addval!(shunt.v, dae)
    nothing
end


function set_v!(shunt::Shunt{T}, y::Vector{T}) where {T<:AbstractFloat}
    setval!(shunt.a, y)
    setval!(shunt.v, y)
    nothing
end


Base.@inline alloc_triplets(::Type{Shunt{T}}, n::N) where {T<:AbstractFloat,N<:Integer} =
    Triplets{T,N}(2n)


Base.@propagate_inbounds function store_triplets!(shunt::Shunt{T}) where {T<:AbstractFloat}
    ndev = shunt.n
    @avx for i = 1:ndev
        #  d resP / dv
        @inbounds shunt.triplets.rows[i] = shunt.a.a[i]
        @inbounds shunt.triplets.cols[i] = shunt.v.a[i]

        # d resQ / dv
        @inbounds shunt.triplets.rows[ndev+i] = shunt.v.a[i]
        @inbounds shunt.triplets.cols[ndev+i] = shunt.v.a[i]
    end
end


Base.@propagate_inbounds function add_triplets!(
    shunt::Shunt{T},
    ::Type{Val{:serial}},
) where {T<:AbstractFloat}
    ndev = shunt.n
    @avx for i = 1:ndev
        #  d resP / dv
        @inbounds shunt.triplets.vals[i] = 2 * shunt.v[i] * shunt.g[i]

        # d resQ / dv
        @inbounds shunt.triplets.vals[ndev+i] = -2 * shunt.v[i] * shunt.b[i]

    end
end


Base.@propagate_inbounds function add_triplets!(
    shunt::Shunt{T},
    ::Type{Val{:threaded}},
) where {T<:AbstractFloat}
    ndev = shunt.n
    Threads.@threads for i = 1:ndev
        #  d resP / da
        @inbounds shunt.triplets.vals[i] = 2 * shunt.v[i] * shunt.g[i]

        # d resQ / dv
        @inbounds shunt.triplets.vals[ndev+i] = -2 * shunt.v[i] * shunt.b[i]

    end
end
