Base.@kwdef struct Line{T} <: Model{T}
    n::Int64

    gh::IntParam{T}
    gk::IntParam{T}
    ghk::IntParam{T}
    bh::IntParam{T}
    bk::IntParam{T}
    bhk::IntParam{T}
    itap::IntParam{T}
    itap2::IntParam{T}
    phi::IntParam{T}

    a1::ExtAlgeb{T}
    a2::ExtAlgeb{T}
    v1::ExtAlgeb{T}
    v2::ExtAlgeb{T}

    triplets::Triplets{T,Int64}
end


Base.@propagate_inbounds function g_update!(
    line::Line{T},
    ::Type{Val{:serial}},
) where {T<:AbstractFloat}
    @inbounds @avx for i = 1:line.n
        line.a1.e[i] = (
            line.v1[i] * line.v1[i] * (line.gh[i] + line.ghk[i]) * line.itap2[i] -
            line.v1[i] *
            line.v2[i] *
            line.itap[i] *
            (
                line.ghk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i]) +
                line.bhk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i])
            )
        )

        line.v1.e[i] = (
            -line.v1[i]^2 * (line.bh[i] + line.bhk[i]) * line.itap2[i] -
            line.v1[i] *
            line.v2[i] *
            line.itap[i] *
            (
                line.ghk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i]) -
                line.bhk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i])
            )
        )

        line.a2.e[i] = (
            line.v2[i]^2 * (line.gh[i] + line.ghk[i]) -
            line.v1[i] *
            line.v2[i] *
            line.itap[i] *
            (
                line.ghk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i]) -
                line.bhk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i])
            )
        )

        line.v2.e[i] = (
            -line.v2[i]^2 * (line.bh[i] + line.bhk[i]) +
            line.v1[i] *
            line.v2[i] *
            line.itap[i] *
            (
                line.ghk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i]) +
                line.bhk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i])
            )
        )
    end
end

Base.@propagate_inbounds function g_update!(
    line::Line{T},
    ::Type{Val{:threaded}},
) where {T<:AbstractFloat}
    Threads.@threads for i = 1:line.n
        @inbounds line.a1.e[i] = (
            line.v1[i] * line.v1[i] * (line.gh[i] + line.ghk[i]) * line.itap2[i] -
            line.v1[i] *
            line.v2[i] *
            (
                line.ghk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i]) +
                line.bhk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i])
            ) * line.itap[i]
        )
        @inbounds line.v1.e[i] = (
            -line.v1[i]^2 * (line.bh[i] + line.bhk[i]) * line.itap2[i] -
            line.v1[i] *
            line.v2[i] *
            (
                line.ghk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i]) -
                line.bhk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i])
            ) * line.itap[i]
        )

        @inbounds line.a2.e[i] = (
            line.v2[i]^2 * (line.gh[i] + line.ghk[i]) -
            line.v1[i] *
            line.v2[i] *
            (
                line.ghk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i]) -
                line.bhk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i])
            ) * line.itap[i]
        )

        @inbounds line.v2.e[i] = (
            -line.v2[i]^2 * (line.bh[i] + line.bhk[i]) +
            line.v1[i] *
            line.v2[i] *
            (
                line.ghk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i]) +
                line.bhk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i])
            ) * line.itap[i]
        )
    end
end

function collect_g!(line::Line{T}, dae::DAE{T}) where {T<:AbstractFloat}
    addval!(line.a1, dae)
    addval!(line.a2, dae)
    addval!(line.v1, dae)
    addval!(line.v2, dae)
    nothing
end

function set_v!(line::Line{T}, y::Vector{T}) where {T<:AbstractFloat}
    setval!(line.a1, y)
    setval!(line.a2, y)
    setval!(line.v1, y)
    setval!(line.v2, y)
    nothing
end


alloc_triplets(::Type{Line{T}}, n::N) where {T<:AbstractFloat,N<:Integer} =
    Triplets{T,N}(16n)


Base.@propagate_inbounds function store_triplets!(line::Line{T}) where {T<:AbstractFloat}
    ndev = line.n
    @avx for i = 1:ndev
        # d a1 / d a1
        @inbounds line.triplets.rows[0ndev+i] = line.a1.a[i]
        @inbounds line.triplets.cols[0ndev+i] = line.a1.a[i]

        # d a1 / d a2
        @inbounds line.triplets.rows[1ndev+i] = line.a1.a[i]
        @inbounds line.triplets.cols[1ndev+i] = line.a2.a[i]

        # d a1 / d v1
        @inbounds line.triplets.rows[2ndev+i] = line.a1.a[i]
        @inbounds line.triplets.cols[2ndev+i] = line.v1.a[i]

        # d a1 / d v2
        @inbounds line.triplets.rows[3ndev+i] = line.a1.a[i]
        @inbounds line.triplets.cols[3ndev+i] = line.v2.a[i]

        # d a2 / d a1
        @inbounds line.triplets.rows[4ndev+i] = line.a2.a[i]
        @inbounds line.triplets.cols[4ndev+i] = line.a1.a[i]

        # d a2 / d a2
        @inbounds line.triplets.rows[5ndev+i] = line.a2.a[i]
        @inbounds line.triplets.cols[5ndev+i] = line.a2.a[i]

        # d a2 / d v1
        @inbounds line.triplets.rows[6ndev+i] = line.a2.a[i]
        @inbounds line.triplets.cols[6ndev+i] = line.v1.a[i]

        # d a2 / d v2
        @inbounds line.triplets.rows[7ndev+i] = line.a2.a[i]
        @inbounds line.triplets.cols[7ndev+i] = line.v2.a[i]

        # d v1 / d a1
        @inbounds line.triplets.rows[8ndev+i] = line.v1.a[i]
        @inbounds line.triplets.cols[8ndev+i] = line.a1.a[i]

        # d v1 / d a2
        @inbounds line.triplets.rows[9ndev+i] = line.v1.a[i]
        @inbounds line.triplets.cols[9ndev+i] = line.a2.a[i]

        # d v1 / d v1
        @inbounds line.triplets.rows[10ndev+i] = line.v1.a[i]
        @inbounds line.triplets.cols[10ndev+i] = line.v1.a[i]

        # d v1 / d v2
        @inbounds line.triplets.rows[11ndev+i] = line.v1.a[i]
        @inbounds line.triplets.cols[11ndev+i] = line.v2.a[i]

        # d v2 / d a1
        @inbounds line.triplets.rows[12ndev+i] = line.v2.a[i]
        @inbounds line.triplets.cols[12ndev+i] = line.a1.a[i]

        # d v2 / d a2
        @inbounds line.triplets.rows[13ndev+i] = line.v2.a[i]
        @inbounds line.triplets.cols[13ndev+i] = line.a2.a[i]

        # d v2 / d v1
        @inbounds line.triplets.rows[14ndev+i] = line.v2.a[i]
        @inbounds line.triplets.cols[14ndev+i] = line.v1.a[i]

        # d v2 / d v2
        @inbounds line.triplets.rows[15ndev+i] = line.v2.a[i]
        @inbounds line.triplets.cols[15ndev+i] = line.v2.a[i]
    end

end


Base.@propagate_inbounds function add_triplets!(
    line::Line{T},
    ::Type{Val{:serial}},
) where {T<:AbstractFloat}
    ndev = line.n

    @avx for i = 1:ndev
        @inbounds line.triplets.vals[0ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[1ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                -line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[2ndev+i] = (
            -line.itap[i] *
            line.v2[i] *
            (
                -line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            ) + 2 * line.itap2[i] * line.v1[i] * (line.gh[i] + line.ghk[i])
        )

        @inbounds line.triplets.vals[3ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            (
                -line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[4ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                -line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[5ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[6ndev+i] =
            -line.itap[i] *
            line.v2[i] *
            (
                line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[7ndev+i] = (
            -line.itap[i] *
            line.v1[i] *
            (
                line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            ) + 2 * line.v2[i] * (line.gh[i] + line.ghk[i])
        )

        @inbounds line.triplets.vals[8ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                -line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[9ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[10ndev+i] = (
            -line.itap[i] *
            line.v2[i] *
            (
                -line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            ) - 2 * line.itap2[i] * line.v1[i] * (line.bh[i] + line.bhk[i])
        )

        @inbounds line.triplets.vals[11ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            (
                -line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[12ndev+i] =
            line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[13ndev+i] =
            line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                -line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[14ndev+i] =
            line.itap[i] *
            line.v2[i] *
            (
                line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[15ndev+i] = (
            line.itap[i] *
            line.v1[i] *
            (
                line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            ) - 2 * line.v2[i] * (line.bh[i] + line.bhk[i])
        )

    end


end


Base.@propagate_inbounds function add_triplets!(
    line::Line{T},
    ::Type{Val{:threaded}},
) where {T<:AbstractFloat}
    ndev = line.n

    Threads.@threads for i = 1:ndev
        @inbounds line.triplets.vals[0ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[1ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                -line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[2ndev+i] = (
            -line.itap[i] *
            line.v2[i] *
            (
                -line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            ) + 2 * line.itap2[i] * line.v1[i] * (line.gh[i] + line.ghk[i])
        )

        @inbounds line.triplets.vals[3ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            (
                -line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[4ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                -line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[5ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[6ndev+i] =
            -line.itap[i] *
            line.v2[i] *
            (
                line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[7ndev+i] = (
            -line.itap[i] *
            line.v1[i] *
            (
                line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            ) + 2 * line.v2[i] * (line.gh[i] + line.ghk[i])
        )

        @inbounds line.triplets.vals[8ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                -line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[9ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[10ndev+i] = (
            -line.itap[i] *
            line.v2[i] *
            (
                -line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            ) - 2 * line.itap2[i] * line.v1[i] * (line.bh[i] + line.bhk[i])
        )

        @inbounds line.triplets.vals[11ndev+i] =
            -line.itap[i] *
            line.v1[i] *
            (
                -line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[12ndev+i] =
            line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) +
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[13ndev+i] =
            line.itap[i] *
            line.v1[i] *
            line.v2[i] *
            (
                -line.bhk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[14ndev+i] =
            line.itap[i] *
            line.v2[i] *
            (
                line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            )

        @inbounds line.triplets.vals[15ndev+i] = (
            line.itap[i] *
            line.v1[i] *
            (
                line.bhk[i] * cos(-line.a1[i] + line.a2[i] + line.phi[i]) -
                line.ghk[i] * sin(-line.a1[i] + line.a2[i] + line.phi[i])
            ) - 2 * line.v2[i] * (line.bh[i] + line.bhk[i])
        )

    end


end
