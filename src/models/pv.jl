Base.@kwdef struct PV{T} <: Model{T}
    n::Int64
    p0::IntParam{T}
    q0::IntParam{T}
    v0::IntParam{T}

    a::ExtAlgeb{T}
    v::ExtAlgeb{T}
    q::ExtAlgeb{T}
    p::ExtAlgeb{T}

    triplets::Triplets{T,Int64}
end


Base.@kwdef struct Slack{T} <: Model{T}
    n::Int64

    p0::IntParam{T}
    q0::IntParam{T}
    v0::IntParam{T}
    a0::IntParam{T}

    a::ExtAlgeb{T}
    v::ExtAlgeb{T}
    q::ExtAlgeb{T}
    p::ExtAlgeb{T}

    triplets::Triplets{T,Int64}
end


Base.@propagate_inbounds function g_update!(
    PV::PV{T},
    ::Type{Val{:serial}},
) where {T<:AbstractFloat}
    @simd for i = 1:PV.n
        @inbounds PV.a.e[i] = -PV.p[i]
        @inbounds PV.v.e[i] = -PV.q[i]
        @inbounds PV.q.e[i] = PV.v0[i] - PV.v[i]
        @inbounds PV.p.e[i] = PV.p0[i] - PV.p[i]
    end
end


Base.@propagate_inbounds function g_update!(
    PV::PV{T},
    ::Type{Val{:threaded}},
) where {T<:AbstractFloat}
    Threads.@threads for i = 1:PV.n
        @inbounds PV.a.e[i] = -PV.p[i]
        @inbounds PV.v.e[i] = -PV.q[i]
        @inbounds PV.q.e[i] = PV.v0[i] - PV.v[i]
        @inbounds PV.p.e[i] = PV.p0[i] - PV.p[i]
    end
end


Base.@propagate_inbounds function g_update!(
    Slack::Slack{T},
    ::Type{Val{:serial}},
) where {T<:AbstractFloat}
    @simd for i = 1:Slack.n
        @inbounds Slack.a.e[i] = -Slack.p[i]
        @inbounds Slack.v.e[i] = -Slack.q[i]
        @inbounds Slack.q.e[i] = Slack.v0[i] - Slack.v[i]
        @inbounds Slack.p.e[i] = Slack.a0[i] - Slack.a[i]
    end
end


Base.@propagate_inbounds function g_update!(
    Slack::Slack{T},
    ::Type{Val{:threaded}},
) where {T<:AbstractFloat}
    Threads.@threads for i = 1:Slack.n
        @inbounds Slack.a.e[i] = -Slack.p[i]
        @inbounds Slack.v.e[i] = -Slack.q[i]
        @inbounds Slack.q.e[i] = Slack.v0[i] - Slack.v[i]
        @inbounds Slack.p.e[i] = Slack.a0[i] - Slack.a[i]
    end
end


function collect_g!(pv::PV{T}, dae::DAE{T}) where {T<:AbstractFloat}
    addval!(pv.a, dae)
    addval!(pv.v, dae)
    addval!(pv.q, dae)
    addval!(pv.p, dae)
    nothing
end


function collect_g!(slack::Slack{T}, dae::DAE{T}) where {T<:AbstractFloat}
    addval!(slack.a, dae)
    addval!(slack.v, dae)
    addval!(slack.q, dae)
    addval!(slack.p, dae)
    nothing
end


function set_v!(pv::PV{T}, y::Vector{T}) where {T<:AbstractFloat}
    setval!(pv.a, y)
    setval!(pv.v, y)
    setval!(pv.q, y)
    setval!(pv.p, y)
    nothing
end


function set_v!(slack::Slack{T}, y::Vector{T}) where {T<:AbstractFloat}
    setval!(slack.a, y)
    setval!(slack.v, y)
    setval!(slack.q, y)
    setval!(slack.p, y)
    nothing
end


alloc_triplets(::Type{PV{T}}, n::N) where {T<:AbstractFloat,N<:Integer} = Triplets{T,N}(5n)


alloc_triplets(::Type{Slack{T}}, n::N) where {T<:AbstractFloat,N<:Integer} =
    Triplets{T,N}(6n)


Base.@propagate_inbounds function store_triplets!(pv::PV{T}) where {T<:AbstractFloat}
    ndev = pv.n
    @simd for i = 1:ndev
        #  d resP / dp
        @inbounds pv.triplets.rows[i] = pv.a.a[i]
        @inbounds pv.triplets.cols[i] = pv.p.a[i]

        # d resQ / dq
        @inbounds pv.triplets.rows[ndev+i] = pv.v.a[i]
        @inbounds pv.triplets.cols[ndev+i] = pv.q.a[i]

        # d Qbal / dv
        @inbounds pv.triplets.rows[2ndev+i] = pv.q.a[i]
        @inbounds pv.triplets.cols[2ndev+i] = pv.v.a[i]

        # d Pbal / dp
        @inbounds pv.triplets.rows[3ndev+i] = pv.p.a[i]
        @inbounds pv.triplets.cols[3ndev+i] = pv.p.a[i]

        # d q / dv avoid singularity
        @inbounds pv.triplets.rows[4ndev+i] = pv.q.a[i]
        @inbounds pv.triplets.cols[4ndev+i] = pv.q.a[i]
        @inbounds pv.triplets.vals[4ndev+i] = 1e-12
    end
end


Base.@propagate_inbounds function add_triplets!(
    pv::PV{T},
    ::Type{Val{:serial}},
) where {T<:AbstractFloat}
    ndev = pv.n
    @simd for i = 1:ndev
        @inbounds pv.triplets.vals[i] = -1
        @inbounds pv.triplets.vals[ndev+i] = -1
        @inbounds pv.triplets.vals[2ndev+i] = -1
        @inbounds pv.triplets.vals[3ndev+i] = -1

    end
end


Base.@propagate_inbounds function add_triplets!(
    pv::PV{T},
    ::Type{Val{:threaded}},
) where {T<:AbstractFloat}
    ndev = pv.n
    Threads.@threads for i = 1:ndev
        @inbounds pv.triplets.vals[i] = -1
        @inbounds pv.triplets.vals[ndev+i] = -1
        @inbounds pv.triplets.vals[2ndev+i] = -1
        @inbounds pv.triplets.vals[3ndev+i] = -1
    end
end

Base.@propagate_inbounds function store_triplets!(slack::Slack{T}) where {T<:AbstractFloat}
    ndev = slack.n
    @simd for i = 1:ndev
        #  d resP / dp
        @inbounds slack.triplets.rows[i] = slack.a.a[i]
        @inbounds slack.triplets.cols[i] = slack.p.a[i]

        #  d resQ / dq
        @inbounds slack.triplets.rows[ndev+i] = slack.v.a[i]
        @inbounds slack.triplets.cols[ndev+i] = slack.q.a[i]

        #  d Qbal / dv
        @inbounds slack.triplets.rows[2ndev+i] = slack.q.a[i]
        @inbounds slack.triplets.cols[2ndev+i] = slack.v.a[i]

        #  d Qbal / dv
        @inbounds slack.triplets.rows[3ndev+i] = slack.p.a[i]
        @inbounds slack.triplets.cols[3ndev+i] = slack.a.a[i]

        @inbounds slack.triplets.rows[4ndev+i] = slack.p.a[i]
        @inbounds slack.triplets.cols[4ndev+i] = slack.p.a[i]
        @inbounds slack.triplets.vals[4ndev+i] = 1e-12
        
        @inbounds slack.triplets.rows[5ndev+i] = slack.q.a[i]
        @inbounds slack.triplets.cols[5ndev+i] = slack.q.a[i]
        @inbounds slack.triplets.vals[5ndev+i] = 1e-12
    end
end

Base.@propagate_inbounds function add_triplets!(
    slack::Slack{T},
    ::Type{Val{:serial}},
) where {T<:AbstractFloat}
    ndev = slack.n
    @simd for i = 1:ndev
        #  d resP / dp
        @inbounds slack.triplets.vals[i] = -1

        #  d resQ / dq
        @inbounds slack.triplets.vals[ndev+i] = -1

        #  d Qbal / dv
        @inbounds slack.triplets.vals[2ndev+i] = -1

        #  d Qbal / dv
        @inbounds slack.triplets.vals[3ndev+i] = -1
    end
end


Base.@propagate_inbounds function add_triplets!(
    slack::Slack{T},
    ::Type{Val{:threaded}},
) where {T<:AbstractFloat}
    ndev = slack.n
    Threads.@threads for i = 1:ndev
        #  d resP / da
        @inbounds slack.triplets.vals[i] = -1

        #  d resQ / dv
        @inbounds slack.triplets.vals[ndev+i] = -1

        #  d Qbal / dv
        @inbounds slack.triplets.vals[2ndev+i] = -1

        #  d Qbal / dv
        @inbounds slack.triplets.vals[3ndev+i] = -1
    end
end

