Base.@kwdef struct PV{T} <: Model{T}
    n::Int64
    p0::IntParam{T}
    q0::IntParam{T}
    v0::IntParam{T}

    a::ExtAlgeb{T}
    v::ExtAlgeb{T}
    q::ExtAlgeb{T}

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

end


Base.@propagate_inbounds function g_update!(PV::PV{T}, ::Type{Val{:serial}}) where T <: AbstractFloat
    @simd for i = 1:PV.n
        @inbounds PV.a.e[i] = -PV.p0[i]
        @inbounds PV.v.e[i] = -PV.q[i]
        @inbounds PV.q.e[i] = PV.v0[i] - PV.v[i]
    end
end

Base.@propagate_inbounds function g_update!(PV::PV{T}, ::Type{Val{:threaded}}) where T <: AbstractFloat
    Threads.@threads for i = 1:PV.n
        @inbounds PV.a.e[i] = -PV.p0[i]
        @inbounds PV.v.e[i] = -PV.q[i]
        @inbounds PV.q.e[i] = PV.v0[i] - PV.v[i]
    end
end



Base.@propagate_inbounds function g_update!(Slack::Slack{T}, ::Type{Val{:serial}}) where T <: AbstractFloat
    @simd for i = 1:Slack.n
        @inbounds Slack.a.e[i] = -Slack.p[i]
        @inbounds Slack.v.e[i] = -Slack.q[i]
        @inbounds Slack.q.e[i] = Slack.v0[i] - Slack.v[i]
        @inbounds Slack.p.e[i] = Slack.a0[i] - Slack.a[i]
    end
end

Base.@propagate_inbounds function g_update!(Slack::Slack{T}, ::Type{Val{:threaded}}) where T <: AbstractFloat
    Threads.@threads for i = 1:Slack.n
        @inbounds Slack.a.e[i] = -Slack.p[i]
        @inbounds Slack.v.e[i] = -Slack.q[i]
        @inbounds Slack.q.e[i] = Slack.v0[i] - Slack.v[i]
        @inbounds Slack.p.e[i] = Slack.a0[i] - Slack.a[i]
    end
end

function collect_g!(pv::PV{T}, dae::DAE{T}) where T <: AbstractFloat
    addval!(pv.a, dae)
    addval!(pv.v, dae)
    addval!(pv.q, dae)
    nothing
end

function collect_g!(slack::Slack{T}, dae::DAE{T}) where T <: AbstractFloat
    addval!(slack.a, dae)
    addval!(slack.v, dae)
    addval!(slack.q, dae)
    addval!(slack.p, dae)
    nothing
end

function set_v!(pv::PV{T}, y::Vector{T}) where T <: AbstractFloat
    setval!(pv.a, y)
    setval!(pv.v, y)
    setval!(pv.q, y)
    nothing
end

function set_v!(slack::Slack{T}, y::Vector{T}) where T <: AbstractFloat
    setval!(slack.a, y)
    setval!(slack.v, y)
    setval!(slack.q, y)
    setval!(slack.p, y)
    nothing
end
