Base.@kwdef struct Line{T} <: Model{T}
    n::Int64

    gh::IntParam{T}
    gk::IntParam{T}
    ghk::IntParam{T}
    bh::IntParam{T}
    bk::IntParam{T}
    bhk::IntParam{T}
    tap::IntParam{T}
    phi::IntParam{T}

    a1::ExtAlgeb{T}
    a2::ExtAlgeb{T}
    v1::ExtAlgeb{T}
    v2::ExtAlgeb{T}

    triplets::Triplets{T, Int64}
end


Base.@propagate_inbounds function g_update!(line::Line{T}, ::Type{Val{:serial}}) where T <: AbstractFloat
    @simd for i = 1:line.n
        @inbounds line.a1.e[i] = (line.v1[i] * line.v1[i] * (line.gh[i] + line.ghk[i]) / line.tap[i] ^ 2  -
                                  line.v1[i] * line.v2[i] * (line.ghk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i]) +
                                  line.bhk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i])) / line.tap[i])

        @inbounds line.v1.e[i] = (-line.v1[i] ^ 2 * (line.bh[i] + line.bhk[i]) / line.tap[i] ^ 2 -
                                  line.v1[i] * line.v2[i] * (line.ghk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i]) -
                                  line.bhk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i])) / line.tap[i])

        @inbounds line.a2.e[i]= (line.v2[i] ^ 2 * (line.gh[i] + line.ghk[i]) -
                                 line.v1[i] * line.v2[i] * (line.ghk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i]) -
                                 line.bhk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i])) / line.tap[i])

        @inbounds line.v2.e[i] = (-line.v2[i] ^ 2 * (line.bh[i] + line.bhk[i]) +
                                  line.v1[i] * line.v2[i] * (line.ghk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i]) +
                                  line.bhk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i])) / line.tap[i])
    end
end

Base.@propagate_inbounds function g_update!(line::Line{T}, ::Type{Val{:threaded}}) where T <: AbstractFloat
    Threads.@threads for i = 1:line.n
        @inbounds line.a1.e[i] = (line.v1[i] * line.v1[i] * (line.gh[i] + line.ghk[i]) / line.tap[i] ^ 2  -
                                  line.v1[i] * line.v2[i] * (line.ghk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i]) +
                                  line.bhk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i])) / line.tap[i])
        @inbounds line.v1.e[i] = (-line.v1[i] ^ 2 * (line.bh[i] + line.bhk[i]) / line.tap[i] ^ 2 -
                                  line.v1[i] * line.v2[i] * (line.ghk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i]) -
                                  line.bhk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i])) / line.tap[i])

        @inbounds line.a2.e[i]= (line.v2[i] ^ 2 * (line.gh[i] + line.ghk[i]) -
                                 line.v1[i] * line.v2[i] * (line.ghk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i]) -
                                 line.bhk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i])) / line.tap[i])

        @inbounds line.v2.e[i] = (-line.v2[i] ^ 2 * (line.bh[i] + line.bhk[i]) +
                                  line.v1[i] * line.v2[i] * (line.ghk[i] * sin(line.a1[i] - line.a2[i] - line.phi[i]) +
                                  line.bhk[i] * cos(line.a1[i] - line.a2[i] - line.phi[i])) / line.tap[i])
    end
end

function collect_g!(line::Line{T}, dae::DAE{T}) where T <: AbstractFloat
    addval!(line.a1, dae)
    addval!(line.a2, dae)
    addval!(line.v1, dae)
    addval!(line.v2, dae)
    nothing
end

function set_v!(line::Line{T}, y::Vector{T}) where T <: AbstractFloat
    setval!(line.a1, y)
    setval!(line.a2, y)
    setval!(line.v1, y)
    setval!(line.v2, y)
    nothing
end


alloc_triplets(::Type{Line{T}}, n::N) where {T <: AbstractFloat, N <: Integer} = Triplets{T, N}(16n)
