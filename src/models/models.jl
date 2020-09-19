module Models

using ..BasicTypes

export Model, Bus, PQ, PV, Slack, Line, Shunt
export g_update!, collect_g!, set_v!
export alloc_triplets, push_triplets!

abstract type Model{T} end

alloc_triplets(ty::T, n::N) where {T <: Type{Model}, N <: Integer} = @error "Model $ty does not define `alloc_triplets()`"


include("./bus.jl")
include("./pq.jl")
include("./pv.jl")
include("./line.jl")
include("./shunt.jl")

end
