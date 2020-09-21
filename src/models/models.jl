module Models

using ..BasicTypes
using ..Dian: THREAD_MODES

export Model, Bus, PQ, PV, Slack, Line, Shunt
export g_update!, collect_g!, set_v!
export alloc_triplets, store_triplets!, add_triplets!, upload_triplets!

abstract type Model{T} end

alloc_triplets(ty::T, n::N) where {T<:Type{Model},N<:Integer} =
    @error "Model $ty does not define `alloc_triplets()`"

"""
Upload triplet values from models to System. Serial.

Overwrites existing values in System triplets.
"""
Base.@propagate_inbounds function upload_triplets!(
    model::Model{T},
    tpl::Triplets{T},
    ::Type{Val{:serial}},
) where {T<:AbstractFloat}
    if model.triplets.n == 0  return end
    
    span = model.triplets.addr[2] - model.triplets.addr[1] + 1
    start_pos = model.triplets.addr[1]

    @simd for i = 1:span
        @inbounds tpl.vals[start_pos + i - 1] = model.triplets.vals[i]
    end
end

"""
Upload triplet values from models to System. Threaded.

Overwrites existing values in System triplets.
"""
Base.@propagate_inbounds function upload_triplets!(
    model::Model{T},
    tpl::Triplets{T},
    ::Type{Val{:threaded}},
) where {T<:AbstractFloat}
    if model.triplets.n == 0  return end
    
    span = model.triplets.addr[2] - model.triplets.addr[1] + 1
    start_pos = model.triplets.addr[1]

    Threads.@threads for i = 1:span
        @inbounds tpl.vals[start_pos + i - 1] = model.triplets.vals[i]
    end
end


include("./bus.jl")
include("./pq.jl")
include("./pv.jl")
include("./line.jl")
include("./shunt.jl")

end
