Base.@kwdef struct Bus{T} <: Model{T}
    n::Int64 = 0
    a::ExtAlgeb{T} = []
    v::ExtAlgeb{T} = []

    triplets::Triplets{T, Int64}
end


alloc_triplets(::Type{Bus{T}}, n::N) where {T <: AbstractFloat, N <: Integer} = Triplets{T, N}(0)


