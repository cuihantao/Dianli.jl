Base.@kwdef struct Bus{T} <: Model{T}
    n::Int64 = 0
    a::ExtAlgeb{T} = []
    v::ExtAlgeb{T} = []
end
