"""
Update NR equations for a given system.
"""
function nr_update(sys::System{T}, y::Vector{T}) where T <: AbstractFloat
    set_v!(sys, y)
    sg_update(sys, Val{:threads})
    collect_g!(sys)
    return sys.dae.g
end

"""
Update NR equations for a given system in place.
"""
function nr_update!(sys::System{T}, G::Vector{T}, y::Vector{T}) where T <: AbstractFloat

    set_v!(sys, y)
    sg_update(sys, Val{:serial})
    collect_g!(sys)
    G .= sys.dae.g
end

"""
Newton-Raphson equation callback.
"""
function nr_eqn_cb!(jss::System{T}) where T <: AbstractFloat
    y0::Vector{T} = jss.dae.y;
    (G::AbstractVector, y::AbstractVector) -> nr_update!(jss, G, y)
end
