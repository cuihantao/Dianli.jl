"""
Update NR equations for a given system.

Models are called in serial.
"""
function nr_serial(
    sys::System{T},
    y::Vector{T},
    mode::THREAD_MODES,
) where {T<:AbstractFloat}
    set_v!(sys, y)
    sg_update!(sys, mode)
    collect_g!(sys)
    return sys.dae.g
end


"""
Update NR equations for a given system in place.

Models are called in serial.
"""
function nr_serial!(
    sys::System{T},
    G::Vector{T},
    y::Vector{T},
    mode::THREAD_MODES,
) where {T<:AbstractFloat}
    set_v!(sys, y)
    sg_update!(sys, mode)
    collect_g!(sys)
    G .= sys.dae.g
    nothing
end


"""
Update NR equations for a given system.

Models equations are called in parallel.
"""
function nr_threaded(
    sys::System{T},
    y::Vector{T},
    mode::THREAD_MODES,
) where {T<:AbstractFloat}
    set_v!(sys, y)
    pg_update!(sys, mode)
    collect_g!(sys)
    sys.dae.g
end



"""
Update NR equations for a given system in place.

Models equations are called in parallel.
"""
function nr_threaded!(
    sys::System{T},
    G::Vector{T},
    y::Vector{T},
    mode::THREAD_MODES,
) where {T<:AbstractFloat}
    set_v!(sys, y)
    pg_update!(sys, mode)
    collect_g!(sys)
    G .= sys.dae.g
    nothing
end


function nr_jac_serial!(
    sys::System{T},
    J,
    y::Vector{T},
    mode::THREAD_MODES,
) where {T<:AbstractFloat}
    set_v!(sys, y)
    j_update!(sys, mode)
    J .= sys.dae.gy
    nothing
end

"""
Newton-Raphson with serial equation calls to each model.

Argument `mode` specifies if calls within each model should be serial or threaded.
"""
function nr_serial!(jss::System{T}, mode::THREAD_MODES) where {T<:AbstractFloat}
    (G::AbstractVector, y::AbstractVector) -> nr_serial!(jss, G, y, mode)
end


"""
Newton-Raphson with serial equation calls to each model.

Argument `mode` specifies if calls within each model should be serial or threaded.
"""
function nr_threaded!(jss::System{T}, mode::THREAD_MODES) where {T<:AbstractFloat}
    (G::AbstractVector, y::AbstractVector) -> nr_threaded!(jss, G, y, mode)
end


function nr_jac_serial!(jss::System{T}, mode::THREAD_MODES) where T<: AbstractFloat
    (J, x) -> nr_jac_serial!(jss, J, x, mode)
end
