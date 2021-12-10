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

"""
Run standard Newton-Raphson power flow
"""
function run_nr(jss::System{T}, y0::Vector{T}, MODE::THREAD_MODES) where T<:AbstractFloat
    err::Float64 = 1
    niter::Int64 = 0
    max_iter::Int64 = 20
    tol::Float64 = 1e-8
    sol = Vector(y0)
    inc = similar(y0)

    nr_serial(jss, sol, MODE)
    j_update!(jss, MODE)
    F = KLU.lu(jss.dae.gy)

    while (niter < max_iter) && (err > tol)
        if niter > 0
            nr_serial(jss, sol, MODE)
            j_update!(jss, MODE)
            F = KLU.lu!(F, jss.dae.gy)
        end
        inc .= F \ jss.dae.g

        err = maximum(abs.(inc))
        sol .-= inc
        niter += 1
    end
    return sol, err, niter
end

"""
Solve Newton power flow using NLsolve.nlsolve.
"""
function nlsolve_nr(jss::System{T}, y0::Vector{T}, MODE::THREAD_MODES; kwargs...) where T<:AbstractFloat
    j_update!(jss, MODE);
    J0 = sparse(jss.dae.gy);
    g0 = Vector(jss.dae.g);
    df = OnceDifferentiable(nr_serial!(jss, MODE),
                            nr_jac_serial!(jss, MODE),
                            y0, g0, J0);
    sol = nlsolve(df, y0; kwargs...)
end