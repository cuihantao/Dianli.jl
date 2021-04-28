using CUDA
using CUDA.CUSPARSE
using CUDA.CUSOLVER
using AMD

function run_nr_cuda(jss::System{T}, y0::Vector{T}, mode::THREAD_MODES) where T<:AbstractFloat
    err::Float64 = 1
    niter::Int64 = 0
    max_iter::Int64 = 20
    tol::Float64 = 1e-5
    sol = Vector(y0)
    inc = similar(y0)

    Dianli.Routines.nr_serial(jss, sol, mode)
    Dianli.PowerSystem.j_update!(jss, mode)
    Cu_gy = CuSparseMatrixCSR(jss.dae.gy)
    Cu_g = CuArray(jss.dae.g)
    Cu_y = CuArray(y0)
    Cu_inc = CuArray(jss.dae.g)

    while (niter < max_iter) && (err > tol)
        if niter > 0
            Dianli.Routines.nr_serial(jss, sol, mode)
            Dianli.PowerSystem.j_update!(jss, mode)

            Cu_gy .= CuSparseMatrixCSR(jss.dae.gy)
            Cu_g .= CuArray(jss.dae.g)
            Cu_y .= CuArray(jss.dae.y)
        end

        Cu_inc .= CUSOLVER.csrlsvqr!(Cu_gy, Cu_g, Cu_y, 1e-4, one(Cint), 'O')
        inc .= collect(Cu_inc)

        err = maximum(abs.(inc))
        sol .-= inc
        niter += 1
    end
    return sol
end


function run_amd(jss::System{T}, y0::Vector{T}, mode::THREAD_MODES) where T<:AbstractFloat
    @timeit to "$mode Total time " begin

        err::Float64 = 1
        niter::Int64 = 0
        max_iter::Int64 = 20
        tol::Float64 = 1e-5
        sol = Vector(y0)
        inc = similar(y0)

        @timeit to "Equation updates" Dianli.Routines.nr_serial(jss, sol, mode)
        @timeit to "jac" Dianli.PowerSystem.j_update!(jss, mode)
        @timeit to "amd" q::Vector = amd(jss.dae.gy);
        @timeit to "first lu" Fq::UmfpackLU{Float64,Int64} = lu(jss.dae.gy[q, q]);

        while (niter < max_iter) && (err > tol)
            @timeit to "Equation updates" Dianli.Routines.nr_serial(jss, sol, mode)

            if niter > 0
                @timeit to "jac" Dianli.PowerSystem.j_update!(jss, mode)
                @timeit to "subsequent lu" lu!(Fq, jss.dae.gy[q, q])
            end
            @timeit to "solve eqn" inc[q] .= Fq \ jss.dae.y[q]

            err = maximum(abs.(inc))
            sol .-= inc
            niter += 1
        end
    end
    return sol
end

TimerOutputs.reset_timer!(to)
@btime run_amd($jss, $y0, Val{:serial});
to

TimerOutputs.reset_timer!(to)
@btime run_amd($jss, $y0, Val{:threaded});
to