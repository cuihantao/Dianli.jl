mpc_dir = joinpath(@__DIR__, "mpc_data")    
cases = ["case5.m", "case14.m", "case39.m", "case118.m", "case300.m"]

@testset "Power flows" begin

for name in cases
    case_path = joinpath(mpc_dir, name)
    @testset "Testing $name" begin

        ss = Dian.Andes.py.load(case_path, no_output=true)
        ss.PFlow.init()

        jss = System{Float64}(ss)
        y0 = Vector(jss.dae.y)
        ss.PFlow.run()
        idx = 1:2jss.Bus.n
        
        ## Tests start - nlsolve without Jacobian
        @testset "nlsolve without Jac" begin
            sol = nlsolve(nr_serial!(jss, Val{:serial}), y0);
            @test maximum(abs.(sol.zero[idx] - ss.dae.y[idx])) < 1e-6 

            sol = nlsolve(nr_serial!(jss, Val{:threaded}), y0);
            @test maximum(abs.(sol.zero[idx] - ss.dae.y[idx])) < 1e-6 

            sol = nlsolve(nr_threaded!(jss, Val{:serial}), y0);
            @test maximum(abs.(sol.zero[idx] - ss.dae.y[idx])) < 1e-6 

            sol = nlsolve(nr_threaded!(jss, Val{:threaded}), y0);
            @test maximum(abs.(sol.zero[idx] - ss.dae.y[idx])) < 1e-6 
        end


        #=
        @testset "nlsolve with Jac" begin
            ## Tests start - nlsolve with Jacobian
            sol = nlsolve(nr_serial!(jss, Val{:serial}),
                          nr_jac_serial!(jss, Val{:serial}),
                          y0);
            @test maximum(abs.(sol.zero[idx] - ss.dae.y[idx])) < 1e-6 

            sol = nlsolve(nr_serial!(jss, Val{:threaded}),
                          nr_jac_serial!(jss, Val{:threaded}),
                          y0);
            @test maximum(abs.(sol.zero[idx] - ss.dae.y[idx])) < 1e-6 
        end
        =#
    end
end
end
