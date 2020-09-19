@testset "Power flow solutions" begin
    mpc_dir = joinpath(@__DIR__, "mpc_data")    
    cases = ["case5.m", "case14.m", "case39.m", "case118.m", "case300.m"]

    for name in cases
        case_path = joinpath(mpc_dir, name)

        @info "Testing $name"

        ss = Andes.py.load(case_path, no_output=true)
        ss.PFlow.init()

        jss = System{Float64}(ss)
        y0 = jss.dae.y

        ss.PFlow.run()
        @time sol = nlsolve(nr_cb_serial!(jss, Val{:serial}), y0);
        @test maximum(abs.(sol.zero - ss.dae.y)) < 1e-6 

    end
end
