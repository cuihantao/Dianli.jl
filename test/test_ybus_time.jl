using Dianli
using BenchmarkTools

case_path = "/home/hacui/repos/matpower/data/case_ACTIVSg70k.m"

ss = Dianli.Andes.py.load(case_path, no_output=true, default_config=true);
ss.PFlow.init();

jss = System{Float64}(ss);
y0 = Vector(jss.dae.y);

Ybus = Dianli.Ymatrix(jss.Line, jss.Bus);

Vbus = ss.Bus.v.v .* exp.(1im * ss.Bus.a.v);

Sbus = zeros(Complex{Float64}, ss.Bus.n);
Pvec = similar(Vbus);
Qvdc = similar(Vbus);

@btime Dianli.PowerSystem.calc_Yinj!(
    Ybus,
    Vbus,
    Sbus,
);

@btime Dianli.PowerSystem.g_update!(jss.Line, Val{:serial});
