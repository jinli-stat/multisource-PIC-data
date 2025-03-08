using Optimization
using OptimizationNLopt
import ForwardDiff

include("liklhdFun.jl")
include("simData.jl")
include("paramEstm.jl")
n = 5000;
p = 5;
J0 = 10;
real_beta = vcat([0.9, -0.7, 0, 0, 0.5], zeros(p-5));
e_rate = 0.2;
Random.seed!(2025)
data_1 = gen_pic_data(n, e_rate, real_beta, "case1")
knots =  initial_knots(union(data_1[:,1], data_1[:,2], 0.0, Inf), J0)[1:end-1]

order = 2
mu_initial = fill(0.1, p)
gamma_initial = fill(0.01, J0-2+order)

optimized_mu, optimized_gamma = one_pass_estimation(data_1, mu_initial, gamma_initial, knots; order=2)
optimized_mu
optimized_gamma