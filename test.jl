using Optimization
using OptimizationNLopt
import ForwardDiff

include("liklhdFun.jl")
include("simData.jl")
include("paramEstm.jl")

J0 = 10
p =  50
n = 1000

mu = vcat(-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, fill(0.0, p-6))
alpha_1 = vcat(0.0, 0.0, 0.5, 0.5, 0.5, 0.5, fill(0.0, p-6))
alpha_2 = vcat(0.0, 0.0, -0.5, -0.5, -0.5, -0.5, fill(0.0, p-6))
alpha_3 = vcat(0.0, 0.0, -0.5, -0.5, -0.5, -0.5, fill(0.0, p-6))
alpha_4 = vcat(0.0, 0.0, 0.5, 0.5, 0.5, 0.5, fill(0.0, p-6))
beta_1 = mu + alpha_1
beta_2 = mu + alpha_2
beta_3 = mu + alpha_3
beta_4 = mu + alpha_4

Random.seed!(2025)
data_1 = gen_pic_data(n, 0.2, beta_1, "case1")
data_2 = gen_pic_data(n, 0.2, beta_2, "case1")
data_3 = gen_pic_data(n, 0.2, beta_3, "case1")
data_4 = gen_pic_data(n, 0.2, beta_4, "case1")

data = (data_1, data_2, data_3, data_4)
data_full = vcat(data_1, data_2, data_3, data_4)
length(data)
knots =  initial_knots(union(data_full[:,1], data_full[:,2], 0.0, Inf), J0)[1:end-1]

order = 3
mu_initial = fill(0.0, p)
gamma_initial = fill(0.01, J0-2+order)

optimized_mu, optimized_gamma = local_estimator(data_full, mu_initial, gamma_initial, knots; order=order)
optimized_mu
optimized_gamma



