using Pkg;
Pkg.activate(".");
# Pkg.add(["DataFrames", "LinearAlgebra", "CSV"])
# Pkg.add(["Statistics", "Random", "Distributions",])
# Pkg.add(["Optimization", "OptimizationNLopt", "ForwardDiff"])
# Pkg.add("BenchmarkTools")
# Pkg.instantiate()
# using Chairmarks

using CSV, LinearAlgebra
using BenchmarkTools, ProgressMeter
using PrettyTables
include("liklhdFun.jl")
include("usefulFuncs.jl")
include("multiEst.jl")

k =4;
J0 = 10;
p = 20;
n_1 = 1000
n_2 = 1000
n_3 = 1000
n_4 = 1000
order = 2

real_mu = vcat(0.5, 0.5, 0.3, 0.3, 0.0, 0.0, fill(0.0, p - 6))

real_alpha1 = vcat(0.0, 0.0, 0.3, 0.3, 0.5, 0.5, fill(0.0, p - 6))
real_alpha2 = vcat(0.0, 0.0, -0.3, -0.3, 0.5, 0.5, fill(0.0, p - 6))
real_alpha3 = vcat(0.0, 0.0, -0.3, -0.3, -0.5, -0.5, fill(0.0, p - 6))
real_alpha4 = vcat(0.0, 0.0, 0.3, 0.3, -0.5, -0.5, fill(0.0, p - 6))

real_alpha = hcat(real_alpha1, real_alpha2, real_alpha3, real_alpha4)

real_beta1 = real_mu + real_alpha1
real_beta2 = real_mu + real_alpha2
real_beta3 = real_mu + real_alpha3
real_beta4 = real_mu + real_alpha4
real_beta_matrix = hcat(real_beta1, real_beta2, real_beta3, real_beta4)

real_alpha_norm = map(norm, eachrow(real_alpha))
real_alpha_0 = Int.(iszero.(real_alpha_norm))
real_alpha_1 = Int.(.!iszero.(real_alpha_norm))
true_zero_num_alpha = sum(real_alpha_0)
true_non_zero_num_alpha = sum(real_alpha_1)
real_mu_0 = Int.(iszero.(real_mu))
real_mu_1 = Int.(.!iszero.(real_mu))
true_zero_num_mu = sum(real_mu_0)
true_non_zero_num_mu = sum(real_mu_1)

alpha_initial = hcat(fill(0.00, p), fill(0.00, p), fill(0.00, p), fill(0.00, p)) #fill(0.0, p*k)
mu_initial = fill(0.0, p)
gamma_initial = fill(0.1, J0-2 + order)

loop_num = 100
res = zeros(loop_num, 8)


@showprogress Threads.@threads for i in 1:loop_num
    Random.seed!(2025*i)
    data_1 = gen_pic_data(n_1, 0.2, real_beta1, "case1")
    data_2 = gen_pic_data(n_2, 0.2, real_beta2, "case1")
    data_3 = gen_pic_data(n_3, 0.2, real_beta3, "case1")
    data_4 = gen_pic_data(n_4, 0.2, real_beta4, "case1")
    data = (data_1, data_2, data_3, data_4)
    data_full = vcat(data_1, data_2, data_3, data_4)
    knots = get_knots(vcat(data_full[:, 1], data_full[:, 2]), J0)

    mu_hat, alpha_hat, gamma_hat = multisource_estimator(data, mu_initial, alpha_initial, gamma_initial, knots; spl_order=order, penalty="scad")

    mu_hat_1 = Int.(.!iszero.(mu_hat))
    TP_mu = sum(mu_hat_1.*real_mu_1)
    FP_mu = sum(mu_hat_1.*real_mu_0)

    alpha_hat_norm = map(norm, eachrow(alpha_hat))
    alpha_hat_1 = Int.(.!iszero.(alpha_hat_norm))
    TP_alpha = sum(alpha_hat_1.*real_alpha_1)
    FP_alpha = sum(alpha_hat_1.*real_alpha_0)

    beta_hat = mu_hat .+ alpha_hat

    SE = vec(sum((beta_hat.-real_beta_matrix).^2, dims=1))
    res[i, :] = [SE..., TP_mu, FP_mu, TP_alpha, FP_alpha]
end

pretty_table(mean(res, dims=1))