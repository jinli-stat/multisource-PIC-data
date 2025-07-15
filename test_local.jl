using Pkg;
Pkg.activate(".");
# Pkg.add(["DataFrames", "LinearAlgebra", "CSV"])
# Pkg.add(["Statistics", "Random", "Distributions",])
# Pkg.add(["Optimization", "OptimizationNLopt", "ForwardDiff"])
# Pkg.add("BenchmarkTools")
# Pkg.instantiate()
# using Chairmarks

using CSV, LinearAlgebra, BenchmarkTools
include("liklhdFun.jl")
include("usefulFuncs.jl")
include("locEstm.jl")

function simulation(n, real_beta; loop_num=100, J0=10, order=2)
    p = size(real_beta, 1)
    real_beta_1 = Int.(.!iszero.(real_beta))
    real_beta_0 = Int.(iszero.(real_beta))

    res = zeros(loop_num, 3) # SE, TP, FP
    beta_res = zeros(loop_num, p)
    gamma_res = zeros(loop_num, J0 - 2 + order)

    beta_initial = fill(0.00, p)
    gamma_initial = fill(0.1, J0 - 2 + order)

    Threads.@threads for i in 1:loop_num
        data = gen_pic_data(n, 0.2, real_beta, "case1")
        knots = get_knots(vcat(data[:, 1], data[:, 2]), J0)
        beta_ini, gamma_ini = local_estimator(data, beta_initial, gamma_initial, knots; spl_order=order, penalty="none")
        beta_hat, gamma_hat = local_estimator(data, beta_ini, gamma_ini, knots; spl_order=order, penalty="gselo")
        # beta_hat, gamma_hat = local_estimator(data, beta_hat, gamma_hat, knots; spl_order=order, penalty = "scad")
        beta_hat[abs.(beta_hat).<=0.05] .= 0.0
        SE = sum(abs2, beta_hat - real_beta)
        beta_hat_1 = Int.(.!iszero.(beta_hat))
        TP_beta = sum(beta_hat_1 .* real_beta_1)
        FP_beta = sum(beta_hat_1 .* real_beta_0)
        res[i, :] = [SE, TP_beta, FP_beta]
        beta_res[i, :] = beta_hat
        gamma_res[i, :] = gamma_hat
    end
    return (mean(res, dims=1), beta_res, gamma_res)
end

p = 50
n = 1000
real_mu = vcat(0.5, 0.5, 0.3, 0.3, 0.0, 0.0, fill(0.0, p - 6))

real_alpha1 = vcat(0.0, 0.0, 0.3, 0.3, 0.5, 0.5, fill(0.0, p - 6))
real_alpha2 = vcat(0.0, 0.0, -0.3, -0.3, 0.5, 0.5, fill(0.0, p - 6))
real_alpha3 = vcat(0.0, 0.0, -0.3, -0.3, -0.5, -0.5, fill(0.0, p - 6))
real_alpha4 = vcat(0.0, 0.0, 0.3, 0.3, -0.5, -0.5, fill(0.0, p - 6))

real_beta1 = real_mu + real_alpha1 # 0.5, 0.5, 0.6, 0.6, 0.5, 0.5
real_beta2 = real_mu + real_alpha2 # 0.5, 0.5, 0.0, 0.0, 0.5, 0.5
real_beta3 = real_mu + real_alpha3 # 0.5, 0.5, 0.0, 0.0, -0.5, -0.5
real_beta4 = real_mu + real_alpha4 # 0.5, 0.5, 0.6, 0.6, -0.5, -0.5

Random.seed!(2025)

println(simulation(n, real_beta1; J0=10, order=2)[1])
println(simulation(n, real_beta2; J0=10, order=2)[1])
println(simulation(n, real_beta3; J0=10, order=2)[1])
println(simulation(n, real_beta4; J0=10, order=2)[1])



# CSV.write("beta_res.csv", DataFrame(beta_res, :auto))
# CSV.write("res.csv", DataFrame(res, :auto))
