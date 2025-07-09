using Pkg; Pkg.activate(".")

# Pkg.add(["DataFrames", "LinearAlgebra", "CSV"])
# Pkg.add(["Statistics", "Random", "Distributions",])
# Pkg.add(["Optimization", "OptimizationNLopt", "ForwardDiff"])

# Pkg.instantiate()
# using Chairmarks
using CSV, LinearAlgebra
include("liklhdFun.jl")
include("paramEstm.jl")
include("usefulFuncs.jl")

J0 =10;
p = 50;
n = 500;
order = 2;
real_beta = vcat(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, fill(0.0, p-6));
# vcat(0.5, 0.5, 1, 1, 0.5, 0.5, fill(0.0, p-6))
# vcat(0.5, 0.5, 0, 0, -0.5, -0.5, fill(0.0, p-6))
# vcat(0.5, 0.5, 0, 0, -0.5, -0.5, fill(0.0, p-6))
# vcat(0.5, 0.5, 1, 1, 0.5, 0.5, fill(0.0, p-6))

real_beta_0 = Int.(iszero.(real_beta));
real_beta_1 = Int.(.!iszero.(real_beta));
loop_num = 50;

res = zeros(loop_num, 3); # SE, TP, FP
beta_res = zeros(loop_num, p);
gamma_res = zeros(loop_num, J0-2 + order) ; 

beta_initial = fill(0.00, p);
gamma_initial = fill(0.1, J0-2 + order);

Random.seed!(1234)
@time Threads.@threads for i in 1:loop_num
    data = gen_pic_data(n, 0.2, real_beta, "case1")
    knots = get_knots(vcat(data[:,1], data[:,2]), J0)
    beta_ini, gamma_ini = local_estimator(data, beta_initial, gamma_initial, knots; spl_order=order, penalty = "none")
    beta_hat, gamma_hat = local_estimator(data, beta_ini, gamma_ini, knots; spl_order=order, penalty = "mic")
    # beta_hat, gamma_hat = local_estimator(data, beta_hat, gamma_hat, knots; spl_order=order, penalty = "scad")
    beta_hat[abs.(beta_hat).<=0.1] .= 0.0
    SE = sum(abs2, beta_hat - real_beta)
    beta_hat_1 = Int.(.!iszero.(beta_hat))
    TP_beta = sum(beta_hat_1.*real_beta_1)
    FP_beta = sum(beta_hat_1.*real_beta_0)
    res[i, :] = [SE, TP_beta, FP_beta]
    beta_res[i, :] = beta_hat
    gamma_res[i, :] = gamma_hat
end
mean(res, dims=1)

CSV.write("beta_res.csv", DataFrame(beta_res, :auto))
CSV.write("res.csv", DataFrame(res, :auto))