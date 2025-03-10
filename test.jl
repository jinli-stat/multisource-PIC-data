using Optimization
using OptimizationNLopt


include("liklhdFun.jl")
include("simData.jl")
include("paramEstm.jl")

J0 = 10
p =  50
n = 1000

real_mu = vcat(-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, fill(0.0, p-6))
alpha_1 = vcat(0.0, 0.0, 0.5, 0.5, 0.5, 0.5, fill(0.0, p-6))
alpha_2 = vcat(0.0, 0.0, -0.5, -0.5, -0.5, -0.5, fill(0.0, p-6))
alpha_3 = vcat(0.0, 0.0, -0.5, -0.5, -0.5, -0.5, fill(0.0, p-6))
alpha_4 = vcat(0.0, 0.0, 0.5, 0.5, 0.5, 0.5, fill(0.0, p-6))
beta_1 = real_mu + alpha_1
beta_2 = real_mu + alpha_2
beta_3 = real_mu + alpha_3
beta_4 = real_mu + alpha_4

Random.seed!(2025)
data_1 = gen_pic_data(n, 0.2, beta_1, "case1")
data_2 = gen_pic_data(n, 0.2, beta_2, "case1")
data_3 = gen_pic_data(n, 0.2, beta_3, "case1")
data_4 = gen_pic_data(n, 0.2, beta_4, "case1")

data = (data_1, data_2, data_3, data_4)



J0 = 10
p =  10
n = 1000
real_mu = vcat(-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, fill(0.0, p-6))
alpha_1 = vcat(0.0, 0.0, 0.5, 0.5, 0.5, 0.5, fill(0.0, p-6))
alpha_2 = vcat(0.0, 0.0, -0.5, -0.5, -0.5, -0.5, fill(0.0, p-6))
alpha_3 = vcat(0.0, 0.0, -0.5, -0.5, -0.5, -0.5, fill(0.0, p-6))
alpha_4 = vcat(0.0, 0.0, 0.5, 0.5, 0.5, 0.5, fill(0.0, p-6))
beta_1 = real_mu + alpha_1
beta_2 = real_mu + alpha_2
beta_3 = real_mu + alpha_3
beta_4 = real_mu + alpha_4
@elapsed begin
    for s in 1:100
        Random.seed!(s)
        data_1 = gen_pic_data(n, 0.2, beta_1, "case1")
        knots =  initial_knots(union(data_1[:,1], data_1[:,2], 0.0, Inf), J0)[1:end-1]
        spl_order = 3
        mu_initial = fill(0.0, p)
        gamma_initial = fill(0.1, J0-2+spl_order)
        mu_0, gamma_0 = local_estimator(data_1, mu_initial, gamma_initial, knots; spl_order=spl_order)
        mu_1, gamma_1 = local_estimator(data_1, mu_0, gamma_0, knots; spl_order = spl_order, penalty = true)
    end
end



Random.seed!(2025)
data_1 = gen_pic_data(n, 0.2, beta_1, "case1")
data_2 = gen_pic_data(n, 0.2, beta_2, "case1")
data_3 = gen_pic_data(n, 0.2, beta_3, "case1")
data_4 = gen_pic_data(n, 0.2, beta_4, "case1")

data = (data_1, data_2, data_3, data_4)
data_full = vcat(data_1, data_2, data_3, data_4)
k=length(data)
alpha_initial = fill(0.0, p*k)
knots =  initial_knots(union(data_full[:,1], data_full[:,2], 0.0, Inf), J0)[1:end-1]
mu_initial = fill(0.0, p)
spl_order = 3
gamma_initial = fill(0.1, J0-2 + spl_order)
mu_0, alpha_0, gamma_0 = multisource_estimator(data, mu_initial, alpha_initial, gamma_initial, knots; spl_order=3, penalty = false)
alpha_0 = vec(alpha_0')
mu_1, alpha_1, gamma_1 = multisource_estimator(data, mu_0, alpha_0, gamma_0, knots; spl_order=3, penalty = true)
mu_1' .+ alpha_1