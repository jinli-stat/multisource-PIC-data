using Optimization
using OptimizationNLopt
include("liklhdFun.jl")

function one_pass_estimation(data, mu_initial, gamma_initial, knots; order=3)
    alpha = zeros(size(mu_initial))
    p = size(mu_initial, 1)
    data_reorgnz = PIC_data_reorgnz(data, order, knots)

    function object_fun(vars, fixed_values)
        mu = vars[1:p]
        gamma = vars[p+1:end]
        alpha, data_reorgnz = fixed_values
        return -logliklhd_k(mu, alpha, gamma, data_reorgnz)
    end
    x0 = vcat(mu_initial, gamma_initial)
    f = OptimizationFunction(object_fun, Optimization.AutoForwardDiff())
    lb = vcat(fill(-Inf, p), fill(0, size(gamma_initial)))
    ub = vcat(fill(Inf, p), fill(Inf, size(gamma_initial)))
    prob = OptimizationProblem(f, x0, (alpha, data_reorgnz), lb = lb, ub = ub)

    sol = solve(
        prob, 
        NLopt.LD_LBFGS(),
        stopval = 1e-5, 
        xtol_abs = 1e-5, 
        maxeval = 100)
    optimized_mu = sol.u[1:p]
    optimized_gamma = sol.u[p+1:end]

    return optimized_mu, optimized_gamma
end