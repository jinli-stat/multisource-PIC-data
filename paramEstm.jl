using Optimization
using OptimizationNLopt
include("liklhdFun.jl")

function local_estimator(data, mu_initial, gamma_initial, knots; order=3)
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

function multisource_estimator(data, mu_initial, gamma_initial, knots; order=3)
    
    p = size(mu_initial, 1)
    data_reorgnz = ()
    k = length(data)

    alpha_initial = fill(0.0, p*k)
    for i in 1:k
        data_reorgnz = (data_reorgnz..., PIC_data_reorgnz(data[i], order, knots))
    end
    
    function object_fun(vars, fixed_values)
        mu = vars[1:p]
        gamma = vars[(k+1)*p+1:end]
        alpha = [vars[p*i+1:p*(i+1)] for i in 1:k]

        logliklhd_all = 0
        for i in 1:k
            logliklhd_all += logliklhd_k(mu, alpha[i], gamma, fixed_values[i])
        end
        return - logliklhd_all
    end

    x0 = vcat(mu_initial, alpha_initial, gamma_initial)
    f = OptimizationFunction(object_fun, Optimization.AutoForwardDiff())
    lb = vcat(fill(-10, p*(k+1)), fill(0, size(gamma_initial)))
    ub = vcat(fill(10, p*(k+1)), fill(10, size(gamma_initial)))
    prob = OptimizationProblem(f, x0, data_reorgnz, lb = lb, ub = ub)

    sol = solve(
        prob, 
        NLopt.LD_LBFGS(),
        stopval = 1e-5, 
        xtol_abs = 1e-5, 
        maxeval = 100)
    
    optimized_mu = sol.u[1:p]
    optimized_alpha = [sol.u[p*i+1:p*(i+1)] for i in 1:k]
    optimized_gamma = sol.u[(k+1)*p+1:end]

    return optimized_mu, optimized_alpha, optimized_gamma
end