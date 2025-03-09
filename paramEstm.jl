using Optimization
using OptimizationNLopt
using Statistics
include("liklhdFun.jl")

function pen(x)
    abs(x)
end

function local_estimator(data, mu_initial, gamma_initial, knots; order=3)
    alpha = zeros(size(mu_initial))
    n = size(data, 1)
    p = size(mu_initial, 1)
    data_reorgnz = PIC_data_reorgnz(data, order, knots)

    function object_fun(vars, fixed_values)
        mu = vars[1:p]
        gamma = vars[p+1:end]
        alpha, data_reorgnz = fixed_values
        return -logliklhd_k(mu, alpha, gamma, data_reorgnz) + (n * 0.01 * sum(pen.(mu)))
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
        ftol_rel = 1e-5,
        xtol_abs = 1e-5, 
        maxeval = 1000)
    optimized_mu = sol.u[1:p]
    optimized_mu[abs.(optimized_mu).<=0.01].=0.0
    optimized_gamma = sol.u[p+1:end]

    return optimized_mu, optimized_gamma
end

function multisource_estimator(data, mu_initial, gamma_initial, knots; order=3)
    p = size(mu_initial, 1)
    n = sum(size(df, 1) for df in data)
    data_reorgnz = ()
    k = length(data)
    # alpha_initial = vcat(fill(0.0, p), fill(1.0, p), fill(2.0, p), fill(3.0, p))
    alpha_initial = fill(0.0, p*k)
    data_reorgnz = ntuple(i -> PIC_data_reorgnz(data[i], order, knots), k)

    tun_1 = tun_2 = 0.01
    function object_fun(vars, fixed_values)
        mu = @view vars[1:p]
        gamma = @view vars[(k+1)*p+1:end]
        alpha_flat = @view vars[p+1:p*(k+1)]
        alpha_mat = reshape(alpha_flat, p, k)'

        logliklhd_all = mapreduce(+, 1:k) do i
            logliklhd_k(mu, alpha_mat[i,:], gamma, fixed_values[i])
        end

        pen_1 = n * tun_1 * sum(pen.(mu))
        pen_2 = n * tun_2 * mapreduce(x -> pen(norm(x)), +, eachcol(alpha_mat))
        
        return - logliklhd_all + pen_1 + pen_2
    end

    x0 = vcat(mu_initial, alpha_initial, gamma_initial)
    lb = vcat(fill(-10, p*(k+1)), fill(0, size(gamma_initial)))
    ub = vcat(fill(10, p*(k+1)), fill(10, size(gamma_initial)))
    
    f = OptimizationFunction(object_fun, Optimization.AutoForwardDiff())

    prob = OptimizationProblem(f, x0, data_reorgnz, lb = lb, ub = ub)

    sol = solve(
        prob, 
        NLopt.LD_LBFGS(),
        stopval = 1e-5, 
        ftol_rel = 1e-5,
        xtol_abs = 1e-5, 
        maxeval = 100,
        maxtime = 30)
    
    optimized_mu = sol.u[1:p]
    optimized_mu[abs.(optimized_mu).<=0.01].=0.0
    optimized_alpha = [sol.u[p*i+1:p*(i+1)] for i in 1:k]
    optimized_alpha = reshape(sol.u[p+1:p*(k+1)], p, k)'
    optimized_alpha = optimized_alpha .- mean(optimized_alpha, dims=1)
    optimized_alpha[abs.(optimized_alpha).<=0.01].=0.0
    optimized_gamma = sol.u[(k+1)*p+1:end]

    return optimized_mu, optimized_alpha, optimized_gamma
end

optimized_mu, optimized_alpha, optimized_gamma = multisource_estimator(data, mu_initial, gamma_initial, knots; order=3)
optimized_alpha .+ optimized_mu'
