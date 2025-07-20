using Optimization, OptimizationNLopt, ForwardDiff
using Statistics, LinearAlgebra

function BayIC(data_reorgnz_val, beta_val, gamma_val, n)
    lkhd = logliklhd_k(beta_val, gamma_val, data_reorgnz_val)
    DF = count(!iszero, beta_val) + 1 * size(gamma_val)[1] # Here should be the number of basis functions, using the size of gamma for simplicity.
    return -2 * lkhd + DF * (log(n) + 2 * log(size(beta_val)[1]))
end

struct local_estimator_result
    xi::Float64
    criterion::Float64
    beta::Vector{Float64}
    gamma::Vector{Float64}
end

function local_estimator(data, beta_initial, gamma_initial, knots; spl_order=2, penalty="none")
    n = size(data, 1)
    p = size(beta_initial, 1)
    data_reorgnz = PIC_data_reorgnz(data, spl_order, knots)

    if penalty == "none"
        # Do nothing
    elseif penalty == "gselo"
        penalty_fun = penalty_gselo
    elseif penalty == "scad"
        penalty_fun = penalty_scad
    elseif penalty == "mcp"
        penalty_fun = penalty_mcp
    elseif penalty == "mic1" || penalty == "mic2"
        penalty_fun = penalty_mic
    else
        error("Wrong name of penalty function!")
    end

    function object_fun(vars, fixed_values)
        beta = vars[1:p]
        beta[abs.(beta).<=0.01] .= 0.0
        gamma = vars[p+1:end]
        data_reorgnz, xi = fixed_values
        if penalty == "none"
            object_fun_val = -logliklhd_k(beta, gamma, data_reorgnz)
        elseif penalty == "mic1"
            object_fun_val = return -logliklhd_k(beta .* penalty_fun.(beta, xi), gamma, data_reorgnz) + log(n) * sum(penalty_fun.(beta, xi))
        elseif penalty == "mic2"
            object_fun_val = return -logliklhd_k(beta, gamma, data_reorgnz) + n * sum(penalty_fun.(beta, xi))
        else
            object_fun_val = return -logliklhd_k(beta, gamma, data_reorgnz) + n * sum(penalty_fun.(beta, xi))
        end
        return object_fun_val
    end

    function evaluate_tuning_param(xi)
        x0 = vcat(beta_initial, gamma_initial)
        f = OptimizationFunction(object_fun, Optimization.AutoForwardDiff())
        lb = vcat(fill(-5, p), fill(0, size(gamma_initial)))
        ub = vcat(fill(5, p), fill(5, size(gamma_initial)))
        prob = OptimizationProblem(f, x0, (data_reorgnz, xi), lb=lb, ub=ub)

        sol = solve(prob,
            NLopt.LD_LBFGS(),
            xtol_abs=1e-3,
            maxeval=10000)

        beta_hat = sol.u[1:p]
        if penalty == "mic1"
            beta_hat = beta_hat .* penalty_fun.(beta_hat, xi)
        end
        beta_hat[abs.(beta_hat).<=0.001] .= 0.0
        gamma_hat = sol.u[p+1:end]
        BIC_val = BayIC(data_reorgnz, beta_hat, gamma_hat, n)

        return local_estimator_result(xi, BIC_val, beta_hat, gamma_hat)
    end

    if penalty == "none"
        this_result = evaluate_tuning_param(0.0)
        return this_result.beta, this_result.gamma
    elseif penalty == "mic1" || penalty == "mic2"
        results = evaluate_tuning_param(n)
        return results.beta, results.gamma
    else
        param_grid = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.15]
        # param_grid = [0.05, 0.07, 0.09, 0.1]
        # param_grid = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    end

    n_combinations = length(param_grid)
    tuning_results = Vector{local_estimator_result}(undef, n_combinations)
    for i in 1:n_combinations
        xi_val = param_grid[i]
        tuning_results[i] = evaluate_tuning_param(xi_val)
    end
    best_idx = argmin(map(r -> r.criterion, tuning_results))
    best_result = tuning_results[best_idx]
    # println("Best tuning parameter: ", best_result.xi, ", BIC value: ", best_result.criterion)
    return best_result.beta, best_result.gamma
end