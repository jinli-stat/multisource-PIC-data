using Optimization, OptimizationNLopt, ForwardDiff
using Statistics, LinearAlgebra

function BayIC(data_reorgnz_val, beta_val, gamma_val, n; thsh = 0.2)
    beta_val[abs.(beta_val) .<= thsh] .= 0.0
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

    thsh = 0.1

    function object_fun(vars, fixed_values)
        beta = vars[1:p]
        gamma = vars[p+1:end]
        data_reorgnz, xi = fixed_values
        if penalty == "mic1"
            neg_logliklhd_k = -logliklhd_k(beta .* penalty_fun.(beta, xi), gamma, data_reorgnz)
        else
            neg_logliklhd_k = -logliklhd_k(beta, gamma, data_reorgnz)
        end
        
        if penalty == "none"
            object_fun_val = neg_logliklhd_k
        elseif penalty == "mic1" || penalty == "mic2"
            object_fun_val = neg_logliklhd_k + log(n) * sum(penalty_fun.(beta, xi))
        else
            object_fun_val = neg_logliklhd_k + n * sum(penalty_fun.(beta, xi))
        end
        return safe_value(object_fun_val)
    end

    function evaluate_tuning_param(xi)
        x0 = vcat(beta_initial, gamma_initial)
        adtype = Optimization.AutoForwardDiff()
        f = OptimizationFunction(object_fun, adtype)
        lb = vcat(fill(-5, p), fill(0, size(gamma_initial)))
        ub = vcat(fill(5, p), fill(5, size(gamma_initial)))
        prob = OptimizationProblem(f, x0, (data_reorgnz, xi), lb=lb, ub=ub)

        sol = solve(prob,
            NLopt.LD_SLSQP(),
            # NLopt.LD_LBFGS(), 
            xtol_abs = 1e-5,
            maxeval = 5000)
        if sol.retcode in [:Failure, :UserStop] || any(isnan, sol.u)
            return local_estimator_result(xi, Inf, beta_initial, gamma_initial)
        end
        beta_hat = sol.u[1:p]
        if penalty == "mic1"
            beta_hat = beta_hat .* penalty_fun.(beta_hat, xi)
        end
        gamma_hat = sol.u[p+1:end]
        BIC_val = BayIC(data_reorgnz, beta_hat, gamma_hat, n)

        return local_estimator_result(xi, BIC_val, beta_hat, gamma_hat)
    end

    if penalty == "none"
        this_result = evaluate_tuning_param(0.0)
        return this_result
    elseif penalty == "mic1" || penalty == "mic2"
        results = evaluate_tuning_param(n/5)
        return results
    else
        param_grid = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.15, 0.2]
    end

    # n_combinations = length(param_grid)
    tuning_results = [evaluate_tuning_param(xi_val) for xi_val in param_grid]
    
    # tuning_results = Vector{local_estimator_result}(undef, n_combinations)
    # for i in 1:n_combinations
    #     xi_val = param_grid[i]
    #     tuning_results[i] = evaluate_tuning_param(xi_val)
    # end

    if isempty(tuning_results)
        error("No valid tuning parameters.")
    end

    best_idx = argmin(map(r -> r.criterion, tuning_results))
    best_result = tuning_results[best_idx]
    # println("Best tuning parameter: ", best_result.xi, ", BIC value: ", best_result.criterion)
    return best_result
end