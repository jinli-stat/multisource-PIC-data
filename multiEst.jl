using Optimization, OptimizationNLopt, ForwardDiff
using Statistics, LinearAlgebra, Logging

function BayIC2(data_reorgnz_val, mu_val, alpha_val, gamma_val, n, k)
    logliklhd_all = sum(1:k) do i
            logliklhd_k(mu_val + alpha_val[:, i], gamma_val, data_reorgnz_val[i])
    end

    non_zero_mu_count = count(!iszero, mu_val)
    non_zero_alpha_count = count(!iszero, alpha_val[:, 1:end-1])
    # non_zero_alpha_count  = count(!iszero, map(norm, eachrow(alpha_val)))

    DF = non_zero_mu_count + non_zero_alpha_count + size(gamma_val)[1]
    criterion = -2 * logliklhd_all + DF * (log(n) + 2 * log(size(mu_val)[1]))
    return criterion, DF
end

struct optimization_result
    xi_1::Float64
    xi_2::Float64
    criterion::Float64
    DF::Float64
    mu::Vector{Float64}
    alpha::AbstractArray{Float64}
    gamma::Vector{Float64}
end

function multisource_estimator(data, mu_initial, alpha_initial, gamma_initial, knots; 
                                spl_order=2, penalty="none")
    p = size(mu_initial, 1)
    n = sum(size(df, 1) for df in data)
    k = length(data)
    J0 = size(gamma_initial)[1]
    alpha_initial = vec(alpha_initial)

    data_reorgnz = ntuple(i -> PIC_data_reorgnz(data[i], spl_order, knots), k)

    if penalty == "none"
        # Do nothing
    elseif penalty == "scad"
        penalty_fun = penalty_scad
    elseif penalty == "mcp"
        penalty_fun = penalty_mcp
    elseif penalty == "mic"
        penalty_fun = penalty_mic
    else
        error("Wrong name of penalty function!")
    end
    if  penalty == "mic"
        thsh = 0.2
    else
        thsh = 0.1
    end

    function object_fun(vars, fixed_val)

        mu = vars[1:p]
        alpha_mat = reshape(vars[p+1:(k+1)*p], p, k)
        gamma = vars[(k+1)*p+1:end]
        all_data, xi_1, xi_2 = fixed_val

        logliklhd_all = sum(1:k) do i
            logliklhd_k(mu + alpha_mat[:, i], gamma, all_data[i])
        end

        if penalty == "none"
            return -logliklhd_all
        else 
            pen_1 = n * sum(penalty_fun.(mu, xi_1))
            alpha_norm = map(norm, eachrow(alpha_mat))
            pen_2 = n * sum(penalty_fun.(alpha_norm, xi_2))
            return -logliklhd_all + pen_1 + pen_2
        end
    end

    function equality_constraint(res, vars, fixed_val)
        res .= sum(reshape(vars[p+1:(k+1)*p], p, k), dims=2)
        return nothing
    end

    function evaluate_tuning_param(xi_1, xi_2)
        x0 = vcat(mu_initial, alpha_initial, gamma_initial)
        adtype = Optimization.AutoForwardDiff()
        f = OptimizationFunction(
            object_fun, 
            adtype;
            cons = equality_constraint)
        lb = vcat(fill(-5, p * (k+1)), fill(0, size(gamma_initial)))
        ub = vcat(fill(5, p * (k+1)), fill(5, size(gamma_initial)))
        constraint_bounds = zeros(p)
        prob = OptimizationProblem(
            f, x0, (data_reorgnz, xi_1, xi_2), 
            lb=lb, 
            ub=ub,
            lcons = constraint_bounds,
            ucons = constraint_bounds)
        sol = solve(
            prob,
            NLopt.LD_SLSQP(), # NLopt.LD_SLSQP(), NLopt.LD_AUGLAG()
            xtol_abs=1e-3,
            # ftol_abs=0.5,
            maxeval = 5000)

        mu_hat = sol.u[1:p]
    
        mu_hat[abs.(mu_hat).<=thsh] .= 0.0
        alpha_hat = sol.u[p+1:p*(k+1)]
        
        alpha_hat = reshape(alpha_hat, p, k)
        alpha_hat[abs.(alpha_hat).<=thsh] .= 0.0
        gamma_hat = sol.u[(k+1)*p+1:end]
        criterion, DF = BayIC2(data_reorgnz, mu_hat, alpha_hat, gamma_hat, n, k)
        return optimization_result(xi_1, xi_2, criterion, DF, mu_hat, alpha_hat, gamma_hat)
    end

    if penalty == "none"
        this_result = evaluate_tuning_param(1.0, 1.0)
        return this_result.mu, this_result.alpha, this_result.gamma
    elseif penalty == "mic"
        this_result = evaluate_tuning_param(n, n)
        return this_result.mu, this_result.alpha, this_result.gamma
    else
        param1 = [0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.15]
        param2 = [0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.15]
    end
    param_grid = collect(Base.Iterators.product(param1, param2)) |> vec
    n_combinations = length(param_grid)
    tuning_results = Vector{optimization_result}(undef, n_combinations)
    for i in 1:n_combinations
        xi_1, xi_2 = param_grid[i]
        tuning_results[i] = evaluate_tuning_param(xi_1, xi_2)
        # println("$xi_1, $xi_2, $(tuning_results[i].criterion), $(tuning_results[i].DF)")
    end

    filter!(x -> x.DF < J0+(k-1)*p, tuning_results)
    best_idx = argmin(map(r -> r.criterion, tuning_results))
    best_result = tuning_results[best_idx]
    result_mu = best_result.mu
    result_alpha = best_result.alpha
    return result_mu, result_alpha, best_result.gamma
end
