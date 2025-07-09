using Optimization, OptimizationNLopt, ForwardDiff
using Statistics, LinearAlgebra
# using ReverseDiff

function penalty_scad(beta_hat, tuning_param)
    a_val = 3.7
    abs_beta_hat = abs(beta_hat)
    if abs_beta_hat <= tuning_param
        return tuning_param * abs_beta_hat
    elseif tuning_param < abs_beta_hat <= a_val * tuning_param
        return (2 * a_val * tuning_param * abs_beta_hat - beta_hat^2 - tuning_param^2) / (2 * (a_val - 1))
    else
        return (tuning_param^2 * (a_val + 1)) / 2
    end
end

function penalty_mcp(beta_hat, tuning_param)
    a = 2.8
    abs_beta = abs(beta_hat)
    if abs_beta <= a * tuning_param
        return tuning_param * abs_beta - abs_beta^2 / (2 * a)
    else
        return (tuning_param^2 * a) / 2
    end
end

function penalty_gselo(beta_hat, tuning_param)
    return 1 - safe_exp(- tuning_param * (beta_hat^2))
end

function penalty_mic(beta_hat, tuning_param)
    temp_val = safe_exp(2 * tuning_param * beta_hat^2)
    return (temp_val - 1.0) / (temp_val + 1.0)
end

function BayIC(data_reorgnz_val, beta_val, gamma_val, n)
    lkhd = logliklhd_k(beta_val, gamma_val, data_reorgnz_val)
    DF = count(!iszero, beta_val) + size(gamma_val)[1] # Here should be the number of basis functions, using the size of gamma for simplicity.
    return - 2*lkhd + DF *(log(n) + 2*log(size(beta_val)[1]))
end

function BayIC2(data_reorgnz_val, mu_val, alpha_val, gamma_val, n)
    logliklhd_all = mapreduce(+, 1:k) do i
        logliklhd_k(mu_val + alpha_val[i,:], gamma_val, data_reorgnz_val[i])
    end
    non_zero_mu_count = count(!iszero, mu_val)
    alpha_val_norm = map(norm, eachcol(alpha_val))
    non_zero_alpha_count = count(!iszero, alpha_val_norm)
    return - 2*logliklhd_all + (non_zero_mu_count + non_zero_alpha_count + size(gamma_val)[1])*(log(n)+ 2*log(2 * size(mu_val)[1]))
end


struct optimization_result
    xi_1::Float64
    xi_2::Float64
    criterion::Float64
    mu::Vector{Float64}
    alpha::AbstractArray{Float64}
    gamma::Vector{Float64}
end

struct local_estimator_result
    xi::Float64
    criterion::Float64
    beta::Vector{Float64}
    gamma::Vector{Float64}
end

function local_estimator(data, beta_initial, gamma_initial, knots; spl_order=2, penalty = "none")
    n = size(data, 1)
    p = size(beta_initial, 1)
    data_reorgnz = PIC_data_reorgnz(data, spl_order, knots)

    if penalty =="none"
        # Do nothing
    elseif penalty == "gselo"
        penalty_fun = penalty_gselo
    elseif penalty == "scad"
        penalty_fun = penalty_scad
    elseif penalty == "mcp"
        penalty_fun = penalty_mcp
    elseif penalty == "mic"
        penalty_fun = penalty_mic
    else
        error("Wrong name of penalty function!")
    end
    
    function object_fun(vars, fixed_values)
        beta = vars[1:p]
        gamma = vars[p+1:end]
        data_reorgnz, xi = fixed_values
        if penalty == "none"
            object_fun_val = -logliklhd_k(beta, gamma, data_reorgnz)
        elseif penalty == "mic"
            object_fun_val = return -logliklhd_k(beta .* penalty_fun.(beta, xi), gamma, data_reorgnz) + log(n) * sum(penalty_fun.(beta, xi))
        elseif penalty == "gselo"
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
        prob = OptimizationProblem(f, x0, (data_reorgnz, xi), lb = lb, ub = ub)

        sol = solve(prob, 
            NLopt.LD_LBFGS(), 
            stopval = 1e-3, 
            ftol_rel = 1e-3, 
            xtol_abs = 1e-3, 
            maxeval = 5000)
            
        beta_hat = sol.u[1:p]
        if penalty == "mic"
            beta_hat = beta_hat.* penalty_fun.(beta_hat, xi)
        end
        beta_hat[abs.(beta_hat).<=0.001] .= 0.0
        gamma_hat = sol.u[p+1:end]
        BIC_val = BayIC(data_reorgnz, beta_hat, gamma_hat, n)

        return local_estimator_result(xi, BIC_val, beta_hat, gamma_hat)
    end

    if penalty == "none"
        this_result = evaluate_tuning_param(0.0)
        return this_result.beta, this_result.gamma
    elseif penalty == "gselo"
        results = evaluate_tuning_param(n/7)
        return results.beta, results.gamma
    elseif penalty == "mic"
        results = evaluate_tuning_param(log(n))
        return results.beta, results.gamma
    else
        param_grid = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
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

    return best_result.beta, best_result.gamma
end

function multisource_estimator(data, mu_initial, alpha_initial, gamma_initial, knots; spl_order=3, penalty = "none")
    p = size(mu_initial, 1)
    n = sum(size(df, 1) for df in data)
    k = length(data) 
    alpha_initial = alpha_initial[1:(end-p)]
    data_reorgnz = ()
    data_reorgnz = ntuple(i -> PIC_data_reorgnz(data[i], spl_order, knots), k)

    if penalty =="none"
        # Do nothing
    elseif penalty == "gselo"
        penalty_fun = penalty_gselo
    elseif penalty == "scad"
        penalty_fun = penalty_scad
    elseif penalty == "mcp"302
        penalty_fun = penalty_mcp
    elseif penalty == "mic"
        penalty_fun = penalty_mic
    else
        error("Wrong name of penalty function!")
    end

    function object_fun(vars, fixed_val)
        mu = vars[1:p]
        gamma = vars[k*p+1:end]
        alpha_flat = vars[p+1:p*k]
        alpha_mat = reshape(alpha_flat, p, (k-1))'
        alpha_mat = vcat(alpha_mat, .-sum(alpha_mat, dims=1))
        all_data, xi_1, xi_2 = fixed_val

        if penalty == "mic"
            mu_pz = mu .* penalty_fun.(mu, xi_1)
            logliklhd_all = mapreduce(+, 1:k) do i
                logliklhd_k(mu_pz + (alpha_mat[i,:] .* penalty_fun.(alpha_mat[i,:], xi_2)), gamma, all_data[i])
            end
        else
            logliklhd_all = mapreduce(+, 1:k) do i
                logliklhd_k(mu + alpha_mat[i,:], gamma, all_data[i])
            end
        end

        if penalty == "none"
            return -logliklhd_all
        elseif penalty == "mic"
            pen_1 = log(n) * sum(penalty_fun.(mu, xi_1))
            alpha_norm = map(norm, eachcol(alpha_mat))
            pen_2 = log(n) * sum(penalty_fun.(alpha_norm,  xi_2))
            return -logliklhd_all + pen_1 + pen_2
        elseif penalty == "gselo"
            pen_1 = log(n) * sum(penalty_fun.(mu, xi_1))
            alpha_norm = map(norm, eachcol(alpha_mat))
            pen_2 = log(n) * sum(penalty_fun.(alpha_norm,  xi_2))
            return -logliklhd_all + pen_1 + pen_2
        else
            pen_1 = n * sum(penalty_fun.(mu, xi_1))
            alpha_norm = map(norm, eachcol(alpha_mat))
            pen_2 = n * sum(penalty_fun.(alpha_norm,  xi_2))
            return -logliklhd_all + pen_1 + pen_2
        end 
    end

    function evaluate_tuning_param(xi_1, xi_2)
        x0 = vcat(mu_initial, alpha_initial, gamma_initial)
        f = OptimizationFunction(object_fun, Optimization.AutoForwardDiff()) # AutoReverseDiff # AutoForwardDiff
        lb = vcat(fill(-5, p*k), fill(0, size(gamma_initial)))
        ub = vcat(fill(5, p*k), fill(5, size(gamma_initial)))
        prob = OptimizationProblem(f, x0, (data_reorgnz, xi_1, xi_2), lb = lb, ub = ub)

        sol = solve(
            prob, 
            NLopt.LD_LBFGS(),
            stopval = 1e-5, 
            ftol_rel = 1e-5,
            xtol_abs = 1e-5, 
            maxeval = 5000)
        
        mu_hat = sol.u[1:p]
        if penalty == "mic"
            mu_hat = mu_hat.* penalty_fun.(mu_hat, xi_1)
        end
        mu_hat[abs.(mu_hat).<=1e-5].=0.0
        alpha_hat = sol.u[p+1:p*k]
        if penalty == "mic"
            alpha_hat = alpha_hat.* penalty_fun.(alpha_hat, xi_2)
        end
        alpha_hat = reshape(alpha_hat, p, (k-1))'
        alpha_hat[abs.(alpha_hat).<=1e-5].=0.0
        alpha_hat = vcat(alpha_hat, .-sum(alpha_hat, dims=1))
        gamma_hat = sol.u[k*p+1:end]
        DF = BayIC2(data_reorgnz, mu_hat, alpha_hat, gamma_hat, n)
        return optimization_result(xi_1, xi_2, DF, mu_hat, alpha_hat, gamma_hat)
    end

    if penalty == "none"
        this_result = evaluate_tuning_param(1.0, 1.0)
        return this_result.mu, this_result.alpha ,this_result.gamma
    elseif penalty == "gselo"
        this_result = evaluate_tuning_param(n, n)
        return this_result.mu, this_result.alpha ,this_result.gamma
    elseif penalty == "mic"
        this_result = evaluate_tuning_param(log(n), log(n))
        return this_result.mu, this_result.alpha ,this_result.gamma
    else
        param1 = [0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.13, 0.25, 0.5]
        param2 = [0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.13, 0.25, 0.5]
    end
    param_grid = collect(Base.Iterators.product(param1, param2)) |> vec
    n_combinations = length(param_grid)
    tuning_results = Vector{optimization_result}(undef, n_combinations)
    for i in 1:n_combinations
        xi_1, xi_2 = param_grid[i]
        tuning_results[i] = evaluate_tuning_param(xi_1, xi_2)
    end
    
    best_idx = argmin(map(r -> r.criterion, tuning_results))
    best_result = tuning_results[best_idx]
    result_mu = best_result.mu
    result_mu[abs.(result_mu).<=0.01].=0.0
    result_alpha = best_result.alpha
    result_alpha[abs.(result_alpha).<=0.02].=0.0
    return result_mu, result_alpha, best_result.gamma
end