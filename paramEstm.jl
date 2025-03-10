using Optimization, OptimizationNLopt, ForwardDiff
using Statistics, LinearAlgebra

include("liklhdFun.jl")
function scad_penalty(beta_hat, tuning_param)
    a_val = 3.7
    if abs.(beta_hat) <= tuning_param
        return tuning_param * abs(beta_hat)
    elseif tuning_param < abs(beta_hat) <= a_val * tuning_param
        return (2 * a_val * tuning_param * abs(beta_hat) - beta_hat^2 - tuning_param^2) / (2 * (a_val - 1))
    else
        return (tuning_param^2 * (a_val + 1)) / 2
    end
end

function scad_derivative(beta_hat, tuning_param)
    a_val = 3.7
    return tuning_param * (
        (beta_hat <= tuning_param) +
        ((a_val * tuning_param - beta_hat) * ((a_val * tuning_param - beta_hat) > 0)) / ((a_val - 1) * tuning_param) *
        (beta_hat > tuning_param)
    )
end

function scad_quadratic_approx(beta_hat, beta_0, tuning_param)
    scad_penalty(beta_0, tuning_param) + 0.5*scad_derivative(beta_0, tuning_param)*(beta_hat^2 - beta_0^2)/(abs(beta_0) + 1e-10)
end

function BayIC(data_reorgnz_val, mu_val, alpha_val, gamma_val, n)
    lkhd = logliklhd_k(mu_val, alpha_val, gamma_val, data_reorgnz_val)
    non_zero_mu_count = count(x -> x != 0, mu_val)
    return - 2*lkhd + non_zero_mu_count*log(n)
end

function DF_val(data_reorgnz_val, mu_val, alpha_val, gamma_val, n)
    logliklhd_all = mapreduce(+, 1:k) do i
        logliklhd_k(mu_val, alpha_val[i,:], gamma_val, data_reorgnz_val[i])
    end
    non_zero_mu_count = count(x -> x != 0, mu_val)
    non_zero_alpha_count = count(x -> x != 0, vec(alpha_val[2:end,:]))
    return - 2*logliklhd_all + (non_zero_mu_count + non_zero_alpha_count)*log(n)
end

struct optimization_result
    xi_1::Float64
    xi_2::Float64
    criterion::Float64
    mu::Vector{Float64}
    alpha::AbstractArray{Float64}
    gamma::Vector{Float64}
end

function local_estimator(data, mu_initial, gamma_initial, knots; spl_order=3, penalty = false)
    alpha = zeros(size(mu_initial))
    n = size(data, 1)
    p = size(mu_initial, 1)
    data_reorgnz = PIC_data_reorgnz(data, spl_order, knots)
    xi = 0.01

    function object_fun(vars, fixed_values)
        mu = vars[1:p]
        gamma = vars[p+1:end]
        alpha, data_reorgnz, xi = fixed_values
        if penalty == true
            return -logliklhd_k(mu, alpha, gamma, data_reorgnz) + n * sum(scad_penalty.(mu, xi)) #sum(scad_quadratic_approx.(mu, mu_initial, xi))
        else
            return -logliklhd_k(mu, alpha, gamma, data_reorgnz)
        end
    end

    function evaluate_tuning_param(xi)
        x0 = vcat(mu_initial, gamma_initial)
        f = OptimizationFunction(object_fun, Optimization.AutoForwardDiff())
        lb = vcat(fill(-10, p), fill(0, size(gamma_initial)))
        ub = vcat(fill(10, p), fill(10, size(gamma_initial)))
        prob = OptimizationProblem(f, x0, (alpha, data_reorgnz, xi), lb = lb, ub = ub)

        sol = solve(prob, 
            NLopt.LD_LBFGS(), 
            stopval = 1e-5, 
            ftol_rel = 1e-5, 
            xtol_abs = 1e-5, 
            maxeval = 1000)
        mu_hat = sol.u[1:p]
        mu_hat[abs.(mu_hat).<=0.01] .= 0.0
        gamma_hat = sol.u[p+1:end]
        BIC_val = BayIC(data_reorgnz, mu_hat, alpha, gamma_hat, n)

        return optimization_result(xi, xi, BIC_val, mu_hat, alpha, gamma_hat)
    end
    if penalty == false
        this_result = evaluate_tuning_param(1.0)
        return this_result.mu, this_result.gamma
    end
    
    param_grid = [0.0001,0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0]
    n_combinations = length(param_grid)
    tuning_results = Vector{optimization_result}(undef, n_combinations)
    for i in 1:n_combinations
        xi_val = param_grid[i]
        tuning_results[i] = evaluate_tuning_param(xi_val)
    end
    best_idx = argmin(map(r -> r.criterion, tuning_results))
    best_result = tuning_results[best_idx]

    return best_result.mu, best_result.gamma
end

function multisource_estimator(data, mu_initial, alpha_initial, gamma_initial, knots; spl_order=3, penalty = false)
    p = size(mu_initial, 1)
    n = sum(size(df, 1) for df in data)
    data_reorgnz = ()
    k = length(data) 
    data_reorgnz = ntuple(i -> PIC_data_reorgnz(data[i], spl_order, knots), k)

    function object_fun(vars, fixed_val)
        mu = @view vars[1:p]
        gamma = @view vars[(k+1)*p+1:end]
        alpha_flat = @view vars[p+1:p*(k+1)]
        alpha_mat = reshape(alpha_flat, p, k)'
        fixed_values, xi_1, xi_2 = fixed_val
        logliklhd_all = mapreduce(+, 1:k) do i
            logliklhd_k(mu, alpha_mat[i,:], gamma, fixed_values[i])
        end
        
        if penalty == true
            pen_1 = n * sum(scad_penalty.(mu, xi_1))# sum(scad_quadratic_approx.(mu, mu_initial, xi_1))
            # alpha_alpha_initial_norm = map(norm, eachcol(alpha_initial))
            alpha_norm = map(norm, eachcol(alpha_mat))
            pen_2 = n * sum(scad_penalty.(alpha_norm,  xi_2))#sum(scad_quadratic_approx.(alpha_norm, alpha_alpha_initial_norm, xi_2))
            return -logliklhd_all + pen_1 + pen_2
        else
            return -logliklhd_all
        end
    end

    function evaluate_tuning_param(xi_1, xi_2)
        x0 = vcat(mu_initial, alpha_initial, gamma_initial)
        f = OptimizationFunction(object_fun, Optimization.AutoForwardDiff())
        lb = vcat(fill(-10, p*(k+1)), fill(0, size(gamma_initial)))
        ub = vcat(fill(10, p*(k+1)), fill(10, size(gamma_initial)))
        prob = OptimizationProblem(f, x0, (data_reorgnz, xi_1, xi_2), lb = lb, ub = ub)

        sol = solve(
            prob, 
            NLopt.LD_LBFGS(),
            stopval = 1e-5, 
            ftol_rel = 1e-5,
            xtol_abs = 1e-5, 
            maxeval = 1000,
            maxtime = 30)
        
        mu_hat = sol.u[1:p]
        mu_hat[abs.(mu_hat).<=0.01].=0.0
        alpha_hat = [sol.u[p*i+1:p*(i+1)] for i in 1:k]
        alpha_hat = reshape(sol.u[p+1:p*(k+1)], p, k)'
        alpha_hat = alpha_hat .- mean(alpha_hat, dims=1)
        alpha_hat[abs.(alpha_hat).<=0.01].=0.0
        gamma_hat = sol.u[(k+1)*p+1:end]
        DF = DF_val(data_reorgnz, mu_hat, alpha_hat, gamma_hat, n)
        return optimization_result(xi_1, xi_2, DF, mu_hat, alpha_hat, gamma_hat)
    end

    if penalty == false
        this_result = evaluate_tuning_param(1.0, 1.0)
        return this_result.mu, this_result.alpha ,this_result.gamma
    end

    param1 = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1]
    param2 = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1]
    param_grid = collect(Base.Iterators.product(param1, param2)) |> vec
    n_combinations = length(param_grid)
    tuning_results = Vector{optimization_result}(undef, n_combinations)
    for i in 1:n_combinations
        xi_1, xi_2 = param_grid[i]
        tuning_results[i] = evaluate_tuning_param(xi_1, xi_2)
    end
    
    best_idx = argmin(map(r -> r.criterion, tuning_results))
    best_result = tuning_results[best_idx]
    return best_result.mu, best_result.alpha, best_result.gamma
end