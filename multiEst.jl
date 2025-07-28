using Optimization, OptimizationNLopt, ForwardDiff
using Statistics, LinearAlgebra, Logging

function BayIC2(data_reorgnz, mu, alpha, gamma, n, k; thsh = 0.01)
    mu_copy = copy(mu)
    alpha_copy = copy(alpha)
    alpha_copy_norm = map(norm, eachrow(alpha_copy))
    mu_copy[abs.(mu_copy) .<= thsh] .= 0.0
    alpha_copy_norm[abs.(alpha_copy_norm) .<= thsh] .= 0.0
    loglik = sum(1:k) do i
        logliklhd_k(mu_copy + alpha_copy[:, i], gamma, data_reorgnz[i])
    end

    deg_freed = count(!iszero, mu_copy) + 
         count(!iszero, alpha_copy_norm) +
         size(gamma, 1)

    # count(!iszero, alpha_copy_norm) + 
    # count(!iszero, alpha_copy[:, 1:end-1]) +
    criterion = -2 * loglik + deg_freed * (log(n) + log(size(mu_copy, 1)))

    return criterion, deg_freed
end

function safe_value(x)
    return any(isnan, x) ? Inf : x
end

struct optimization_result
    xi_1::Float64
    xi_2::Float64
    criterion::Float64
    deg_freed::Int
    mu::Vector{Float64}
    alpha::AbstractArray{Float64}
    gamma::Vector{Float64}
end
Base.show(io::IO, r::optimization_result) = 
    print(io, "xi_1 = $(r.xi_1), xi_2 = $(r.xi_2), criterion = $(r.criterion), deg_freed = $(r.deg_freed)")

function multisource_estimator(data, mu_init, alpha_init, gamma_init, knots;
                                spl_order=2, penalty="none")
    p = size(mu_init, 1)
    n = sum(size(df, 1) for df in data)
    k = length(data)
    J0 = size(gamma_init)[1]
    alpha_init = vec(alpha_init)

    data_reorgnz = ntuple(i -> PIC_data_reorgnz(data[i], spl_order, knots), k)

    if penalty == "none"
        # Do nothing
    elseif penalty == "scad"
        penalty_fun = penalty_scad
    elseif penalty == "mcp"
        penalty_fun = penalty_mcp
    elseif penalty == "mic1" || penalty == "mic2"
        penalty_fun = penalty_mic
    else
        error("Wrong name of penalty function!")
    end

    function object_fun(vars, fixed_val)
        mu = vars[1:p]
        alpha_vec = vars[p+1:(k+1)*p]
        alpha_mat = reshape(alpha_vec, p, k)
        gamma = vars[(k+1)*p+1:end]
        data_reorgnz, xi_1, xi_2 = fixed_val

        loglik = sum(1:k) do i
            logliklhd_k(mu + alpha_mat[:, i], gamma, data_reorgnz[i])
        end
        if penalty == "none"
            return safe_value(-loglik)
        end

        alpha_norm = map(norm, eachrow(alpha_mat))
        if penalty == "mic2"
            pen_1 = log(n) * sum(penalty_fun.(mu, xi_1))
            pen_2 = log(n) * sum(penalty_fun.(alpha_norm, xi_2))
            return safe_value(-loglik + pen_1 + pen_2)
        else 
            pen_1 = n * sum(penalty_fun.(mu, xi_1))
            pen_2 = n * sum(penalty_fun.(alpha_norm, xi_2))
            return safe_value(-loglik + pen_1 + pen_2)
        end
    end
    
    function equality_constraint(res, vars, fixed_val)
        res .= sum(reshape(vars[p+1:(k+1)*p], p, k), dims=2)
        return nothing
    end

    function evaluate_tuning_param(xi_1, xi_2)
        x0 = vcat(mu_init, alpha_init, gamma_init)
        adtype = Optimization.AutoForwardDiff()
        f = OptimizationFunction(
            object_fun, 
            adtype;
            cons = equality_constraint)
        lb = vcat(fill(-5.0, p * (k+1)), fill(0.0, length(gamma_init)))
        ub = vcat(fill(5.0, p * (k+1)), fill(5.0, length(gamma_init)))
        constraint_bounds = zeros(p)
        prob = OptimizationProblem(
            f, x0, (data_reorgnz, xi_1, xi_2), 
            lb = lb, 
            ub = ub,
            lcons = constraint_bounds,
            ucons = constraint_bounds)
        
        sol = solve(
            prob,
            NLopt.LD_SLSQP(), # NLopt.LD_SLSQP(), NLopt.LD_AUGLAG()
            xtol_abs = 1e-3,
            maxeval = 5000)
        if sol.retcode in [:Failure, :UserStop] || any(isnan, sol.u)
            return optimization_result(xi_1, xi_2, Inf, Inf, mu_init, alpha_init, gamma_init)
        end

        mu_hat = sol.u[1:p]
        alpha_hat = sol.u[p+1:p*(k+1)]
        alpha_hat = reshape(alpha_hat, p, k)
        gamma_hat = sol.u[(k+1)*p+1:end]

        criterion, deg_freed = BayIC2(data_reorgnz, mu_hat, alpha_hat, gamma_hat, n, k)
        # print("#")
        return optimization_result(xi_1, xi_2, criterion, deg_freed, mu_hat, alpha_hat, gamma_hat)
    end

    if penalty == "none"
        return evaluate_tuning_param(1.0, 1.0)
    elseif penalty == "mic2"
        return evaluate_tuning_param(n/50,n/50)
        # param1 = [n/2, n]
        # param2 = [n/2, n]
    elseif penalty == "mic1"
        param1 = [0.001, 0.005, 0.01, 0.05, 0.1]
        param2 = [0.001, 0.005, 0.01, 0.05, 0.1]
    else
        param1 = [0.005, 0.01, 0.03, 0.05, 0.07, 0.15, 0.3]
        param2 = [0.005, 0.01, 0.03, 0.05, 0.07, 0.15, 0.3]
    end 
    param_grid = collect(Base.Iterators.product(param1, param2)) |> vec
    
    tuning_results = [evaluate_tuning_param(xi_1, xi_2) for (xi_1, xi_2) in param_grid]
    # df_threshold = 2*p + J0 / 2
    # filter!(x -> x.deg_freed < df_threshold, tuning_results)
    if isempty(tuning_results)
        error("No valid tuning parameters.")
    end

    best_idx = argmin(map(r -> r.criterion, tuning_results))
    best_result = tuning_results[best_idx]

    return best_result
end
