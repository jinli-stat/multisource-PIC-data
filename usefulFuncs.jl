using Distributions, Random, DataFrames

const MAX_EXP = log(prevfloat(Inf))
const MIN_EXP = -MAX_EXP
const EPS_FLOAT64 = eps(Float64)

function safe_exp(x)
    return exp(clamp(x, MIN_EXP, MAX_EXP))
end

function safe_log(x)
    return log(max(x, EPS_FLOAT64))
end

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
    return 1 - safe_exp(-tuning_param * (beta_hat^2))
end

function penalty_mic(beta_hat, tuning_param)
    temp_val = safe_exp(2 * tuning_param * beta_hat^2)
    return (temp_val - 1.0) / (temp_val + 1.0)
end

function get_knots(arr, J0)
    filtered_arr = filter(x -> x != 0 && isfinite(x), arr)
    if isempty(filtered_arr)
        return nothing
    end
    
    max_value = quantile(filtered_arr, 0.975)
    min_value = quantile(filtered_arr, 0.025)
    
    knots_J0 = range(min_value, max_value, length=J0)
    knots_J0 = round.(knots_J0, digits=2)
    return knots_J0
end

function gen_pic_data(n, exact_rate, true_beta, cum_base_hzd="case1")
    # status 1: exact observe; 2: left censored; 3: right censored; 4: interval censored
    # cum_base_hzd: 
    #           case1: Λ_0 = 0.5*t
    #           case2: Λ_0 = 0.2*t^2  
    p = length(true_beta)
    cov_matrix = [0.5^abs(i-j) for i in 1:p, j in 1:p]
    inv_Lambda0 = if cum_base_hzd == "case1"
        x -> x / 0.5
    elseif cum_base_hzd == "case2"
        x -> sqrt(x / 0.2)
    else
        error("Unsupported cum_base_hzd case")
    end

    mean_vector = zeros(p)
    x = rand(MvNormal(mean_vector, cov_matrix), n)'
    x[:, 2] = rand(Binomial(1, 0.5), n)
    x[:, 4] = rand(Binomial(1, 0.5), n)
    x[:, 6] = rand(Binomial(1, 0.5), n)
    risk_score =  safe_exp.(x * true_beta)
    true_time = inv_Lambda0.(rand(Exponential(1), n) ./ risk_score)
    obv_point = 1 .+ rand(Poisson(4), n)

    u = fill(0.0, n)
    v = fill(0.0, n)
    status = fill(4, n)
    ind = sample(1:n, round(Int, n * exact_rate), replace = false) |> sort

    for i = 1:n
        if (i in ind) && (true_time[i]<=15)
            u[i] = true_time[i]
            v[i] = true_time[i]
            status[i] = 1
        else
            time_lag = rand(Exponential(0.5), obv_point[i])
            time_seq = vcat(0, cumsum(time_lag), Inf)
            num_obs = length(time_seq)
            if true_time[i] == 0
                u[i] = time_seq[1]
                v[i] = time_seq[2]
            end
            for j in 2:num_obs
                if time_seq[j-1] < true_time[i] <= time_seq[j]
                    u[i] = time_seq[j-1]
                    v[i] = time_seq[j]
                end
            end
        end
        if u[i] == 0
            status[i] = 2
        end
        if v[i] == Inf 
            status[i] = 3
        end
    end

    data = DataFrame(u = u, v = v, status = status)
    for i in 1:p
        data[!, Symbol("x$i")] = x[:, i]
    end
    return(data)
end


# test
# p =  50
# n = 1000

# mu = vcat(-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, fill(0.0, p-6))
# alpha_1 = vcat(0.0, 0.0, 0.5, 0.5, 0.5, 0.5, fill(0.0, p-6))
# alpha_2 = vcat(0.0, 0.0, -0.5, -0.5, -0.5, -0.5, fill(0.0, p-6))
# alpha_3 = vcat(0.0, 0.0, -0.5, -0.5, -0.5, -0.5, fill(0.0, p-6))
# alpha_4 = vcat(0.0, 0.0, 0.5, 0.5, 0.5, 0.5, fill(0.0, p-6))

# beta_1 = mu + alpha_1
# beta_2 = mu + alpha_2
# beta_3 = mu + alpha_3
# beta_4 = mu + alpha_4

# data_1 = gen_pic_data(n, 0.2, beta_1, "case1");
# data_2 = gen_pic_data(n, 0.2, beta_2, "case1");
# data_3 = gen_pic_data(n, 0.2, beta_3, "case1");
# data_4 = gen_pic_data(n, 0.2, beta_4, "case1");

# counts = combine(groupby(data, :status), nrow => :count)
# total = sum(counts.count)
# counts.Percent = counts.count ./ total .* 100