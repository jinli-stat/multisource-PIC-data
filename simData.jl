using Distributions, Random, DataFrames

function gen_pic_data(n, exact_rate, true_beta, cum_base_hzd="case1")
    # status 1: exact observe; 2: left censored; 3: right censored; 4: interval censored
    # cum_base_hzd: 
    #           case1: Λ_0 = 0.5*t
    #           case2: Λ_0 = 0.2*t^2  
    p = length(true_beta)
    cov_matrix = [0.5^abs(i-j) for i in 1:p, j in 1:p]
    cov_matrix[cov_matrix.<=10e-5] .= 0
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
    risk_score =  exp.(x * true_beta)
    true_time = inv_Lambda0.(rand(Exponential(1), n) ./ risk_score)
    obv_point = 1 .+ rand(Poisson(4), n)

    u = fill(0.0, n)
    v = fill(0.0, n)
    status = fill(4, n)
    ind = sample(1:n, Int(round(n * exact_rate, digits = 0)), replace = false) |> sort

    for i = 1:n
        if (i in ind) & (true_time[i]<=15)
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