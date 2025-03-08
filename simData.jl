using Distributions, Random, DataFrames
function create_cov_matrix(p)
    return [0.5^abs(i-j) for i in 1:p, j in 1:p]
end

function gen_pic_data(n, exact_rate, true_beta, cum_base_hzd="case1"; cov_matrix=nothing)
    # status 1: exact observe; 2: left censored; 3: right censored; 4: interval censored
    # cum_base_hzd: 
    #           case1: Λ_0 = 0.5*t
    #           case2: Λ_0 = 0.2*t^2
    p = length(true_beta)
    inv_Lambda0 = if cum_base_hzd == "case1"
        x -> x / 0.5
    elseif cum_base_hzd == "case2"
        x -> sqrt(x / 0.2)
    else
        error("Unsupported cum_base_hzd case")
    end
    if cum_base_hzd == "case1"
        x = hcat(rand(Binomial(1, 0.5), n), randn(n, p-1))
    elseif cum_base_hzd == "case2"
        mean_vector = zeros(p-1)
        x = hcat(rand(Binomial(1, 0.5), n), rand(MvNormal(mean_vector, cov_matrix), n)')
    end
    risk_score =  exp.(x * true_beta)
    true_time = inv_Lambda0.(rand(Exponential(1), n) ./ risk_score)
    obv_point = 1 .+ rand(Poisson(4), n)

    u = repeat([0.0], n)
    v = repeat([0.0], n)
    status = repeat([4], n)
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

# p =  50
# real_beta = vcat([0.9, -0.5, 0, 0, 0.7], zeros(p-5))
# cov_matrix = create_cov_matrix(p-1)
# n = 100000
# data = gen_pic_data(n, 0.2, real_beta, "case1");
# counts = combine(groupby(data, :status), nrow => :count)
# total = sum(counts.count)
# counts.Percent = counts.count ./ total .* 100


# p =  10
# real_beta = vcat([0.9, -0.5, 0, 0, 0.7], zeros(p-5))
# cov_matrix = create_cov_matrix(p-1)
# n = 1000
# data = gen_pic_data(n, 0.2, real_beta, "case2"; cov_matrix)
# counts = combine(groupby(data, :status), nrow => :count)
# total = sum(counts.count)
# counts.Percent = counts.count ./ total .* 100