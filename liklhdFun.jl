using DataFrames, LinearAlgebra
include("splineBasis.jl")

function initial_knots(values, J0)
    sorted_values = sort(values)
    sorted_values = sorted_values[1:(end-1)]
    n = length(sorted_values)
    step = div(n, (J0-1))
    a = [sorted_values[i] for i in 1:step:n]
    if length(a) == (J0-1)
        a = vcat(a, (a[end] + a[end-1])/2)
        sort!(a)        
    end
    a[end] = (a[end] + a[end-1])/2
    a[end] = (a[end] + a[end-1])/2
    push!(a, Inf)
    return a
end

function PIC_data_reorgnz(data, order, knots)
    # Exact observations
    data_e = filter(row -> row.status == 1, data)
    n_e = size(data_e, 1)
    exact_obs = if n_e > 0
        x_e = hcat(eachcol(data_e[:, 4:end])...)
        t_e = data_e[:, 1]
        mspline_e = Mspline(t_e, order, knots)'
        ispline_e = Ispline(t_e, order, knots)'
        (n_e, x_e, mspline_e, ispline_e)
    else
        nothing
    end

    # Left-censored data
    data_l = filter(row -> row.status == 2, data)
    n_l = size(data_l, 1)
    left_cens = if n_l > 0
        x_l = hcat(eachcol(data_l[:, 4:end])...)
        v_l = data_l[:, 2]
        ispline_l = Ispline(v_l, order, knots)'
        (n_l, x_l, ispline_l)
    else
        nothing
    end

    # Right-censored data
    data_r = filter(row -> row.status == 3, data)
    n_r = size(data_r, 1)
    right_cens = if n_r > 0
        x_r = hcat(eachcol(data_r[:, 4:end])...)
        u_r = data_r[:, 1]
        ispline_r = Ispline(u_r, order, knots)'
        (n_r, x_r, ispline_r)
    else
        nothing
    end

    # Interval-censored data
    data_i = filter(row -> row.status == 4, data)
    n_i = size(data_i, 1)
    interval_cens = if n_i > 0
        x_i = hcat(eachcol(data_i[:, 4:end])...)
        u_i = data_i[:, 1]
        v_i = data_i[:, 2]
        ispline_iu = Ispline(u_i, order, knots)'
        ispline_iv = Ispline(v_i, order, knots)'
        (n_i, x_i, ispline_iu, ispline_iv)
    else
        nothing
    end
    
    return (exact_obs, left_cens, right_cens, interval_cens)
end

function logliklhd_k(mu, alpha_k, gamma, data_reorgnz)

    exact_obs, left_cens, right_cens, interval_cens = data_reorgnz
    n_e, x_e, mspline_e, ispline_e = exact_obs
    n_l, x_l, ispline_l = left_cens
    n_r, x_r, ispline_r = right_cens
    n_i, x_i, ispline_iu, ispline_iv = interval_cens

    beta_hat = mu + alpha_k

    lkhd_e = 0
    if n_e !=0
        base_hzd_at_ti = mspline_e * gamma
        part1 = sum([xi > 0 ? log(xi) : log(1) for xi in base_hzd_at_ti])
        lkhd_e = part1 + sum(x_e*beta_hat) - sum(ispline_e * gamma .* exp.(x_e*beta_hat))
    end
    
    lkhd_l = 0
    if n_l !=0
        part2 = 1 .- exp.(- ispline_l * gamma .* exp.(x_l*beta_hat)) 
        lkhd_l = sum([xi > 0 ? log(xi) : log(1) for xi in part2])
    end
    
    lkhd_r = 0
    if n_r !=0
        lkhd_r = - sum(ispline_r * gamma .* exp.(x_r*beta_hat))
    end
    
    lkhd_i = 0
    if n_i !=0
        u_part = exp.(- ispline_iu * gamma .* exp.(x_i*beta_hat))
        v_part = exp.(- ispline_iv * gamma .* exp.(x_i*beta_hat))
        part3 = u_part - v_part
        lkhd_i = sum([xi > 0 ? log(xi) : log(1) for xi in part3])
    end

    return lkhd_e + lkhd_l + lkhd_r + lkhd_i
end