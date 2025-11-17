using DataFrames, LinearAlgebra

function Ispline(x, spl_order, knots)
    k = spl_order + 1  
    num_knots = length(knots)
    num_basis = num_knots - 2 + k    

    extended_knots = vcat(
        fill(knots[1], k), 
        knots[2:(num_knots-1)], 
        fill(knots[num_knots], k)
    )
    
    basis_functions = zeros(num_basis + k - 1, length(x)) 
    for i = k:num_basis 
        basis_functions[i, :] = (extended_knots[i] .<= x .< extended_knots[i+1]) ./ 
                                (extended_knots[i+1] - extended_knots[i])
    end
    
    rec_basis = basis_functions
    for recursion_level = 1:spl_order
        next_basis = zeros(num_basis + k - 1 - recursion_level, length(x))
        for i = (k - recursion_level):num_basis
            factor = (recursion_level + 1) / recursion_level
            knot_span = extended_knots[i + recursion_level + 1] - extended_knots[i]
            
            left_term = (x .- extended_knots[i]) .* rec_basis[i, :]
            right_term = (extended_knots[i + recursion_level + 1] .- x) .* rec_basis[i + 1, :]
            
            next_basis[i, :] = factor * (left_term + right_term) ./ knot_span
        end
        rec_basis = next_basis
    end
    
    # Calculate knot indices for each evaluation point
    knot_indices = zeros(length(x))
    for i = 1:size(x)[1]
        knot_indices[i] = sum(extended_knots .<= x[i])
    end
    
    ispline_basis = zeros(num_basis - 1, length(x))

    if spl_order == 1 
        for i = 2:num_basis
            ispline_basis[i - 1, :] = (i .< knot_indices .- spl_order .+ 1) .+ 
                            (i .== knot_indices) .* 
                            (extended_knots[i + spl_order + 1] - extended_knots[i]) .* 
                            rec_basis[i, :] ./ (spl_order + 1)
        end
    else
        for point_idx = 1:length(x)
            for basis_idx = 2:num_basis
                if basis_idx < (knot_indices[point_idx] - spl_order + 1)
                    ispline_basis[basis_idx - 1, point_idx] = 1
                elseif basis_idx >= (knot_indices[point_idx] - spl_order + 1) && 
                       basis_idx <= knot_indices[point_idx]
                    start_idx = basis_idx
                    end_idx = Int(knot_indices[point_idx])
                    
                    left_knots = extended_knots[start_idx:end_idx]
                    right_knots = extended_knots[(basis_idx + spl_order + 1):Int(knot_indices[point_idx] + spl_order + 1)]
                    
                    ispline_basis[basis_idx - 1, point_idx] = sum((right_knots - left_knots) .* 
                            rec_basis[start_idx:end_idx, point_idx]) / (spl_order + 1)
                else
                    ispline_basis[basis_idx - 1, point_idx] = 0
                end
            end
        end
    end
    
    return ispline_basis
end

function Mspline(x, spl_order, knots)
    k = spl_order 
    num_knots = length(knots)  
    num_basis = num_knots - 2 + k    
    
    extended_knots = vcat(
        fill(knots[1], k), 
        knots[2:(num_knots-1)], 
        fill(knots[num_knots], k)
    )
    
    basis_functions = zeros(num_basis + k - 1, length(x)) 
    for i = k:num_basis 
        basis_functions[i, :] = (extended_knots[i] .<= x .< extended_knots[i+1]) ./ 
                              (extended_knots[i+1] - extended_knots[i])
    end
    
    if spl_order == 1
        return basis_functions
    end
    
    rec_basis = basis_functions
    for recursion_level = 1:(spl_order-1)
        next_basis = zeros(num_basis + k - 1 - recursion_level, length(x))
        for i = (k - recursion_level):num_basis
            factor = (recursion_level + 1) / recursion_level
            knot_span = extended_knots[i + recursion_level + 1] - extended_knots[i]
            
            left_term = (x .- extended_knots[i]) .* rec_basis[i, :]
            right_term = (extended_knots[i + recursion_level + 1] .- x) .* rec_basis[i + 1, :]
            
            next_basis[i, :] = factor * (left_term + right_term) ./ knot_span
        end
        rec_basis = next_basis
    end
    
    return rec_basis
end

function PIC_data_reorgnz(data, spl_order, knots)
    # Exact observations
    data_e = filter(row -> row.status == 1, data)
    n_e = size(data_e, 1)
    exact_obs = if n_e > 0
        x_e = hcat(eachcol(data_e[:, 4:end])...)
        t_e = data_e[:, 1]
        mspline_e = Mspline(t_e, spl_order, knots)'
        ispline_e = Ispline(t_e, spl_order, knots)'
        (n_e, x_e, mspline_e, ispline_e)
    else
        (n_e, nothing, nothing, nothing)
    end

    # Left-censored data
    data_l = filter(row -> row.status == 2, data)
    n_l = size(data_l, 1)
    left_cens = if n_l > 0
        x_l = hcat(eachcol(data_l[:, 4:end])...)
        v_l = data_l[:, 2]
        ispline_l = Ispline(v_l, spl_order, knots)'
        (n_l, x_l, ispline_l)
    else
        (n_l, nothing, nothing)
    end

    # Right-censored data
    data_r = filter(row -> row.status == 3, data)
    n_r = size(data_r, 1)
    right_cens = if n_r > 0
        x_r = hcat(eachcol(data_r[:, 4:end])...)
        u_r = data_r[:, 1]
        ispline_r = Ispline(u_r, spl_order, knots)'
        (n_r, x_r, ispline_r)
    else
        (n_r, nothing, nothing)
    end

    # Interval-censored data
    data_i = filter(row -> row.status == 4, data)
    n_i = size(data_i, 1)
    interval_cens = if n_i > 0
        x_i = hcat(eachcol(data_i[:, 4:end])...)
        u_i = data_i[:, 1]
        v_i = data_i[:, 2]
        ispline_iu = Ispline(u_i, spl_order, knots)'
        ispline_iv = Ispline(v_i, spl_order, knots)'
        (n_i, x_i, ispline_iu, ispline_iv)
    else
        (n_i, nothing, nothing, nothing)
    end
    
    return (exact_obs, left_cens, right_cens, interval_cens)
end

function logliklhd_k(beta_val, gamma_val, data_reorgnz)

    exact_obs, left_cens, right_cens, interval_cens = data_reorgnz
    n_e, x_e, mspline_e, ispline_e = exact_obs
    n_l, x_l, ispline_l = left_cens
    n_r, x_r, ispline_r = right_cens
    n_i, x_i, ispline_iu, ispline_iv = interval_cens

    # beta_val = mu .+ alpha_k

    lkhd_e = 0
    if n_e !=0
        base_hzd_at_ti = mspline_e * gamma_val
        # part1 = sum([xi > 0 ? safe_log(xi) : safe_log(1) for xi in base_hzd_at_ti])
        part1 = sum(safe_log.(base_hzd_at_ti))
        lkhd_e = part1 + sum(x_e*beta_val) - sum(ispline_e * gamma_val .* safe_exp.(x_e*beta_val))
    end
    
    lkhd_l = 0
    if n_l !=0
        part2 = 1 .- safe_exp.(- ispline_l * gamma_val .* safe_exp.(x_l*beta_val)) 
        lkhd_l = sum(safe_log.(part2))
        # lkhd_l = sum([xi > 0 ? safe_log(xi) : safe_log(1) for xi in part2])
    end
    
    lkhd_r = 0
    if n_r !=0
        lkhd_r = - sum(ispline_r * gamma_val .* safe_exp.(x_r*beta_val))
    end
    
    lkhd_i = 0
    if n_i !=0
        u_part = safe_exp.(- ispline_iu * gamma_val .* safe_exp.(x_i*beta_val))
        v_part = safe_exp.(- ispline_iv * gamma_val .* safe_exp.(x_i*beta_val))
        part3 = u_part - v_part
        lkhd_i = sum(safe_log.(part3))
        # sum([xi > 0 ? safe_log(xi) : safe_log(1) for xi in part3])
    end

    return lkhd_e + lkhd_l + lkhd_r + lkhd_i
end