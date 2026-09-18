# Integrative Analysis of Heterogeneous Multisource Partly Interval-Censored Data

This repository provides the Julia implementation of the method proposed in "Integrative analysis of heterogeneous multisource partly interval-censored data" by JIN Li and HU Tao.


## Repository Structure

```
├── IAPIC/
│   ├── Project.toml          # Julia project dependencies
│   ├── Manifest.toml         # Dependency manifest
│   ├── liklhdFun.jl          # Log-likelihood functions
│   ├── multiEst.jl           # Proposed estimator with penalty selection
│   └── usefulFuncs.jl        # Utility functions
├── LICENSE                   # GNU GPL v3.0
└── README.md
```

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/jinli-stat/multisource-PIC-data.git
   cd multisource-PIC-data
   ```

2. Activate and instantiate the Julia environment:
   ```julia
   using Pkg
   Pkg.activate("./IAPIC")
   Pkg.instantiate()  # install dependencies (run once)
   ```

## Quick Start

```julia
using LinearAlgebra, Random

include("IAPIC/liklhdFun.jl")
include("IAPIC/multiEst.jl")
include("IAPIC/usefulFuncs.jl")

# Model settings 
J0 = 8
order = 3
p = 20

# True parameters
real_mu = vcat(0.5, 0.5, 0.5, 0.5, 0.0, 0.0, fill(0.0, p - 6))
real_alpha1 = vcat(0.0, 0.0,  0.5,  0.5, -0.5, -0.5, fill(0.0, p - 6))
real_alpha2 = vcat(0.0, 0.0, -0.5, -0.5,  0.5,  0.5, fill(0.0, p - 6))
real_alpha3 = vcat(0.0, 0.0, -0.5, -0.5,  0.5,  0.5, fill(0.0, p - 6))
real_alpha4 = vcat(0.0, 0.0,  0.5,  0.5, -0.5, -0.5, fill(0.0, p - 6))

# Source-specific coefficients
real_beta1 = real_mu + real_alpha1
real_beta2 = real_mu + real_alpha2
real_beta3 = real_mu + real_alpha3
real_beta4 = real_mu + real_alpha4

# Simulate data from 4 sources
Random.seed!(2026)
n_1, n_2, n_3, n_4 = 600, 600, 1400, 1400

data_1 = gen_pic_data(n_1, 0.2, real_beta1)
data_2 = gen_pic_data(n_2, 0.2, real_beta2)
data_3 = gen_pic_data(n_3, 0.2, real_beta3)
data_4 = gen_pic_data(n_4, 0.2, real_beta4)
data = (data_1, data_2, data_3, data_4)

data_full = vcat(data_1, data_2, data_3, data_4)
knots = get_knots(vcat(data_full[:, 1], data_full[:, 2]), J0)

# Unpenalized estimation to get initial values
mu_initial    = fill(0.0, p)
alpha_initial  = hcat(fill(0.0, p), fill(0.0, p), fill(0.0, p), fill(0.0, p))
gamma_initial  = fill(0.1, J0 - 2 + order)

res_i = multisource_estimator(data, mu_initial, alpha_initial, gamma_initial, knots; spl_order=order, penalty="none")

# Penalized estimation with MIC
best_result = multisource_estimator(data, res_i.mu, res_i.alpha, res_i.gamma, knots; spl_order=order, penalty="mic")
mu_hat    = best_result.mu
alpha_hat = best_result.alpha
beta_hat  = mu_hat .+ alpha_hat
beta_hat[abs.(beta_hat) .<= 0.05] .= 0.0
beta_hat
```

## Acknowledgments

The spline implementations are adapted from:

> Qiu, M., & Hu, T. (2024). Bayesian transformation model for spatial partly interval-censored data. *Journal of Applied Statistics*, 51(11), 2139-2156.

## Citation

```bibtex
@article{Jin2026,
   title = {Integrative analysis of heterogeneous multisource partly interval-censored data},
   author = {Jin, Li and Hu, Tao},
   journal = {Statistical Analysis and Data Mining: An ASA Data Science Journal},
   year = {2026},
   volume = {19},
   number = {5},
   pages = {e70113}
}

```

## License

This project is licensed under the GNU General Public License v3.0.
