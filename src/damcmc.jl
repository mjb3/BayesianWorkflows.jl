#### data-augmented MCMC
include("damc/mbp.jl")
include("damc/mcmc.jl")

## public wrapper
"""
    run_mcmc_analysis(model, obs_data; ... )

Run an `n_chains`-MCMC analysis using the designated algorithm (*MBP-MCMC* by default.)

The `initial_parameters` are sampled from the prior distribution unless otherwise specified by the user. A Gelman-Rubin convergence diagnostic is automatically carried out (for n_chains > 1) and included in the [multi-chain] analysis results.

**Parameters**
- `model`               -- `DPOMPModel` (see [DCTMPs.jl models]@ref).
- `obs_data`            -- `Observations` data.

**Optional**
- `n_chains`            -- number of Markov chains (default: 3.)
- `initial_parameters`  -- 2d array of initial model parameters. Each column vector correspondes to a single model parameter.
- `steps`               -- number of iterations.
- `adapt_period`        -- number of discarded samples.
- `mbp`                 -- model based proposals (MBP). Set `mbp = false` for standard proposals.
- `ppp`                 -- the proportion of parameter (vs. trajectory) proposals in Gibbs sampler. Default: 30%. NB. not required for MBP.
- `fin_adapt`           -- finite adaptive algorithm. The default is `false`, i.e. [fully] adaptive.
- `mvp`                 -- increase for a higher proportion of 'move' proposals. NB. not applicable if `MBP = true` (default: 2.)

**Example**
```@repl
y = x.observations                          # some simulated data
model = generate_model("SIR", [50, 1, 0])   # a model
results = run_mcmc_analysis(model, y; fin_adapt = true) # finite-adaptive MCMC
tabulate_results(results)                   # optionally, show the results
```

"""
function run_mcmc_analysis(model::DPOMPModel, prior::Distributions.Distribution, obs_data::Array{Observation,1};
    n_chains::Int64 = 3, steps::Int64 = C_DF_MCMC_STEPS,
    adapt_period::Int64 = Int64(floor(steps * C_DF_MCMC_ADAPT)), fin_adapt::Bool = false,
    mbp::Bool = true, ppp::Float64 = 0.3, mvp::Int64 = 3)
    #, initial_parameters = rand(prior, n_chains)

    mdl = get_private_model(model, prior, obs_data)
    if mbp
        return run_mbp_mcmc(mdl, n_chains, steps, adapt_period, fin_adapt)
    else
        return run_std_mcmc(mdl, n_chains, steps, adapt_period, fin_adapt, ppp, mvp)
    end
end
