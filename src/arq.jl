#### ARQ-MCMC ####

## arq mcmc algorithm
include("arq_main.jl")
using .ARQMCMC

## - for direct access with internal model
function run_arq_mcmc_analysis(model::HiddenMarkovModel, sample_interval::Array{Float64,1};
    sample_offset::Array{Float64, 1} = (sample_interval / 2), sample_dispersal::Int64 = ARQMCMC.C_DF_ARQ_SR
    , sample_limit::Int64 = C_DF_ARQ_SL, n_chains::Int64 = ARQMCMC.C_DF_ARQ_MC, steps::Int64 = C_DF_MCMC_STEPS
    , burnin::Int64 = df_adapt_period(steps), tgt_ar::Float64 = ARQMCMC.C_DF_ARQ_AR, np::Int64 = 200
    , ess_crit = 0.3) #, sample_cache = Dict{Array{Int64, 1}, GridPoint}()

    # sc::Dict{Array{Int64, 1}, ARQMCMC.GridPoint} = sample_cache
    mdl = ARQModel(get_log_pdf_fn(model, np; essc = ess_crit), sample_interval, sample_offset)
    pr::Array{Distributions.Distribution,1} = [model.prior]
    println("ARQ model initialised: ", model.name)
    return ARQMCMC.run_arq_mcmc_analysis(mdl, pr; sample_dispersal=sample_dispersal, sample_limit=sample_limit, n_chains=n_chains, steps=steps, burnin=burnin, tgt_ar=tgt_ar)#, sample_cache=sc
end

## - for direct access with internal model
# function run_arq_mcmc_analysis(particle_filter::Function, sample_interval::Array{Float64,1};
#     sample_offset::Array{Float64, 1} = (sample_interval / 2), sample_dispersal::Int64 = ARQMCMC.C_DF_ARQ_SR
#     , sample_limit::Int64 = C_DF_ARQ_SL, n_chains::Int64 = ARQMCMC.C_DF_ARQ_MC, steps::Int64 = C_DF_MCMC_STEPS
#     , burnin::Int64 = df_adapt_period(steps), tgt_ar::Float64 = ARQMCMC.C_DF_ARQ_AR, np::Int64 = 200
#     , ess_crit = 0.3) #, sample_cache = Dict{Array{Int64, 1}, GridPoint}()
#
#     # sc::Dict{Array{Int64, 1}, ARQMCMC.GridPoint} = sample_cache
#     mdl = ARQModel(particle_filter, sample_interval, sample_offset)
#     pr::Array{Distributions.Distribution,1} = [model.prior]
#     println("ARQ model initialised: ", model.name)
#     return ARQMCMC.run_arq_mcmc_analysis(mdl, pr; sample_dispersal=sample_dispersal, sample_limit=sample_limit, n_chains=n_chains, steps=steps, burnin=burnin, tgt_ar=tgt_ar)#, sample_cache=sc
# end

## - public wrapper
"""
    run_arq_mcmc_analysis(model, obs_data, theta_range; ... )

Run ARQ-MCMC analysis with `n_chains` Markov chains.

The Gelman-Rubin convergence diagnostic is computed automatically.

**Parameters**
- `model`               -- `DPOMPModel` (see docs.)
- `obs_data`            -- `Observations` data.
- `sample_interval`     -- An array specifying the (fixed or fuzzy) interval between samples.

**Optional**
- `sample_dispersal`   -- i.e. the length of each dimension in the importance sample.
- `sample_limit`        -- sample limit, should be increased when the variance of `model.pdf` is high (default: 1.)
- `n_chains`            -- number of Markov chains (default: 3.)
- `steps`               -- number of iterations.
- `burnin`              -- number of discarded samples.
- `tgt_ar`              -- acceptance rate (default: 0.33.)
- `np`                  -- number of SMC particles in PF (default: 200.)
- `ess_crit`            -- acceptance rate (default: 0.33.)
- `sample_cache`        -- the underlying model likelihood cache - can be retained and reused for future analyses.
"""
function run_arq_mcmc_analysis(model::DPOMPModel, prior::Distributions.Distribution, obs_data::Array{Observation,1}, sample_interval::Array{Float64,1};
    sample_offset::Array{Float64, 1} = (sample_interval / 2), sample_dispersal::Int64 = ARQMCMC.C_DF_ARQ_SR
    , sample_limit::Int64 = ARQMCMC.C_DF_ARQ_SL, n_chains::Int64 = ARQMCMC.C_DF_ARQ_MC, steps::Int64 = C_DF_MCMC_STEPS
    , burnin::Int64 = df_adapt_period(steps), tgt_ar::Float64 = ARQMCMC.C_DF_ARQ_AR, np::Int64 = 200
    , ess_crit = 0.3)#, sample_cache = Dict{Array{Int64, 1}, GridPoint}()

    hmm = get_private_model(model, prior, obs_data)
    return run_arq_mcmc_analysis(hmm, sample_interval; sample_offset=sample_offset, sample_dispersal=sample_dispersal, sample_limit=sample_limit, n_chains=n_chains, steps=steps, burnin=burnin, tgt_ar=tgt_ar, np=np, ess_crit=ess_crit)#, sample_cache=sample_cache
end
