#### Seq. Monte Carlo stuffs

## Particle filter for HMM
include("smc/resample.jl")
include("smc/pf.jl")
include("smc/ibis.jl")

## interfaces
# - custom function
# - dpomp model

## pf interface
"""
    get_particle_filter_lpdf(model, obs_data; ... )

Generate a SMC [log] likelihood estimating function for a `DPOMPModel` -- a.k.a. a particle filter.

**Parameters**
- `model`               -- `DPOMPModel` (see [DCTMPs.jl models]@ref).
- `obs_data`            -- `Observations` data.

**Named parameters**
- `np`                  -- number of particles.
- `rs_type`             -- resampling method: 2 for stratified, 3 for multinomial, or 1 for systematic (the default.)
- `ess_rs_crit`         -- effective-sample-size resampling criteria.
```
"""
function get_particle_filter_lpdf(model::DPOMPModel, prior::Distributions.Distribution, obs_data::Array{Observation,1}; np::Int64 = C_DF_PF_P, rs_type::Int64 = 1, essc::Float64 = C_DF_ESS_CRIT)
    mdl = get_private_model(model, prior, obs_data)
    return get_log_pdf_fn(mdl, np, rs_type; essc = essc)
end

"""
    run_smc2_analysis(model, obs_data; ... )

Run an *SMC^2* (i.e. particle filter IBIS) analysis based on `model` and `obs_data` of type `Observations`.

**Parameters**
- `model`               -- `DPOMPModel` (see [DCTMPs.jl models]@ref).
- `obs_data`            -- `Observations` data.

**Optional**
- `np`                  -- number of [outer, i.e. theta] particles (default = 2000.)
- `npf`                 -- number of [inner] particles (default = 200.)
- `ess_rs_crit`         -- resampling criteria (default = 0.5.)
- `ind_prop`            -- true for independent theta proposals (default = false.)
- `alpha`               -- user-defined, increase for lower acceptance rate targeted (default = 1.002.)

**Example**
```@repl
# NB. using 'y' and 'model' as above
results = run_smc2_analysis(model, y)   # importance sample
tabulate_results(results)               # show the results
```
"""
function run_smc2_analysis(model::DPOMPModel, prior::Distributions.Distribution, obs_data::Array{Observation,1};
    np = C_DF_SMC2_P, npf = C_DF_PF_P, ess_rs_crit = C_DF_ESS_CRIT, ind_prop = true, alpha = C_DF_ALPHA)

    mdl = get_private_model(model, prior, obs_data)
    theta_init = rand(mdl.prior, np)
    # run_pibis(model::HiddenMarkovModel, theta::Array{Float64, 2}, ess_rs_crit::Float64, ind_prop::Bool, alpha::Float64, np::Int64
    # pf = get_particle_filter_lpdf(model, prior, obs_data; np = npf)
    println("Running: ", np, "-particle SMC^2 analysis (model: ", model.name, ")")
    return run_pibis(mdl, theta_init, ess_rs_crit, ind_prop, alpha, npf)
end

## MBP IBIS algorithm
"""
    run_mbp_ibis_analysis(model, obs_data; ... )

Run an *MBP-IBIS* analysis based on `model`, and `obs_data` of type `Observations`.

**Parameters**
- `model`               -- `DPOMPModel` (see [DCTMPs.jl models]@ref).
- `obs_data`            -- `Observations` data.

**Optional**
- `np`                  -- number of particles (default = 4000.)
- `ess_rs_crit`         -- resampling criteria (default = 0.5.)
- `n_props`             -- MBP mutations per step (default = 3.)
- `ind_prop`            -- true for independent theta proposals (default = false.)
- `alpha`               -- user-defined, increase for lower acceptance rate targeted (default = 1.002.)

**Example**
```@repl
# NB. using 'y' and 'model' as above
results = run_mbp_ibis_analysis(model, y)# importance sample
tabulate_results(results)                # show the results
```

"""
function run_mbp_ibis_analysis(model::DPOMPModel, prior::Distributions.Distribution, obs_data::Array{Observation,1}; np = C_DF_MBPI_P, ess_rs_crit = C_DF_MBPI_ESS_CRIT, n_props=C_DF_MBPI_MUT, ind_prop = false, alpha = C_DF_ALPHA)
    mdl = get_private_model(model, prior, obs_data)
    theta_init = rand(prior, np)
    return run_mbp_ibis(mdl, theta_init, ess_rs_crit, n_props, ind_prop, alpha)
end
