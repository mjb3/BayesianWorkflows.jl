#### common inference types, etc
include("cmn_struct_hmm.jl")

## DPOMP model
"""
    DPOMPModel

A `mutable struct` which represents a DSSCT model (see [Models](@ref) for further details).

**Fields**
- `name`                -- string, e,g, `"SIR"`.
- `n_events`            -- number of distinct event types.
- `event_rates!`        -- event rate function.
- `initial_state`       -- function for sampling initial model (i.e. population) state.
- `transition!`         -- transition function of form f!(population, event_type).
- `obs_loglike`         -- observation model likelihood function.
- `obs_function`        -- observation function, use this to add 'noise' to simulated observations.
- `t0_index`            -- index of the parameter that represents the initial time. `0` if fixed at `0.0`.

"""
mutable struct DPOMPModel
    name::String                        # model name
    n_events::Int64                     # number of event types
    event_rates!::Function              # computes event rates (in place)
    initial_state::Function             # aka initial condition
    transition!::Function               # i.e adjusts the population according to event type
    obs_loglike::Function               # observation model (log likelihood)
    obs_function::Function              # observation function (sim only) - TO BE REMOVED (-> sim param)
    t0_index::Int64                     # == 0 if initial time known
end
# - `prior`               -- prior [multivariate] Distributions.Distribution.

## for internal use
struct HiddenMarkovModel{RFT<:Function, ICT<:Function, TFT<:Function, OFT<:Function, OMT<:Function, PFT<:Distributions.Distribution}
    name::String                        # model name
    n_events::Int64                     # number of event types
    rate_function::RFT                  # computes event rates (in place)
    initial_state::ICT               # aka initial condition
    transition!::TFT                    # adjusts the population according to event type
    obs_model::OMT                      # observation model (log likelihood)
    obs_function::OFT                   # observation function (sim only) - TO BE REMOVED (-> sim param)
    t0_index::Int64                     # == 0 if initial time known
    obs_data::Array{Observation,1}      # obs data
    prior::PFT                          # prior distribution
end

## generic rejection sample
# - MERGE THIS WITH MCMC?
"""
    RejectionSample

Essentially, the main results of an MCMC analysis, consisting of samples, mean, and covariance matrix.

**Fields**
- `samples`         -- three dimensional array of samples, e.g. parameter; iteration; Markov chain.
- `mu`              -- sample mean.
- `cv`              -- sample covariance matrix.

"""
struct RejectionSample
    theta::Array{Float64,3}         # dims: theta index; chain; sample
    mu::Array{Float64,1}
    cv::Array{Float64,2}
end

## IBIS sample
"""
    ImportanceSample

The results of an importance sampling analysis, such as iterative batch importance sampling algorithms.

**Fields**
- `mu`              -- weighted sample mean.
- `cv`              -- weighted covariance.
- `theta`           -- two dimensional array of samples, e.g. parameter; iteration.
- `weight`          -- sample weights.
- `run_time`        -- application run time.
- `bme`             -- Estimate (or approximation) of the Bayesian model evidence.

"""
struct ImportanceSample
    mu::Array{Float64,1}
    cv::Array{Float64,2}
    theta::Array{Float64,2}
    weight::Array{Float64,1}
    run_time::UInt64
    bme::Array{Float64,1}
end

## e.g. MBP MCMC
"""
    MCMCSample

The results of an MCMC analysis, mainly consisting of a `RejectionSample`.

**Fields**
- `samples`         -- samples of type `RejectionSample`.
- `adapt_period`    -- adaptation (i.e. 'burn in') period.
- `sre`             -- scale reduction factor estimate, i.e. Gelman diagnostic.
- `run_time`        -- application run time.

"""
struct MCMCSample
    samples::RejectionSample
    adapt_period::Int64
    psrf::Array{Float64,2}
    run_time::UInt64
end

## ADD DOCS
"""
    SingleModelResults

The results of a single-model inference analysis.

**Fields**
- `model`       -- model names.
- `ibis`        -- primary analysis IBIS results.
- `mcmc`        -- validation analysis MCMC results.

"""
struct SingleModelResults
    model::DPOMPModel
    ibis::ImportanceSample
    mcmc #::MCMCSample
end

## ADD DOCS
# struct MultiModelResults
#     results::Array{SingleModelResults, 1}
# end

"""
    ModelComparisonResults

The results of a model comparison, based on the Bayesian model evidence (BME.)

**Fields**
- `names`       -- model names.
- `bme`         -- matrix of BME estimates.
- `mu`          -- vector of BME estimate means (by model.)
- `sigma`       -- vector of BME estimate standard deviations.
- `n_runs`      -- number of independent analyses for each model.
- `run_time`    -- total application run time.

"""
struct ModelComparisonResults
    names::Array{String, 1}
    bme::Array{Float64,2}
    mu::Array{Float64,1}
    sigma::Array{Float64,1}
    n_runs::Int64
    run_time::UInt64
    theta_mu::Array{Array{Float64,1}, 2}
end
