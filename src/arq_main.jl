#### ARQ-MCMC ####
## module
module ARQMCMC

const C_DEBUG = false
## MOVE TO CMN **
const C_DF_MCMC_STEPS = 50000
const C_DF_MCMC_ADAPT = 0.2
df_adapt_period(steps::Int64) = Int64(floor(steps * C_DF_MCMC_ADAPT))

## ARQ defaults
const C_DF_ARQ_SL = 1       # sample limit
# const C_DF_ARQ_SR = 50      # inital sample distribution
const C_DF_ARQ_MC = 5       # chains
const C_DF_ARQ_AR = 0.33    # targeted AR
const C_DF_ARQ_JT = 0.0     # jitter

import Distributions
import StatsBase
import Random
import Statistics
import PrettyTables
import UnicodePlots     # https://github.com/Evizero/UnicodePlots.jl

## constants
const C_ALG_STD = "ARQ"
const C_ALG_DAUG  = "DAQ"
# const C_ALG_AD  = "ADARQ"

##
export ARQModel, ARQMCMCSample#, run_arq_mcmc_analysis, GridPoint
export tabulate_results
export plot_parameter_trace, plot_parameter_marginal, plot_parameter_heatmap

## structs
include("cmn/cmn_structs.jl")
include("arq/arq_structs.jl")
# length(x::ARQMCMCSample) = 1

## common functions, macro
include("cmn/cmn.jl")
include("arq/arq_alg_cmn.jl")
## standard ARQ MCMC algorithm
include("arq/arq_alg_std.jl")
## delayed acceptance ARQ MCMC algorithm
# include("arq_alg_da.jl")
## data augmented ARQ MCMC algorithm
# include("arq_alg_daug.jl")
## common functions, printing, etc
include("cmn/utils.jl")
include("arq/arq_utils.jl")
## visualisation tools
include("cmn/viz_uc_cmn.jl")
include("arq/viz_uc_arq.jl")

## for internal use (called by public functions)
function run_inner_mcmc_analysis(mdl::LikelihoodModel, da::Bool, steps::Int64, burnin::Int64, chains::Int64, tgt_ar::Float64, grid::Dict{Array{Int64, 1}, GridPoint})
    start_time = time_ns()
    ## designate inner MCMC function and results array
    mcmc_fn = da ? daarq_met_hastings! : arq_met_hastings!
    # is_mu_fn = da ? compute_da_is_mean : compute_is_mean
    ## initialise
    n_theta = length(mdl.sample_interval)
    samples = zeros(n_theta, steps, chains)
    is_uc = 0.0
    fx = zeros(Int64, chains)
    for i in 1:chains   # run N chains using designated function
        # theta_init = rand(1:mdl.sample_dispersal, n_theta)     # choose initial theta coords TEST PRIOR HERE *****
        print(" chain ", i, " initialised")
        mcmc = arq_met_hastings!(samples, i, grid, mdl, steps, burnin, tgt_ar)
        fx[i] = mcmc[1]
        println(" - complete (calls to f(θ) := ", mcmc[1], "; AAR := ", round(mcmc[3] * 100, digits = 1), "%)")
    end
    ## compute scale reduction factor est.
    rejs = handle_rej_samples(samples, burnin)      # shared HMM functions
    sre = gelman_diagnostic(samples, burnin).psrf
    ## get importance sample
    theta_w = collect_theta_weight(grid, n_theta)
    is_mu = zeros(n_theta)
    cv = zeros(length(is_mu),length(is_mu))
    # shared HMM fn:
    compute_is_mu_covar!(is_mu, cv, theta_w[1], theta_w[2])
    # grsp = mdl.sample_dispersal ^ n_theta
    is_output = ImportanceSample(is_mu, cv, theta_w[1], theta_w[2], 0, [-log(sum(theta_w[2]) / length(theta_w[2])), -log(sum(theta_w[2]) / (length(theta_w[2]) ^ (1 / n_theta)))])
    mc_output = MCMCSample(rejs, burnin, sre, time_ns() - start_time)
    ## return results
    output = ARQMCMCSample(is_output, mc_output, mdl.sample_interval, mdl.sample_limit, fx, grid)
    println("- finished in ", print_runtime(output.rej_sample.run_time), ". (Iμ = ", round.(is_output.mu; sigdigits = C_PR_SIGDIG), "; Rμ = ", round.(rejs.mu; sigdigits = C_PR_SIGDIG), "; BME = ", round.(output.imp_sample.bme[1]; sigdigits = C_PR_SIGDIG), ")")
    return output
end

## run standard ARQMCMC analysis
# - `sample_dispersal`   -- the dispersal of intial samples.
# """
#     run_arq_mcmc_analysis(model::ARQModel, priors = []; ... )
#
# Run ARQMCMC analysis with `chains` Markov chains, where `n_chains > 1` the Gelman-Rubin convergence diagnostic is also run.
#
# **Parameters**
# - `model`               -- `ARQModel` (see docs.)
# - `priors`              -- optional: prior distributions or density function. I.e. `Array` of `Function` or `Distributions.Distribution` types.
# **Named parameters**
# - `sample_limit`        -- sample limit, should be increased when the variance of `model.pdf` is high (default: 1.)
# - `n_chains`            -- number of Markov chains (default: 3.)
# - `steps`               -- number of iterations.
# - `burnin`              -- number of discarded samples.
# - `tgt_ar`              -- acceptance rate (default: 0.33.)
# - `jitter`              --  (default: 0.0.)
# - `sample_cache`        -- the underlying model likelihood cache - can be retained and reused for future analyses.
# """
# function run_arq_mcmc_analysis(model::ARQModel, priors::Array{Function,1};
#     sample_limit::Int64 = C_DF_ARQ_SL, steps::Int64 = C_DF_MCMC_STEPS
#     , burnin::Int64 = df_adapt_period(steps), n_chains::Int64 = C_DF_ARQ_MC, tgt_ar::Float64 = C_DF_ARQ_AR
#     , jitter::Float64 = C_DF_ARQ_JT, sample_cache = Dict{Array{Int64, 1}, GridPoint}())
#     # sample_dispersal::Int64 = C_DF_ARQ_SR,
#
#     output = []
#     for i in eachindex(priors)
#         println("Running: ARQMCMC analysis ",  length(priors) == 1 ? "" : string(i, " / ", length(priors)," -"), " (", n_chains, " x " , steps, " steps):")
#         mdl = LikelihoodModel(model.pdf, model.sample_interval, model.sample_offset, sample_limit, sample_dispersal, jitter, priors[i])
#         push!(output, run_inner_mcmc_analysis(mdl, false, steps, burnin, n_chains, tgt_ar, sample_cache))
#     end
#     length(priors) == 1 && (return output[1])
#     return output
# end

## function prior
# function run_arq_mcmc_analysis(model::ARQModel, prior::Function = get_df_arq_prior();
#     sample_dispersal::Int64 = C_DF_ARQ_SR, sample_limit::Int64 = C_DF_ARQ_SL, steps::Int64 = C_DF_MCMC_STEPS, burnin::Int64 = df_adapt_period(steps), n_chains::Int64 = C_DF_ARQ_MC, tgt_ar::Float64 = C_DF_ARQ_AR, jitter::Float64 = C_DF_ARQ_JT, sample_cache = Dict{Array{Int64, 1}, GridPoint}())
#
#     prs = Array{Function,1}(undef, 1)
#     prs[1] = prior
#     return run_arq_mcmc_analysis(model, prs; sample_limit = sample_limit,
#         steps=steps, burnin=burnin, n_chains=n_chains, tgt_ar=tgt_ar, jitter=jitter, sample_cache=sample_cache)
# end

## multiple priors
"""
    run_arq_mcmc_analysis(model::ARQModel, priors = []; ... )

Run ARQMCMC analysis with `chains` Markov chains, where `n_chains > 1` the Gelman-Rubin convergence diagnostic is also run.

**Parameters**
- `model`               -- `ARQModel` (see docs.)
- `priors`              -- prior distributions. I.e. `Array` of `Distributions.Distribution` types.
**Named parameters**
- `sample_limit`        -- sample limit, should be increased when the variance of `model.pdf` is high (default: 1.)
- `n_chains`            -- number of Markov chains (default: 3.)
- `steps`               -- number of iterations.
- `burnin`              -- number of discarded samples.
- `tgt_ar`              -- acceptance rate (default: 0.33.)
- `jitter`              --  (default: 0.0.)
- `sample_cache`        -- the underlying model likelihood cache - can be retained and reused for future analyses.
"""
function run_arq_mcmc_analysis(model::ARQModel, priors::Array{Distributions.Distribution,1};
    sample_limit::Int64=C_DF_ARQ_SL, steps::Int64=C_DF_MCMC_STEPS, burnin::Int64=df_adapt_period(steps), n_chains::Int64=C_DF_ARQ_MC,
    tgt_ar::Float64=C_DF_ARQ_AR, jitter::Float64=C_DF_ARQ_JT, sample_cache=Dict{Array{Int64, 1}, GridPoint}())

    ##
    output = []
    for i in eachindex(priors)
        println("Running: ARQMCMC analysis ",  length(priors) == 1 ? "" : string(i, " / ", length(priors)," -"), " (", n_chains, " x " , steps, " steps):")
        mdl = LikelihoodModel(model.pdf, model.sample_interval, model.sample_offset, sample_limit, jitter, priors[i])
        push!(output, run_inner_mcmc_analysis(mdl, false, steps, burnin, n_chains, tgt_ar, sample_cache))
    end
    length(priors) == 1 && (return output[1])
    return output

    # pfn = Array{Function,1}(undef, length(priors))
    # for i in eachindex(priors)
    #     pfn[i] = get_arq_prior(priors[i])
    # end
    # return run_arq_mcmc_analysis(model, pfn; sample_limit = sample_limit,
        # steps = steps, burnin = burnin, n_chains = n_chains, tgt_ar = tgt_ar, jitter = jitter, sample_cache = sample_cache)
end

## single prior
function run_arq_mcmc_analysis(model::ARQModel, prior::Distributions.Distribution;
    sample_limit::Int64 = C_DF_ARQ_SL, steps::Int64 = C_DF_MCMC_STEPS
    , burnin::Int64 = df_adapt_period(steps), n_chains::Int64 = C_DF_ARQ_MC, tgt_ar::Float64 = C_DF_ARQ_AR
    , jitter::Float64 = C_DF_ARQ_JT, sample_cache = Dict{Array{Int64, 1}, GridPoint}())

    return run_arq_mcmc_analysis(model, [prior]; sample_limit=sample_limit,
        steps=steps, burnin=burnin, n_chains=n_chains, tgt_ar=tgt_ar, jitter=jitter, sample_cache=sample_cache)
end

end ## end of module
