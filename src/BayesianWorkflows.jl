module BayesianWorkflows

## dependencies
import Distributions
import DataFrames
import CSV
import StatsBase
import LinearAlgebra
import Statistics
import PrettyTables
import UnicodePlots

### global constants
const C_DEBUG = false               # debug mode
# const C_RT_UNITS = 1000000000
# const C_PR_SIGDIG = 3
const C_LBL_BME = "-ln p(y)"
const C_ALG_NM_SMC2 = "SMC2"
const C_ALG_NM_MBPI = "MBPI"
const C_ALG_NM_MBPM = "MBPM"
const C_ALG_NM_ARQ = "ARQ"
# const C_INF_DELTA = 0.0000000000000001
const MAX_TRAJ = 196000

## MCMC defaults
const C_DF_MCMC_STEPS = 50000
const C_DF_MCMC_ADAPT = 0.2
const C_MCMC_ADAPT_INTERVALS = 10

## IBIS defaults
const C_DF_MBPI_P = 10000       # number of ibis particles
const C_DF_SMC2_P = 4000        # number of smc2 particles
const C_DF_PF_P = 200           # number of pf particles
const C_DF_ESS_CRIT = 0.3       # resampling criteria:
const C_DF_MBPI_ESS_CRIT = 0.5
const C_DF_MBPI_MUT = 3         # mutations
const C_DF_ALPHA = 1.002        # acceptance alpha

## helpers
df_adapt_period(steps::Int64) = Int64(floor(steps * C_DF_MCMC_ADAPT))

## common resources
include("cmn_structs.jl")
include("cmn.jl")
include("cmn_hmm.jl")

## model interface
include("models.jl")

## simulation
include("gillespie.jl")

## inference algorithms
include("damcmc.jl")
include("smc.jl")
include("arq.jl")

## printing, tabulation, visualisation
include("utils.jl")
include("tab.jl")
include("viz_uc_cmn.jl")
include("viz_uc_hmm.jl")

## PH - replace this with julia Exceptions **
struct AlgorithmException <: Exception
   msg::String
end

## workflows
# - single model workflow
function run_inference_workflow(model::DPOMPModel, prior::Distributions.Distribution, obs_data::Array{Observation,1};
    primary=C_ALG_NM_SMC2, validation=C_ALG_NM_MBPM, sample_interval=nothing)

    ## type conversion
    get_type_vals(x) = [getfield(x, v) for v in fieldnames(typeof(x))]
    function get_mcmc_sample(x)
        xv = get_type_vals(x)
        xv[1] = RejectionSample(get_type_vals(x.samples)...)
        return MCMCSample(xv...)
    end
    ## primary analysis (ibis)
    if primary == C_ALG_NM_SMC2
        ibis = run_smc2_analysis(model, prior, obs_data)
    elseif primary == C_ALG_NM_MBPI
        ibis = run_mbp_ibis_analysis(model, prior, obs_data)
    else
        ibis = nothing
        msg = string("Algorithm '", primary, "' not recognised. Valid values are: ", (C_ALG_NM_SMC2, C_ALG_NM_MBPI))
        throw(AlgorithmException(msg))
    end
    ## secondary analysis (mcmc)
    if validation == C_ALG_NM_MBPM
        mcmc = run_mcmc_analysis(model, prior, obs_data)
    elseif validation == C_ALG_NM_ARQ
        mcmc = run_arq_mcmc_analysis(model, prior, obs_data, sample_interval)
        mcmc = get_mcmc_sample(mcmc.rej_sample)
    else
        mcmc = nothing
        msg = string("Validation algorithm '", validation, "' not recognised. Valid values are: ", (C_ALG_NM_MBPM, C_ALG_NM_ARQ))
        throw(AlgorithmException(msg))
    end
    println("- workflow complete.")
    return SingleModelResults(model, ibis, mcmc)
end
# - multi model
function run_inference_workflow(models::Array{DPOMPModel}, prior::Array{Distributions.Distribution}, obs_data::Array{Observation,1})

end
# - multi prior

## public functions
export generate_model
export gillespie_sim
export get_observations
export tabulate_results, print_results
export plot_trajectory, plot_parameter_trace, plot_parameter_heatmap, plot_model_comparison
export run_inference_workflow

end # module
