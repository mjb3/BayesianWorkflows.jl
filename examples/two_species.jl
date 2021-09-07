## Two-species interaction model case study ##
# - *** WIP ***

using BayesianWorkflows
using Distributions
import Random
import CSV
import DataFrames
import Dates

Random.seed!(1)

## data (downloaded using 'outbreaks' pkg in R)
data_fp = "data/dengue_fais_2011.csv"
# data_fp = "data/zika_yap_2007.csv"
y = get_observations(data_fp; time_col=3, val_seq=4)
# df = CSV.read(data_fp, DataFrames.DataFrame)
# println("----\ndf: ", df, "----\ny: ", y)

## variables
population_size = 294 # fais (2011)
# population_size = 7391 # yap (2007 - 2011)
initial_condition = [population_size - 1, 1, 0]
# theta = [0.003, 0.1]                # model parameters

## define transmission model
function get_model(freq_dep::Bool)
    STATE_S = 1
    STATE_I = 2
    STATE_R = 3
    PRM_BETA = 1
    PRM_LAMBDA = 2
    PRM_PHI = 3
    ## rate function


    model = generate_model("SIR", initial_condition; freq_dep=freq_dep, t0_index=4)
    ## statistical model (or 'observation' model in the context of DPOMPs)
    OBS_BEDRIDDEN = 1
    OBS_CONV = 2
    function obs_model(y::BayesianWorkflows.Observation, population::Array{Int64,1}, parameters::Array{Float64,1})
        try
            population[STATE_I] < y.val[OBS_BEDRIDDEN] && (return -Inf)
            obs_dist = NegativeBinomial(population[STATE_I], parameters[PRM_PHI]^-1)
            return logpdf(obs_dist, y.val[OBS_BEDRIDDEN])
        catch e
            println("y: ", y, "\n pop:", population, "\n prm: ", parameters)
            throw(e)
        end
    end
    model.obs_model = obs_model
    ## for sampling y
    function obs_fn!(y::BayesianWorkflows.Observation, population::Array{Int64,1}, parameters::Vector{Float64})
        obs_dist = NegativeBinomial(population[STATE_I], parameters[PRM_PHI]^-1)
        y.val[OBS_BEDRIDDEN] = rand(obs_dist)
        y.val[OBS_CONV] = population[STATE_R]
    end
    model.obs_function = obs_fn!
    return model
end

## prior distribution
function get_prior(freq_dep::Bool)
    beta_p = freq_dep ? [2.0, 1.0] : [2.0, 1.0] / population_size
    pr_beta = Truncated(Normal(beta_p...), 0.0, Inf)
    pr_lambda = Truncated(Normal(0.4, 0.5), 0.0, Inf)
    phi_inv = Truncated(Exponential(5.0), 1.0, Inf)
    t0_lim = Dates.value.(Dates.Date.(["1978-01-16", "1978-01-22"], "yyyy-mm-dd"))
    println("t0_lim: ", t0_lim)
    t0 = Uniform(t0_lim...)
    return Product([pr_beta, pr_lambda, phi_inv, t0])
end

## fit models and check results
function fit_model(freq_dep::Bool)
    # - sampling interval for ARQMCMC algoritm:
    sample_interval = [freq_dep ? 0.05 : 0.05 / population_size, 0.01, 0.05, 0.5]
    ## fetch model, prior and fit:
    println("\n---- ---- ", freq_dep ? "FREQUENCY" : "DENSITY", " DEPENDENT MODEL: PARAMETER INFERENCE ---- ----")
    model = get_model(freq_dep)
    prior = get_prior(freq_dep)
    # - data augmented:
    println("\n---- DATA AUGMENTATED ALGORITHMS ----")
    results = run_inference_analysis(model, prior, y; primary=BayesianWorkflows.C_ALG_NM_MBPI)
    tabulate_results(results)
    # - smc:
    println("\n---- SMC ALGORITHMS ----")
    results = run_inference_analysis(model, prior, y; validation=BayesianWorkflows.C_ALG_NM_ARQ, sample_interval=sample_interval)
    tabulate_results(results)
end
fit_model(false)
fit_model(true)

## predict
# - i.e. resample parameters from posterior samples and simulate
function simulate(freq_dep::Bool)


end

## prior predictive check
# - same but sample parameters from prior
# println("\nprior samples:\n", rand(prior, 1000))

## model comparison
