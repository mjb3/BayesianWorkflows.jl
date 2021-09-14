## Stan case study ##
# This [DPOMP] example mirrors the Bayesian workflow for disease transmission
# modelling described here [for ODE models in Stan]:
# https://mc-stan.org/users/documentation/case-studies/boarding_school_case_study.html

using BayesianWorkflows
using Distributions
import Random
import CSV
import DataFrames
import Dates

Random.seed!(0)

## constants
population_size = 763
initial_condition = [population_size - 1, 1, 0]
n_observations = 40
max_time = 40.0

## influenza data (downloaded using 'outbreaks' pkg in R)
data_fp = "data/influenza_england_1978_school.csv"
y = get_observations(data_fp; time_col=2, val_seq=3:4)
# df = CSV.read(data_fp, DataFrames.DataFrame)
# println("----\ndf: ", df, "----\ny: ", y)
println(plot_observations(y; plot_index=1))

## model construction
function get_model(freq_dep::Bool)
    STATE_S = 1
    STATE_I = 2
    STATE_R = 3
    PRM_BETA = 1
    PRM_GAMMA = 2
    PRM_PHI = 3
    model = generate_model("SIR", initial_condition; freq_dep=freq_dep, t0_index=4)
    ## rename
    model.name = string("SIR", freq_dep ? "fd" : "dd")
    ## statistical model (or 'observation' model in the context of DPOMPs)
    OBS_BEDRIDDEN = 1
    OBS_CONV = 2
    function obs_loglike(y::BayesianWorkflows.Observation, population::Array{Int64,1}, parameters::Array{Float64,1})
        try
            population[STATE_I] == y.val[OBS_BEDRIDDEN] == 0 && (return 0.0)
            population[STATE_I] < y.val[OBS_BEDRIDDEN] && (return -Inf)
            obs_dist = NegativeBinomial(population[STATE_I], parameters[PRM_PHI]^-1)
            return logpdf(obs_dist, y.val[OBS_BEDRIDDEN])
        catch e
            println("y: ", y, "\n pop:", population, "\n prm: ", parameters)
            throw(e)
        end
    end
    model.obs_loglike = obs_loglike
    ## for sampling y
    function obs_sample!(y::BayesianWorkflows.Observation, population::Array{Int64,1}, parameters::Vector{Float64})
        if population[STATE_I] > 0
            obs_dist = NegativeBinomial(population[STATE_I], parameters[PRM_PHI]^-1)
            y.val[OBS_BEDRIDDEN] = max(rand(obs_dist), population_size)
        else
            y.val[OBS_BEDRIDDEN] = 0
        end
        y.val[OBS_CONV] = population[STATE_R]
    end
    model.obs_function = obs_sample!
    return model
end

## simulate
function simulate(freq_dep::Bool)
    model = get_model(freq_dep)
    parameters = [1.8, 0.405, 1.85, 722104.0]
    x = gillespie_sim(model, parameters; tmax=max_time, num_obs=n_observations, n_sims=10)
    println(plot_trajectory(x[1]))             # plot a full state trajectory (optional)
    println(plot_observations(x; plot_index=1))
end
# simulate(true)

## prior distribution
function get_prior(freq_dep::Bool)
    beta_p = freq_dep ? [2.0, 1.0] : [2.0, 1.0] / population_size
    pr_beta = Truncated(Normal(beta_p...), 0.0, Inf)
    pr_lambda = Truncated(Normal(0.4, 0.5), 0.0, Inf)
    phi_inv = Truncated(Exponential(5.0), 1.0, Inf)
    t0_lim = ["1978-01-16", "1978-01-22"]
    t0_values = Dates.value.(Dates.Date.(t0_lim, "yyyy-mm-dd"))
    freq_dep && println("NB. t0 mapping: ", t0_lim, " := ", t0_values)
    t0 = Uniform(t0_values...)
    return Product([pr_beta, pr_lambda, phi_inv, t0])
end

##  sampling interval for ARQMCMC algoritm:
sample_interval(freq_dep::Bool) = [freq_dep ? 0.05 : 0.05 / population_size, 0.01, 0.05, 0.5]

## fit models and check results
function fit_model(freq_dep::Bool)
    ## fetch model, prior and fit:
    println("\n---- ---- ", freq_dep ? "FREQUENCY" : "DENSITY", " DEPENDENT MODEL: PARAMETER INFERENCE ---- ----")
    model = get_model(freq_dep)
    prior = get_prior(freq_dep)
    # - data augmented:
    println("\n---- DATA AUGMENTATED ALGORITHMS ----")
    da_results = run_inference_analysis(model, prior, y; primary=BayesianWorkflows.C_ALG_NM_MBPI)
    # println("da_results MC TYPE: ", typeof(da_results.mcmc))
    tabulate_results(da_results)
    save_to_file(da_results, "out/flu/da/")
    # - smc:
    println("\n---- SMC ALGORITHMS ----")
    smc_results = run_inference_analysis(model, prior, y; validation=BayesianWorkflows.C_ALG_NM_ARQ, sample_interval=sample_interval(freq_dep))
    # println("smc_results MC TYPE: ", typeof(smc_results.mcmc))
    tabulate_results(smc_results)
    save_to_file(smc_results, "out/flu/smc/")
    ## return as named tuple
    return (da_results = da_results, smc_results = smc_results)
end

## model comparison
# - compare density vs. frequency dependent model
function model_comparison()
    println("\n---- ---- ---- ---- MODEL INFERENCE ---- ---- ---- ----")
    models = get_model.([false, true])
    priors = get_prior.([false, true])
    sample_intervals = sample_interval.([false, true])
    ## run analysis
    results = run_inference_analysis(models, priors, y; validation=BayesianWorkflows.C_ALG_NM_ARQ, sample_intervals)
    tabulate_results(results)
end

## predict
# - i.e. resample parameters from posterior samples and simulate
function predict(results::SingleModelResults)
    println("\n-- POSTERIOR PREDICTIVE CHECK --")
    model = results.model
    parameters = resample(results; n = 10)
    x = gillespie_sim(model, parameters; tmax=max_time, num_obs=n_observations)
    println(plot_trajectory(x[1]))             # plot a full state trajectory (optional)
    println(plot_observations(x; plot_index=1))
end

## prior predictive check
# - same but sample parameters from prior
function prior_predict(freq_dep::Bool)
    println("\n-- PRIOR PREDICTIVE CHECK --")
    model = get_model(freq_dep)
    prior = get_prior(freq_dep)
    parameters = rand(prior, 10)
    x = gillespie_sim(model, parameters; tmax=max_time, num_obs=n_observations)
    println(plot_observations(x; plot_index=1))
    # for i in eachindex(x)
    #     println("PARAMS: ", parameters[:,i])
    #     println(plot_trajectory(x[i]))             # plot a full state trajectory (optional)
    #     println("OBS: ", x[i].observations)
    #     println(plot_observations(x[i]))
    # end
end

## simulated inference
function simulated_inference(freq_dep::Bool)
    println("\n-- SIMULATED INFERENCE CHECK --")
    model = get_model(freq_dep)
    prior = get_prior(freq_dep)
    ## simulate observations
    parameters = rand(prior)
    x = gillespie_sim(model, parameters; tmax=max_time, num_obs=n_observations)
    println(plot_trajectory(x))
    println(plot_observations(x.observations))
    plot_observations
    ## run inference
    # - data augmented:
    # println("\n---- DATA AUGMENTATED ALGORITHMS ----")
    # da_results = run_inference_analysis(model, prior, x.observations; primary=BayesianWorkflows.C_ALG_NM_MBPI)
    # tabulate_results(da_results)
    # - smc:
    println("\n---- SMC ALGORITHMS ----")
    smc_results = run_inference_analysis(model, prior, x.observations; validation=BayesianWorkflows.C_ALG_NM_ARQ, sample_interval=sample_interval(freq_dep))
    tabulate_results(smc_results)
end

## RUN WORKFLOW FROM HERE:
# dd_results = fit_model(false)   # single model inference:
fd_results = fit_model(true)
model_comparison()              # compare models
predict(fd_results.smc_results) # posterior predictive check
prior_predict(true)             # prior predictive check
simulated_inference(true)       # simulation check
