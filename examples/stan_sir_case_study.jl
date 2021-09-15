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

## constants
rnd_seed = 3
population_size = 763
initial_condition = [population_size - 1, 1, 0]
n_observations = 25
max_time = 25.0
path_out = string("out/flu/rs", rnd_seed, "/")

Random.seed!(rnd_seed)

## influenza data (downloaded using 'outbreaks' pkg in R)
data_fp = "data/influenza_england_1978_school.csv"
y = get_observations(data_fp; time_col=2, val_seq=3:4)
# df = CSV.read(data_fp, DataFrames.DataFrame)
# println("----\ndf: ", df, "----\ny: ", y)
println(plot_observations(y; plot_index=1))

## model construction
function get_model(freq_dep::Bool, neg_bin_om::Bool)
    STATE_S = 1
    STATE_I = 2
    STATE_R = 3
    PRM_BETA = 1
    PRM_GAMMA = 2
    PRM_PHI = 3
    model = generate_model("SIR", initial_condition; freq_dep=freq_dep, t0_index=4)
    ## rename
    model.name = string("SIR", freq_dep ? "fd" : "dd", "_", neg_bin_om ? "nbin" : "bin")
    ## statistical model (or 'observation' model in the context of DPOMPs)
    OBS_BEDRIDDEN = 1
    OBS_CONV = 2
    function obs_loglike(y::BayesianWorkflows.Observation, population::Array{Int64,1}, parameters::Array{Float64,1})
        try
            population[STATE_I] == y.val[OBS_BEDRIDDEN] == 0 && (return 0.0)
            population[STATE_I] < y.val[OBS_BEDRIDDEN] && (return -Inf)
            if neg_bin_om   # -ve binomial obs model
                dist = truncated(NegativeBinomial(population[STATE_I], parameters[PRM_PHI]^-1), 0, population_size)
            else            # binomial obs model
                dist = Binomial(population[STATE_I], parameters[PRM_PHI])
            end
            return logpdf(dist, y.val[OBS_BEDRIDDEN])
        catch e
            println("y: ", y, "\n pop:", population, "\n prm: ", parameters)
            throw(e)
        end
    end
    model.obs_loglike = obs_loglike
    ## for sampling y
    function obs_sample!(y::BayesianWorkflows.Observation, population::Array{Int64,1}, parameters::Vector{Float64})
        if population[STATE_I] > 0
            if neg_bin_om   # -ve binomial obs model
                dist = truncated(NegativeBinomial(population[STATE_I], parameters[PRM_PHI]^-1), 0, population_size)
            else            # binomial obs model
                dist = Binomial(population[STATE_I], parameters[PRM_PHI])
            end
            y.val[OBS_BEDRIDDEN] = rand(dist)
        else
            y.val[OBS_BEDRIDDEN] = 0
        end
        y.val[OBS_CONV] = population[STATE_R]
    end
    model.obs_function = obs_sample!
    return model
end

## simulate
function simulate(freq_dep::Bool, neg_bin_om::Bool)
    model = get_model(freq_dep)
    parameters = [1.8, 0.405, 1.85, 722104.0]
    x = gillespie_sim(model, parameters; tmax=max_time, num_obs=n_observations, n_sims=10)
    println(plot_trajectory(x[1]))             # plot a full state trajectory (optional)
    println(plot_observations(x; plot_index=1))
end
# simulate(true)

## prior distribution
function get_prior(freq_dep::Bool, neg_bin_om::Bool)
    beta_p = freq_dep ? [2.0, 1.0] : [2.0, 1.0] / population_size
    pr_beta = truncated(Normal(beta_p...), 0.0, Inf)
    pr_lambda = truncated(Normal(0.4, 0.5), 0.0, Inf)
    phi = neg_bin_om ? truncated(Exponential(5.0), 1.0, Inf) : Uniform(0.4, 1.0)
    t0_lim = ["1978-01-16", "1978-01-22"]
    t0_values = Dates.value.(Dates.Date.(t0_lim, "yyyy-mm-dd"))
    freq_dep && println("NB. t0 mapping: ", t0_lim, " := ", t0_values)
    t0 = Uniform(t0_values...)
    return Product([pr_beta, pr_lambda, phi, t0])
end

##  sampling interval for ARQMCMC algoritm:
sample_interval(freq_dep::Bool) = [freq_dep ? 0.05 : 0.05 / population_size, 0.01, 0.05, 0.5]

## fit models and check results
function fit_model(freq_dep::Bool, neg_bin_om::Bool)
    ## fetch model, prior and fit:
    println("\n---- ---- ", freq_dep ? "FREQUENCY" : "DENSITY", " DEPENDENT ", neg_bin_om ? "-ve " : "", "Bin. MODEL: PARAMETER INFERENCE ---- ----")
    model = get_model(freq_dep, neg_bin_om)
    prior = get_prior(freq_dep, neg_bin_om)
    # - data augmented:
    println("\n---- DATA AUGMENTATED ALGORITHMS ----")
    da_results = run_inference_analysis(model, prior, y; primary=BayesianWorkflows.C_ALG_NM_MBPI, n_particles=12000, n_mutations=7, n_mcmc_chains=3, n_mcmc_steps=100000)
    # println("da_results MC TYPE: ", typeof(da_results.mcmc))
    tabulate_results(da_results)
    save_to_file(da_results, string(path_out, model.name, "/da/"))
    # - smc:
    println("\n---- SMC ALGORITHMS ----")
    smc_results = run_inference_analysis(model, prior, y; validation=BayesianWorkflows.C_ALG_NM_ARQ, sample_interval=sample_interval(freq_dep), n_mcmc_chains=5)
    # println("smc_results MC TYPE: ", typeof(smc_results.mcmc))
    tabulate_results(smc_results)
    save_to_file(smc_results, string(path_out, model.name, "/smc/"))
    ## return as named tuple
    return (da_results = da_results, smc_results = smc_results)
end

## model comparison
# - compare density vs. frequency dependent model
function model_comparison(freq_dep::Bool)
    println("\n---- ---- ---- ", freq_dep ? "FREQUENCY" : "DENSITY", " DEPENDENT MODEL INFERENCE ---- ---- ----")
    models = get_model.(freq_dep, [true, false])
    priors = get_prior.(freq_dep, [true, false])
    sample_intervals = sample_interval.([true, false])
    ## run analysis
    results = run_inference_analysis(models, priors, y; validation=BayesianWorkflows.C_ALG_NM_ARQ, sample_intervals)
    tabulate_results(results)
end

# import Statistics
## predict
# - i.e. resample parameters from posterior samples and simulate
function predict(results::SingleModelResults)
    println("\n-- POSTERIOR PREDICTIVE CHECK --")
    model = results.model
    parameters = resample(results; n = 1000)
    x = gillespie_sim(model, parameters; tmax=max_time, num_obs=n_observations)
    save_to_file(x, string(path_out, model.name, "/predict/"); complete_trajectory=false)
    # println(plot_trajectory(x[1]))             # plot a full state trajectory (optional)
    println(plot_observations(x; plot_index=1))
    println(BayesianWorkflows.plot_observation_quantiles(x))
end

## prior predictive check
# - same but sample parameters from prior
function prior_predict(freq_dep::Bool, neg_bin_om::Bool)
    println("\n-- PRIOR PREDICTIVE CHECK --")
    model = get_model(freq_dep, neg_bin_om)
    prior = get_prior(freq_dep, neg_bin_om)
    parameters = rand(prior, 1000)
    x = gillespie_sim(model, parameters; tmax=max_time, num_obs=n_observations)
    # save_to_file(x, string(path_out, model.name, "/prior_predict/"))
    println(plot_observations(x; plot_index=1); complete_trajectory=false)
end

## simulated inference
function simulated_inference(freq_dep::Bool, neg_bin_om::Bool)
    println("\n-- SIMULATED INFERENCE CHECK --")
    model = get_model(freq_dep, neg_bin_om)
    prior = get_prior(freq_dep, neg_bin_om)
    ## simulate observations
    parameters = rand(prior)
    x = gillespie_sim(model, parameters; tmax=max_time, num_obs=n_observations)
    println(plot_trajectory(x))
    println(plot_observations(x.observations))
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
fd = true                       # frequency (or density) dependent models
nb = false                      # -ve (or plain) binomial obs model
results = fit_model(fd, nb)     # single model inference
# model_comparison(fd)            # compare obs models
predict(results.smc_results)    # posterior predictive check
prior_predict(fd, nb)           # prior predictive check
# simulated_inference(fd, nb)     # simulation check
println("workflow finished. rnd seed: ", rnd_seed)
exit()
