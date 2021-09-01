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

Random.seed!(1)

## influenza data (downloaded using 'outbreaks' pkg in R)
data_fp = "data/influenza_england_1978_school.csv"
y = get_observations(data_fp; time_col=2, val_seq=3:4)
# df = CSV.read(data_fp, DataFrames.DataFrame)
# println("----\ndf: ", df, "----\ny: ", y)

## variables
population_size = 763
initial_condition = [population_size - 1, 1, 0]
# theta = [0.003, 0.1]                # model parameters

## define transmission model
function get_model(freq_dep::Bool)
    STATE_S = 1
    STATE_I = 2
    STATE_R = 3
    PRM_BETA = 1
    PRM_GAMMA = 2
    PRM_PHI = 3
    model = generate_model("SIR", initial_condition; freq_dep=freq_dep, t0_index=4)
    ## statistical model (or 'observation' model in the context of DPOMPs)
    OBS_BEDRIDDEN = 1
    OBS_CONV = 2
    function obs_model(y::BayesianWorkflows.Observation, population::Array{Int64,1}, parameters::Array{Float64,1})
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
    model.obs_model = obs_model
    ## for sampling y
    function obs_fn!(y::BayesianWorkflows.Observation, population::Array{Int64,1}, parameters::Vector{Float64})
        if population[STATE_I] > 0
            obs_dist = NegativeBinomial(population[STATE_I], parameters[PRM_PHI]^-1)
            y.val[OBS_BEDRIDDEN] = rand(obs_dist)
        else
            y.val[OBS_BEDRIDDEN] = 0
        end
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
    results = run_inference_workflow(model, prior, y; primary=BayesianWorkflows.C_ALG_NM_MBPI)
    tabulate_results(results)
    # - smc:
    println("\n---- SMC ALGORITHMS ----")
    results = run_inference_workflow(model, prior, y; validation=BayesianWorkflows.C_ALG_NM_ARQ, sample_interval=sample_interval)
    tabulate_results(results)
end
# fit_model(false)
# fit_model(true)

## predict
# - i.e. resample parameters from posterior samples and simulate
function simulate(freq_dep::Bool, parameters)
    model = get_model(freq_dep)
    x = gillespie_sim(model, parameters; tmax=40.0, num_obs=40, n_sims=10)
    println(plot_trajectory(x[1]))             # plot a full state trajectory (optional)
    println(plot_observations(x; plot_index=1))
end
simulate(true, [1.8, 0.405, 1.85, 722104.0])

## prior predictive check
# - same but sample parameters from prior
# println("\nprior samples:\n", rand(prior, 1000))

## model comparison
