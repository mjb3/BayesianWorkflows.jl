#### Discrete-state-space Partially Observed Markov Processes
using BayesianWorkflows
import Random
import Distributions

Random.seed!(1)

## variables
theta = [0.003, 0.1]                # model parameters
initial_condition = [100, 1]
data_fp = "data/pooley.csv"
y = get_observations(data_fp)

## generate a model
model = generate_model("SIS", initial_condition)
prior = Distributions.Product(Distributions.Uniform.(zeros(2), [0.01, 0.5]))

## simulation # NB. first define the SIS 'model' variable, per above
println("-- RUNNING MODEL SIMULATION --")
x = gillespie_sim(model, theta)	    # run simulation
println(plot_trajectory(x))			# plot (optional)

#### BAYESIAN INFERENCE WORKFLOWS ####
sample_interval = [0.0005, 0.02]    # intervals used for ARQ algorithm

## single model workflow
function parameter_inference_example()
    println("\n-- RUNNING SINGLE-MODEL INFERENCE WORKFLOW --")
    results = run_inference_analysis(model, prior, y; validation=BayesianWorkflows.C_ALG_NM_ARQ, sample_interval=sample_interval)
    tabulate_results(results)
end

## model comparison workflow
function model_comparison_example()
    println("\n-- RUNNING MULTI-MODEL INFERENCE WORKFLOW --")
    # define alternative model
    seis_model = generate_model("SEIS", [100, 0, 1])
    seis_model.obs_model = partial_gaussian_obs_model(2.0, seq = 3, y_seq = 2)
    seis_prior = Distributions.Product(Distributions.Uniform.(zeros(3), [0.1,0.5,0.5]))
    # run model comparison workflow
    models::Array{DPOMPModel, 1} = [model, seis_model]
    priors::Array{Distributions.Distribution, 1} = [prior, seis_prior]
    results = run_inference_analysis(models, priors, y)
    tabulate_results(results)
end

## run examples:
parameter_inference_example()
model_comparison_example()

#### INDIVIDUAL ALGORITHM CALLS ####

## ARQMCMC
# results = BayesianWorkflows.run_arq_mcmc_analysis(model, prior, y, sample_interval)
# tabulate_results(results)
# println(plot_parameter_trace(results, 1))
# save_to_file(results, "out/arq/")

## MBP MCMC
# results = BayesianWorkflows.run_mcmc_analysis(model, prior, y)
# tabulate_results(results)

## MBP IBIS
# results = BayesianWorkflows.run_mbp_ibis_analysis(model, prior, y)
# tabulate_results(results)

## SMC^2
# results = BayesianWorkflows.run_smc2_analysis(model, prior, y)
# tabulate_results(results)
