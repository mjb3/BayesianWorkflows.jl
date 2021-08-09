#### Discrete-state-space Partially Observed Markov Processes
using BayesianWorkflows
import Random
import Distributions

Random.seed!(1)

## variables
theta = [0.003, 0.1]
initial_condition = [100, 1]
data_fp = "data/pooley.csv"

## getting started
y = get_observations(data_fp) # val_seq=2:3
model = generate_model("SIS", initial_condition)
prior = Distributions.Product(Distributions.Uniform.(zeros(2), [0.01, 0.5]))

## simulation # NB. first define the SIS 'model' variable, per above
x = gillespie_sim(model, theta)	    # run simulation
println(plot_trajectory(x))			# plot (optional)

## ARQMCMC
sample_interval = [0.0005, 0.02]
# results = BayesianWorkflows.run_arq_mcmc_analysis(model, prior, y, sample_interval)
# tabulate_results(results)
# println(plot_parameter_trace(results, 1))

## MBP MCMC
# results = BayesianWorkflows.run_mcmc_analysis(model, prior, y)
# tabulate_results(results)

## MBP IBIS
# results = BayesianWorkflows.run_mbp_ibis_analysis(model, prior, y)
# tabulate_results(results)

## SMC^2
# results = BayesianWorkflows.run_smc2_analysis(model, prior, y)
# tabulate_results(results)

## single model workflow
results = run_inference_workflow(model, prior, y; validation=BayesianWorkflows.C_ALG_NM_ARQ, sample_interval=sample_interval)
tabulate_results(results)
