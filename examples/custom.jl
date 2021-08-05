#### custom particle filter examples
using BayesianWorkflows
import Random
import Distributions

Random.seed!(1)

## variables
theta = [0.003, 0.1]
initial_condition = [100, 1]
data_fp = "data/pooley.csv"

## define a particle filter in form f(theta) (NB. use log PDF)
y = get_observations(data_fp) # val_seq=2:3
model = generate_model("SIS", initial_condition)
prior = Distributions.Product(Distributions.Uniform.(zeros(2), [0.01, 0.5]))
# pm = BayesianWorkflows.get_private_model(model, prior, y)
# pf = BayesianWorkflows.get_log_pdf_fn(pm)
particle_filter = get_particle_filter_lpdf(model, prior, obs_data)

## ARQMCMC
# sample_interval = [0.0005, 0.02]
# rs = BayesianWorkflows.run_arq_mcmc_analysis(m
