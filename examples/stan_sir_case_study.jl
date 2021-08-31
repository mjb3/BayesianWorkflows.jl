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
df = CSV.read(data_fp, DataFrames.DataFrame)
println("----\ndf: ", df, "----\ny: ", y)

## variables
population_size = 763
initial_condition = [population_size - 1, 1, 0]
# theta = [0.003, 0.1]                # model parameters

## generate transmission model
STATE_S = 1
STATE_I = 2
STATE_R = 3
PRM_BETA = 1
PRM_LAMBDA = 2
PRM_PHI = 3
model = generate_model("SIR", initial_condition; freq_dep=true, t0_index=4)

## prior distribution
function get_prior()
    pr_beta = Truncated(Normal(2.0, 1.0), 0.0, Inf)
    pr_lambda = Truncated(Normal(0.4, 0.5), 0.0, Inf)
    phi_inv = Truncated(Exponential(5.0), 1.0, Inf)
    t0_l = Dates.DateTime.(["1978-01-12", "1978-01-22"], "yyyy-mm-dd")
    t0_b::Vector{Float64} = Dates.value.(t0_l)
    println("t0_l: ", t0_l, "\nt0_b: ", t0_b)
    t0 = Uniform(t0_b...)
    return Product([pr_beta, pr_lambda, phi_inv, t0])
end
prior = get_prior()

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

## fit the model and check results
# results = run_inference_workflow(model, prior, y)
# tabulate_results(results)

## predict
# - i.e. resample parameters from posterior samples and simulate

## prior predictive check
# - same but sample parameters from prior
println("\nprior samples:\n", rand(prior, 10))
