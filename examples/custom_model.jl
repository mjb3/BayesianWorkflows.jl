### SEIRD model example
import BayesianWorkflows
import Distributions
import Random

Random.seed!(0)

## define model
function get_model()
    MODEL_NAME = "SEIRD"
    N_EVENT_TYPES = 4
    # - discrete state space
    SUSCEPTIBLE = 1
    EXPOSED = 2
    INFECTIOUS = 3
    RECOVERED = 4
    DEATH = 5
    OBSERVED_STATES = [INFECTIOUS, DEATH]
    # - observation probabilities
    PROB_DETECTION_I = 0.6
    PROB_DETECTION_D = 0.95
    # - model parameters
    T_ZERO = 0  # I.E. NO INITIAL INFECTION TIME PARAMETER (ASSUME t0=0.0)
    CONTACT = 1
    ASYMPT = 2
    RECOVER = 3
    MORTALITY = 4
    # - rate function
    function seird_rf(output, parameters::Vector{Float64}, population::Array{Int64, 1})
        output[1] = parameters[CONTACT] * population[SUSCEPTIBLE] * population[INFECTIOUS]
        output[2] = 1 / parameters[ASYMPT] * population[EXPOSED]
        output[3] = 1 / parameters[RECOVER] * population[INFECTIOUS]
        output[4] = parameters[MORTALITY] * population[INFECTIOUS]
    end
    # - transition matrix and function
    tm = [-1 1 0 0 0; 0 -1 1 0 0; 0 0 -1 1 0; 0 0 -1 0 1]
    fnt = BayesianWorkflows.generate_trans_fn(tm)
    # - initial condition
    fnic(parameters::Vector{Float64}) = [1000, 0, 10, 0, 0]
    # - observation function
    function obs_fn!(y::BayesianWorkflows.Observation, population::Array{Int64,1}, parameters::Vector{Float64 })
        di = Distributions.Binomial(population[INFECTIOUS], PROB_DETECTION_I)
        dd = Distributions.Binomial(population[DEATH], PROB_DETECTION_I)
        y.val[OBSERVED_STATES] .= [rand(di), rand(dd)]
    end
    # - observation model
    # obs_model = BayesianWorkflows.partial_gaussian_obs_model(2.0; seq = 3)
    function obs_model(y::BayesianWorkflows.Observation, population::Array{Int64,1}, theta::Array{Float64,1})
        d = Distributions.Binomial.(population[OBSERVED_STATES], [PROB_DETECTION_I, PROB_DETECTION_D])
        return Distributions.logpdf(Distributions.Product(d), y.val[OBSERVED_STATES])
    end
    # - construct model and return
    return BayesianWorkflows.DPOMPModel(MODEL_NAME, N_EVENT_TYPES, seird_rf, fnic, fnt, obs_model, obs_fn!, T_ZERO)
end

## simulate and return some 'observations'
function run_simulation(model::BayesianWorkflows.DPOMPModel)
    parameters = [0.01, 0.25, 0.2, 0.001]
    x = BayesianWorkflows.gillespie_sim(model, parameters; tmax=10.0, num_obs=20)
    println(BayesianWorkflows.plot_trajectory(x))
    return x.observations
end

## single model inference
function single_model_inference()
    model = get_model()
    ## get some simulated observations
    y = run_simulation(model)
    # println("y: ", y)
    ## uniform prior
    # lb =
    ub = [0.1, 1.0, 1.0, 0.01]
    prior = Distributions.Product(Distributions.Uniform.(zeros(4), ub))
    ## run analysis
    results = BayesianWorkflows.run_inference_analysis(model, prior, y)
    BayesianWorkflows.tabulate_results(results)
end

## multi model inference
# TBA

## run
single_model_inference()
