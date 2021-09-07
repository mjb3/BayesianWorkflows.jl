### SEIRD model example
import BayesianWorkflows
import Distributions
import Random

Random.seed!(0)

## define model
function get_model()
    MODEL_NAME = "SEIRDs"
    N_EVENT_TYPES = 6
    # - discrete state space
    SUSCEPTIBLE = 1
    EXPOSED = 2
    INFECTIOUS = 3
    RECOVERED = 4
    DEATH = 5
    ALIVE = 1:4
    OBSERVED_STATES = [INFECTIOUS, DEATH]
    # - observation probabilities
    PROB_DETECTION_I = 0.6
    PROB_DETECTION_D = 0.95
    # - model parameters
    T_ZERO = 0  # I.E. NO INITIAL INFECTION TIME PARAMETER (ASSUME t0=0.0)
    CONTACT = 1
    LATENCY = 2
    RECOVER = 3
    BIRTH_DEATH = 4
    # - rate function
    function seird_rf(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        output[1] = parameters[CONTACT] * population[SUSCEPTIBLE] * population[INFECTIOUS] / population[ALIVE]
        output[2] = parameters[LATENCY] * population[EXPOSED]
        output[3] = parameters[RECOVER] * population[INFECTIOUS]
        output[4] = 2 * parameters[BIRTH_DEATH] * population[ALIVE]
    end
    # - transition matrix and function
    tm = [-1 1 0 0 0; 0 -1 1 0 0; 0 0 -1 1 0]
    function transition()
        body
    end
