#### common hmm types
import Base: isless

## event
"""
    Event

Requires no explanation.

**Fields**
- `time`        -- the time of the event.
- `event_type`  -- indexes the rate function and transition matrix.

"""
struct Event
    time::Float64
    event_type::Int64
end
# - for sorting:
isless(a::Event, b::Event) = isless(a.time, b.time)

## observation tuple
"""
    Observation

A single observation. Note that by default `val` has the same size as the model state space. However that is not necessary - it need only be compatible with the observation model.

**Fields**
- `time`        -- similar to `Event.time`, the time of the observation.
- `obs_id`      -- <1 if not a resampling step.
- `prop`        -- optional information for the observation model.
- `val`         -- the observation value.

"""
struct Observation
    time::Float64
    obs_id::Int64   # <1 if not a resampling step
    prop::Float64 #df: 1.0
    val::Array{Int64,1}
end
# - for sorting:
isless(a::Observation, b::Observation) = isless(a.time, b.time)

## a single realisation of the model
"""
    Particle

E.g. the main results of a simulation including the initial and final conditions, but not the full state trajectory.

**Fields**
- `theta`               -- e.g. simulation parameters.
- `initial_condition`   -- initial system state.
- `final_condition`     -- final system state.
- `trajectory`          -- the event history.
- `log_like`            -- trajectory log likelihood, mainly for internal use.

"""
struct Particle
    theta::Array{Float64,1}
    initial_condition::Array{Int64}
    final_condition::Array{Int64}
    trajectory::Array{Event,1}
    prior::Float64              # log prior;
    log_like::Array{Float64,1}  # full log like g(x); [latest] marginal g(x) / proposal likelihood (SMC / MCMC)
end

# - for dependent f/g
# struct DFGParticle
#     theta::Array{Float64,1}
#     initial_condition::Array{Int64}
#     final_condition::Array{Int64}
#     trajectory::Array{Event,1}
#     log_like::Array{Float64,1}  # prior, g(x)
#     g_trans::Array{Int64,2}
# end

## results of gillespie sim
"""
    SimResults

The results of a simulation, including the full state trajectory.

**Fields**
- `model_name`      -- string, e,g, `"SIR"`.
- `particle`        -- the 'trajectory' variable, of type `Particle`.
- `population`      -- records the complete system state over time.
- `observations`    -- simulated observations data (an `Array` of `Observation` types.)

"""
struct SimResults
    model_name::String
    particle::Particle
    population::Array{Array{Int64},1}
    observations::Array{Observation,1}
end
