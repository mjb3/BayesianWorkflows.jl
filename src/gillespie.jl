#### simulation ####

## gillespie sim iteration
# - NEED TO REDO WITH MACROS ***************
function iterate_particle!(p::Particle, model::HiddenMarkovModel, time::Float64, y::Observation) #tmax::Float64
    cum_rates = Array{Float64, 1}(undef, model.n_events)
    while true
        model.rate_function(cum_rates, p.theta, p.final_condition)
        cumsum!(cum_rates, cum_rates)
        cum_rates[end] == 0.0 && break          # 0 rate test
        time -= log(rand()) / cum_rates[end]
        time > y.time && break                  # break if max time exceeded
        et = choose_event(cum_rates)            # else choose event type (init as final event)
        p.final_condition .+= model.fn_transition(et)  # update population
        push!(p.trajectory, Event(time, et))    # add event to sequence
        if length(p.trajectory) > MAX_TRAJ      # HACK
            p.log_like[1] = -Inf
            return p.log_like[1]
        end
    end
    output = model.obs_model(y, p.final_condition, p.theta)
    y.obs_id > 0 && (p.log_like[1] += output)
    return output
end

## gillespie sim iteration (full state vectors)
function iterate_particle!(p::Particle, pop_v::Array{Array{Int64},1}, model::HiddenMarkovModel, time::Float64, y::Observation, cmpt_ll::Bool = true) # GET RID *
    cum_rates = Array{Float64, 1}(undef, model.n_events)
    while true
        model.rate_function(cum_rates, p.theta, p.final_condition)
        # C_DEBUG && println(" r := ", cum_rates)
        cumsum!(cum_rates, cum_rates)
        cum_rates[end] == 0.0 && break          # 0 rate test
        time -= log(rand()) / cum_rates[end]
        time > y.time && break                  # break if max time exceeded
        et = choose_event(cum_rates)            # else choose event type (init as final event)
        p.final_condition .+= model.fn_transition(et)  # update population
        push!(p.trajectory, Event(time, et))    # add event to sequence
        push!(pop_v, copy(p.final_condition))
    end
    cmpt_ll && (p.log_like[1] += model.obs_model(y, p.final_condition, p.theta))
end

## generate 'blank' observations for sim
# HACK: need to replace C_DEFAULT_OBS_PROP with [optional] random number at some point... **
C_DEFAULT_OBS_PROP = 1.0
function generate_observations(tmax::Float64, num_obs::Int64, n_states::Int64, n_test_types::Int64)
    obs = Observation[]
    t = collect(tmax / num_obs : tmax / num_obs : tmax)
    for i in eachindex(t)
        test_type = rand(1:n_test_types)
        push!(obs, Observation(t[i], test_type, C_DEFAULT_OBS_PROP, zeros(Int64, n_states)))
    end
    return obs
end

## run sim and return trajectory (full state var)
# reconstruct with macros? ***
function gillespie_sim(model::HiddenMarkovModel, theta::Array{Float64, 1}, observe::Bool) #, y::Observations
    # initialise some things
    y = deepcopy(model.obs_data)
    ic = model.fn_initial_state(theta)
    p = Particle(theta, ic, copy(ic), Event[], Distributions.logpdf(model.prior, theta), zeros(2))
    pop_v = Array{Int64}[]
    t = model.t0_index == 0 ? 0.0 : theta[model.t0_index]
    # run
    for i in eachindex(y)
        iterate_particle!(p, pop_v, model, t, y[i])
        # observe && (y[i].val .= model.obs_function(y[i], p.final_condition, theta))
        observe && model.obs_function(y[i], p.final_condition, theta)
        t = y[i].time
    end
    # return sequence
    return SimResults(model.name, p, pop_v, y)
end

#### BTB testing scenario code ####
## get next observation
# nb. seed initial in sim
# ditto with IFN
function get_next_obs(obs::Array{Observation,1})
    C_INT_SI = 60
    C_INT_FU = 180
    C_INT_RH = 360
    if obs[end].val[1] > 0                     ## SI
        return Observation(obs[end].time + C_INT_SI, 2, C_DEFAULT_OBS_PROP, zeros(Int64,1))
    else
        if obs[end].obs_id > 1              ## breakdown in progress
            if obs[length(obs) - 1].val[1] > 0 ## SI
                return Observation(obs[end].time + C_INT_SI, 2, C_DEFAULT_OBS_PROP, zeros(Int64,1))
            else                            ## cleared - follow up
                return Observation(obs[end].time + C_INT_FU, 1, C_DEFAULT_OBS_PROP, zeros(Int64,1))
            end
        else                                ## schedule RHT
            return Observation(obs[end].time + C_INT_RH, 1, C_DEFAULT_OBS_PROP, zeros(Int64,1))
        end
    end
end
##
function init_obs()
    y = Observation[]
    push!(y, Observation(0.0, 1, C_DEFAULT_OBS_PROP, zeros(Int64,1)))
    return y
end
## TO BE MOVED
# NB. where is this called from? ***********
function gillespie_scenario(model::HiddenMarkovModel, theta::Array{Float64, 1}; tmax::Float64 = 720.0, ifn_y::Int64 = 0)
    ## initialise some things
    y = init_obs()
    ic = model.fn_initial_state(theta)
    p = Particle(theta, ic, copy(ic), Event[], [model.fn_log_prior(theta), 0.0])
    pop_v = Array{Int64}[]
    t = model.t0_index == 0 ? 0.0 : theta[model.t0_index]
    ## run
    while y[end].time < tmax
        iterate_particle!(p, pop_v, model, t, y[end], false)
        y[end].val .= model.obs_function(y[end], p.final_condition, theta)
        t = y[end].time
        if ifn_y == length(y) ## WHAT?*
            ## IFN
            push!(y, Observation(t + 1, 3, C_DEFAULT_OBS_PROP, zeros(Int64,1)))
        else
            ## schedule next test
            push!(y, get_next_obs(y))
        end
    end
    ## return sequence
    return SimResults(model.name, p, pop_v, y)
end
#### #### #### #### #### #### ####

## for inference
function generate_x0(model::HiddenMarkovModel, ntries = 50000)#, theta::Array{Float64, 1}
    theta = rand(model.prior)
    for i in 1:ntries
        x0 = gillespie_sim(model, theta, false).particle
        x0.log_like[1] != -Inf && return x0
    end
    ## ADD PROPER ERR HANDLING ***
    println("WARNING: having an issue generating a valid initial trajectory\n- parameters: ", theta)
    return generate_x0(model, ntries)
end

"""
    gillespie_sim(model, parameters; tmax = 100.0, num_obs = 5)

Run a Doob-Gillespie (DGA) simulation based on `model`.

Returns a SimResults type containing the trajectory and observations data, or an array of the same if `n_sims` > 1.

**Parameters**
- `model`       -- `DPOMPModel` (see [DCTMPs.jl models]@ref).
- `parameters`  -- model parameters.
**Optional**
- `tmax`        -- maximum time (default: 100.)
- `n_obs`       -- the number of observations to draw (default: 5.)
- `n_sims`      -- number of simulations to draw (default: 1.)

**Example**
```@repl
using DiscretePOMP
m = generate_model("SIR", [50, 1, 0])
x = DiscretePOMP.gillespie_sim(model, [0.005, 0.12])
println(DiscretePOMP.plot_trajectory(x))
```

"""
function gillespie_sim(model::DPOMPModel, parameters::Array{Float64, 1}; tmax::Float64 = 100.0, num_obs::Int64 = 5, n_sims::Int64 = 1, n_test_types::Int64=1)
    mdl = get_private_model(model, generate_uniform_prior(length(parameters)), y)
    if n_sims == 1
        print("Running: ", model.name, " DGA for θ := ", parameters)
        y = generate_observations(tmax, num_obs, get_pop_size(model, parameters), n_test_types)
        output = gillespie_sim(mdl, parameters, true)
        println("- finished.")
        return output
    else
        print("Running: ", model.name, " DGA for θ := ", parameters, " x ", n_sims)
        output = Array{SimResults,1}(undef, n_sims)
        for i in eachindex(output)
            y = generate_observations(tmax, num_obs, get_pop_size(model, parameters), n_test_types)
            output[i] = gillespie_sim(mdl, parameters, true)
        end
        println("- finished.")
        return output
    end
end
# - multiple parameter sets
function gillespie_sim(model::DPOMPModel, parameters::Array{Float64, 2}; tmax::Float64 = 100.0, num_obs::Int64 = 5, n_test_types::Int64=1)
    print("Running: ", model.name, " DGA for θ := ", parameters, " x ", n_sims)
    output = Array{SimResults,1}(undef, size(parameters, 2))
    for i in eachindex(output)
        y = generate_observations(tmax, num_obs, get_pop_size(model, parameters), n_test_types)
        output[i] = gillespie_sim(mdl, parameters[:,i], true)
    end
    println("- finished.")
    return output
end
