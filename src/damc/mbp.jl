#### model based proposal (Pooley, 2015) ####

## iterator
# - ADJUST FOR SMC?
function iterate_mbp!(xf, pop_i::Array{Int64}, model::HiddenMarkovModel, obs_i::Int64, evt_i::Int64, time::Float64, xi::Particle)
    ## workspace
    # - zeros(model.n_events)
    lambda_f = Array{Float64,1}(undef, model.n_events)
    lambda_i = Array{Float64,1}(undef, model.n_events)
    lambda_d = Array{Float64,1}(undef, model.n_events)
    ## iterate until next observation
    while true
        tmax = evt_i > length(xi.trajectory) ? model.obs_data[obs_i].time : min(model.obs_data[obs_i].time, xi.trajectory[evt_i].time)
        model.rate_function(lambda_i, xi.theta, pop_i)
        while true
            model.rate_function(lambda_f, xf.theta, xf.final_condition)
            for i in eachindex(lambda_d)            # compute rate delta
                lambda_d[i] = max(lambda_f[i] - lambda_i[i], 0.0)
            end
            cumsum!(lambda_d, lambda_d)
            lambda_d[end] == 0.0 && break           # 0 rate test
            time -= log(rand()) / lambda_d[end]
            time > tmax && break                    # break if event time exceeded
            et = choose_event(lambda_d)             # else choose event type (init as final event)
            # xf.final_condition .+= model.fn_transition(et, xf.final_condition)
            model.transition!(xf.final_condition, et)
            push!(xf.trajectory, Event(time, et))   # add event to trajectory
        end
        ## handle event
        evt_i > length(xi.trajectory) && break
        xi.trajectory[evt_i].time > model.obs_data[obs_i].time && break
        et::Int64 = xi.trajectory[evt_i].event_type
        time = xi.trajectory[evt_i].time
        prob_keep = lambda_f[et] / lambda_i[et]
        if (prob_keep > 1.0 || prob_keep > rand())
            push!(xf.trajectory, Event(time, et))
            # xf.final_condition .+= model.fn_transition(et, xf.final_condition)
            model.transition!(xf.final_condition, et)
        end
        # pop_i .+= model.fn_transition(et, pop_i)
        model.transition!(pop_i, et)
        evt_i += 1
    end
    return evt_i
end

## initialise sequence for MBP
function initialise_trajectory!(xf, pop_i::Array{Int64}, model::HiddenMarkovModel, xi)
    # xf_trajectory::Array{Event,1}
    evt_i::Int64 = 1
    if model.t0_index == 0
        return evt_i
    else
        if xf.theta[model.t0_index] < xi.theta[model.t0_index]
            ## 'sim'
            lambda_f = zeros(model.n_events)                    # workspace
            t = xf.theta[model.t0_index]
            while true
                # compute rates:
                model.rate_function(lambda_f, xf.theta, xf.final_condition)
                cumsum!(lambda_f, lambda_f)
                lambda_f[end] == 0.0 && break                   # 0 rate test
                t -= log(rand()) / lambda_f[end]
                t > xi.theta[model.t0_index] && break           # break if prev t0 time exceeded
                et = choose_event(lambda_f)                     # else choose event type (init as final event)
                push!(xf.trajectory, Event(t, et))              # add event to trajectory
                # xf.final_condition .+= model.fn_transition(et, xf.final_condition)  # update population
                model.transition!(xf.final_condition, et)       # update population
                length(xf.trajectory) > MAX_TRAJ && (return evt_i)
            end
        else
            ## 'delete'
            while true
                evt_i > length(xi.trajectory) && break
                xi.trajectory[evt_i].time > xf.theta[model.t0_index] && break
                model.transition!(pop_i, xi.trajectory[evt_i].event_type)  # update population
                evt_i += 1                                                      # iterate event counter
            end
        end
        return evt_i
    end
end

## up to time y_i
function partial_model_based_proposal(model::HiddenMarkovModel, theta_f::Array{Float64,1}, xi::Particle, ymax::Int64)
    xf = Particle(theta_f, copy(xi.initial_condition), copy(xi.initial_condition), Event[], Distributions.logpdf(model.prior,theta_f), zeros(2))
    ## evaluate prior density
    if xf.prior == -Inf
        xf.log_like .= -Inf
        return xf
    else
        # debug && println("init ll: ", xf.log_like[2])
        ## make mbp
        pop_i = copy(xi.initial_condition) # GET RID *
        evt_i = initialise_trajectory!(xf, pop_i, model, xi)
        time::Float64 = model.t0_index == 0 ? 0.0 : max(xf.theta[model.t0_index], xi.theta[model.t0_index])
        for obs_i in 1:ymax
            ## iterate MBP
            evt_i = iterate_mbp!(xf, pop_i, model, obs_i, evt_i, time, xi)
            if length(xf.trajectory) > MAX_TRAJ # - HACK: MAX trajectory check
                xf.log_like[1] = -Inf
                return xf
            end
            time = model.obs_data[obs_i].time   ## handle observation
            xf.log_like[2] = model.obs_model(model.obs_data[obs_i], xf.final_condition, theta_f)
            model.obs_data[obs_i].obs_id > 0 && (xf.log_like[1] += xf.log_like[2])
        end
        return xf
    end
end

## model based proposal algorithm
function model_based_proposal(model::HiddenMarkovModel, theta_f::Array{Float64,1}, xi::Particle)
    ## make full mbp
    return partial_model_based_proposal(model, theta_f, xi, length(model.obs_data))
end
