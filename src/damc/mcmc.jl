#### data-augmented MCMC

## helpers
# sample_space(theta_init::Array{Float64,2}, steps::Int64) = zeros(size(theta_init,1), steps, size(theta_init,2))

## initialise mcmc algorithm
const C_INITIAL = 0.1     # initial proposal scalar
macro initialise_mcmc()
    esc(quote
    steps::Int64 = size(theta, 2) #GET RID
    ADAPT_INTERVAL = adapt_period / C_MCMC_ADAPT_INTERVALS   # interval between adaptation steps
    xi = x0
    ## covar matrix
    covar = zeros(length(xi.theta), length(xi.theta))
    for i in eachindex(xi.theta)
        covar[i,i] = xi.theta[i] == 0.0 ? 1 : (xi.theta[i]^2)
    end
    propd = Distributions.MvNormal(covar)
    c = C_INITIAL                           # autotune
    theta[:,1,mc] .= xi.theta               # initial sample
    a_cnt = zeros(Int64, 2)
    a_cnt[1] = 1
    end)
end

## end of adaption period
macro mcmc_adapt_period()
    esc(quote
    if i % ADAPT_INTERVAL == 0
        covar = Distributions.cov(transpose(theta[:,1:i,mc]))
        propd = get_prop_density(covar, propd)
    end
    end)
end

## adaptation
macro met_hastings_adapt()
    esc(quote
    ## adaptation
    if (!fin_adapt || i < adapt_period)
        c *= (accepted ? 1.002 : 0.999)
        @mcmc_adapt_period      # end of adaption period
    end
    end)
end

## adaptation
macro gibbs_adapt()
    esc(quote
    ## adaptation
    if (!fin_adapt || i < adapt_period)
        pp && (c *= (accepted ? 1.002 : 0.999))
        @mcmc_adapt_period      # end of adaption period
    end
    end)
end

## acceptance handling (also used by GS)
macro mcmc_handle_mh_step()
    esc(quote
    if accepted
        xi = xf
        a_cnt[i > adapt_period ? 2 : 1] += 1
    end
    theta[:,i,mc] .= xi.theta
    end)
end

## analyse results
macro mcmc_tidy_up()
    esc(quote
    rejs = handle_rej_samples(samples, adapt_period)
    gd = gelman_diagnostic(samples, adapt_period)         # run convergence diagnostic
    output = MCMCSample(rejs, adapt_period, gd.psrf, time_ns() - start_time)
    println("- finished in ", print_runtime(output.run_time), ". E(θ) := ", output.samples.mu)
    end)
end

## compute full log likelihood
# - used by gibbs sampler
function compute_full_log_like!(model::HiddenMarkovModel, p::Particle)
    t = model.t0_index > 0 ? p.theta[model.t0_index] : 0.0
    lambda = zeros(model.n_events)
    if length(p.trajectory) > 0 && p.trajectory[1].time < t
        p.log_like[1] = -Inf                    # void sequence
    else
        p.log_like[1] = 0.0                     # reset and initialise
        p.final_condition .= p.initial_condition
        evt_i = 1
        for obs_i in eachindex(model.obs_data)  # handle trajectory segments
            while evt_i <= length(p.trajectory)
                p.trajectory[evt_i].time > model.obs_data[obs_i].time && break
                model.rate_function(lambda, p.theta, p.final_condition)
                try
                    p.log_like[1] += log(lambda[p.trajectory[evt_i].event_type]) - (sum(lambda) * (p.trajectory[evt_i].time - t))
                catch
                    C_DEBUG && println("ERROR:\n theta := ", p.theta, "; pop := ", p.final_condition, "; r := ", lambda)
                    p.log_like[1] = -Inf
                    return
                end
                # p.final_condition .+= model.fn_transition(p.trajectory[evt_i].event_type, p.final_condition)
                model.transition!(p.final_condition, p.trajectory[evt_i].event_type)
                if any(x->x<0, p.final_condition)
                    p.log_like[1] = -Inf
                    return
                else
                    t = p.trajectory[evt_i].time
                    evt_i += 1
                end
            end
            model.rate_function(lambda, p.theta, p.final_condition)
            p.log_like[1] += model.obs_model(model.obs_data[obs_i], p.final_condition, p.theta)
            p.log_like[1] -= sum(lambda) * (model.obs_data[obs_i].time - t)
            p.log_like[1] == -Inf && return
            t = model.obs_data[obs_i].time
        end
    end
end

## Single particle adaptive Metropolis Hastings algorithm
function met_hastings_alg!(theta::Array{Float64,3}, mc::Int64, model::HiddenMarkovModel, adapt_period::Int64, x0::Particle, proposal_alg::Function, fin_adapt::Bool)
    @initialise_mcmc
    for i in 2:steps            # met_hastings_step
        xf = proposal_alg(model, get_mv_param(propd, c, theta[:,i-1,mc]), xi)
        if (xf.prior == -Inf || xf.log_like[1] == -Inf)
            accepted = false    # reject automatically
        else                    # accept or reject
            # NB: [2] == full g(x) log like
            mh_prob::Float64 = exp(xf.prior - xi.prior) * exp(xf.log_like[1] - xi.log_like[1])
            accepted = (mh_prob > 1 || mh_prob > rand())
        end
        @mcmc_handle_mh_step    # handle accepted proposals
        # accepted && (xi = xf)
        # theta[:,i,mc] .= xi.theta
        @met_hastings_adapt     # adaptation
    end ## end of MCMC loop
    C_DEBUG && print(" - Xn := ", length(xi.trajectory), " events; ll := ", xi.log_like, " - ")
    return a_cnt
end

## Single particle adaptive Gibbs sampler - TO BE FINISHED ****
function gibbs_mh_alg!(theta::Array{Float64,3}, mc::Int64, model::HiddenMarkovModel, adapt_period::Int64, x0::Particle, proposal_alg::Function, fin_adapt::Bool, ppp::Float64, adapt_prop_alg::Function)
    @initialise_mcmc
    prop_fn = adapt_prop_alg
    for i in 2:steps            # Gibbs
        pp = rand() < ppp
        if pp                   # parameter proposal
            theta_f = get_mv_param(propd, c, theta[:,i-1,mc])
            xf = Particle(theta_f, xi.initial_condition, xi.final_condition, xi.trajectory, Distributions.logpdf(model.prior, theta_f), zeros(2))
        else                    # trajectory proposal
            xf = prop_fn(xi)
        end
        (xf.prior == -Inf || xf.log_like[2] == -Inf) || compute_full_log_like!(model, xf)
        if (xf.prior == -Inf || sum(xf.log_like) == -Inf)
            accepted = false    # reject automatically
        else                    # accept or reject
            # NB: [3] == proposal log like
            mh_prob::Float64 = exp(xf.prior - xi.prior) * exp(sum(xf.log_like[1:2]) - xi.log_like[1])
            accepted = (mh_prob > 1 || mh_prob > rand())
        end
        @mcmc_handle_mh_step    # handle accepted proposals
        @gibbs_adapt            # adaptation
        i == Int64(floor(adapt_period * 0.2)) && (prop_fn = proposal_alg)
    end ## end of MCMC loop
    C_DEBUG && print(" - Xn := ", length(xi.trajectory), " events; ll := ", xi.log_like, " - ")
    return a_cnt
end

## standard DA-MCMC
function run_std_mcmc(model::HiddenMarkovModel, theta_n::Int64, steps::Int64, adapt_period::Int64, fin_adapt::Bool, ppp::Float64, mvp::Int64)
    function x0_prop()
        x0 = generate_x0(model)         # simulate initial particle
        compute_full_log_like!(model, x0)           # NB. sim initialises with OM ll only
        return x0
    end
    trajectory_prop =  get_std_mcmc_proposal_fn(model, mvp)
    adapt_tp = get_std_mcmc_proposal_fn(model, 2)
    println("Running: ", n_chains, "-chain ", steps, "-sample ", fin_adapt ? "finite-" : "", "adaptive DA-MCMC analysis (model: ", model.name, ")")
    start_time = time_ns()
    # samples = sample_space(theta_init, steps)
    samples = zeros(theta_n, steps, n_chains)
    for i in 1:n_chains
        print(" initialising chain ", i)
        x0 = x0_prop()
        ## run inference
        C_DEBUG && print(" with x0 := ", x0.theta, " (", length(x0.trajectory), " events)")
        a_cnt = gibbs_mh_alg!(samples, i, model, adapt_period, x0, trajectory_prop, fin_adapt, ppp, adapt_tp)
        println(" - complete (AAR := ", round(100 * a_cnt[2] / (steps - adapt_period), digits = 1), "%)")
    end
    @mcmc_tidy_up
    return output
end

## MBP MCMC
function run_mbp_mcmc(model::HiddenMarkovModel, n_chains::Int64, steps::Int64, adapt_period::Int64, fin_adapt::Bool)
    theta_n = length(rand(model.prior)) # hacky...
    start_time = time_ns()
    # samples = sample_space(theta_init, steps)
    samples = zeros(theta_n, steps, n_chains)
    println("Running: ", n_chains, "-chain ", steps, "-sample ", fin_adapt ? "finite-" : "", "adaptive MBP-MCMC analysis (model: ", model.name, ")")
    for i in 1:n_chains
        print(" initialising chain ", i)
        x0 = generate_x0(model)    # simulate initial particle
        ## run inference
        C_DEBUG && print( " with x0 := ", x0.theta, " (", length(x0.trajectory), " events)")
        a_cnt = met_hastings_alg!(samples, i, model, adapt_period, x0, model_based_proposal, fin_adapt)
        println(" - complete (AAR := ", round(100 * a_cnt[2] / (steps - adapt_period), digits = 1), "%)")
    end
    @mcmc_tidy_up
    return output
end
