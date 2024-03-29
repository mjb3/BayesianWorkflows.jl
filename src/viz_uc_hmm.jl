#### unicode visualisation (HMM)

## trajectory
"""
    plot_trajectory(x; plot_index=[:])

Plot the trajectory of a DGA simulation using [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl).

The only input parameter required is `x` of type `SimResults`, i.e. from a call to `gillespie_sim`. All system states are plotted by default, but a subset can be specified by passing an integer array to the `plot_index` option, which contains the indices of the desired subset. E.g. [1,2] for the first two 'compartments' only.
"""
function plot_trajectory(x::SimResults; plot_index=collect(1:length(x.particle.initial_condition)))
    ## collect time and population
    t = zeros(length(x.particle.trajectory) + 1)
    pop = zeros(Int64, length(x.particle.trajectory) + 1, length(plot_index))
    pop[1,:] .= x.particle.initial_condition[plot_index]
    for i in eachindex(x.particle.trajectory)
        t[i+1] = x.particle.trajectory[i].time
        pop[i+1, :] .= x.population[i][plot_index]
    end
    ## plot
    p = UnicodePlots.lineplot(t, pop[:,1], title = string(x.model_name, " simulation"), name = string(x.model_name[plot_index[1]]), ylim = [0, maximum(pop) + 1])#
    for i in 2:size(pop, 2)
        UnicodePlots.lineplot!(p, t, pop[:,i], name = string(x.model_name[plot_index[i]]))
    end
    UnicodePlots.xlabel!(p, "time")
    UnicodePlots.ylabel!(p, "population")
    return p
end

#### observations data
max_obs(observations::Vector{Vector{Observation}}, val_index::Int64) = maximum([observations[i][j].val[val_index] for i in eachindex(observations) for j in eachindex(observations[i])])

## plot quantiles
C_PLT_OBS_QT_TTL = "Observation quantiles"
"""
    plot_observation_quantiles(x; plot_index=1, quantiles=[0.25, 0.5, 0.75])

Plot perecentile range for a common set of observation data sets using [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl).

The only input parameter required is a `Vector` of type `SimResults` (or `Vector{Observation}`), i.e. from a call to `gillespie_sim`. The first observation value is plotted by default, but another can be specified by passing an integer to the `plot_index` option.
"""
function plot_observation_quantiles(observations::Vector{Vector{Observation}}; plot_index::Int64=1, quantiles::Vector{Float64}=[0.25, 0.5, 0.75], title::String=C_PLT_OBS_QT_TTL)
    mx = max_obs(observations, plot_index)
    y = get_observation_quantiles(observations, plot_index, quantiles)
    # t1 = [y.time for y in observations[1]]
    p = UnicodePlots.lineplot(y.t, y.y[:,1], ylim = [0, mx + 1], title=title, name=string(quantiles[1]))
    for i in 2:length(quantiles)
        # t = [y.time for y in observations[i]]
        UnicodePlots.lineplot!(p, y.t, y.y[:,i], name=string(quantiles[i]))
    end
    return p
end
# - vector of simulation trajectories
function plot_observation_quantiles(x::Vector{SimResults}; plot_index::Int64=1, quantiles::Vector{Float64}=[0.25, 0.5, 0.75], title::String=C_PLT_OBS_QT_TTL)
    y = [xx.observations for xx in x]
    return plot_observation_quantiles(y; plot_index=plot_index, quantiles=quantiles, title=title)
end

C_PLT_OBS_TTL = "Observations trajectory"
"""
    plot_observations(x; plot_index=1)

Plot the trajectory of observation values for one or more [simulated or real] data sets using [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl).

The only input parameter required is `x` of type `SimResults`, i.e. from a call to `gillespie_sim`. The first observation value is plotted by default, but another can be specified by passing an integer to the `plot_index` option.
"""
function plot_observations(observations::Vector{Vector{Observation}}; plot_index=1, date_type=Float64, title=C_PLT_OBS_TTL)
    mx = max_obs(observations, plot_index)
    # mx = maximum([observations[i][j].val[plot_index] for i in eachindex(observations) for j in eachindex(observations[i])])
    ## plot
    t1 = [observations[1][i].time for i in eachindex(observations[1])]
    y1 = [observations[1][i].val[plot_index] for i in eachindex(observations[1])]
    p = UnicodePlots.lineplot(t1, y1, title=title, ylim = [0, mx + 1])#, name="y"
    for i in 2:length(observations)
        t = [observations[i][j].time for j in eachindex(observations[i])]
        y = [observations[i][j].val[plot_index] for j in eachindex(observations[i])]
        UnicodePlots.lineplot!(p, t, y)#, name = string(x.model_name[plot_index[i]])
    end
    UnicodePlots.xlabel!(p, "time")
    UnicodePlots.ylabel!(p, "y")
    return p
end
# -
function plot_observations(observations::Vector{Observation}; plot_index=1, date_type=Float64, title=C_PLT_OBS_TTL)
    return plot_observations([observations]; plot_index=plot_index, date_type=date_type, title=title)
end
# -
function plot_observations(x::SimResults; plot_index=1, date_type=Float64, title=C_PLT_OBS_TTL)
    return plot_observations(x.observations; plot_index=plot_index, date_type=date_type, title=title)
end
# -
function plot_observations(x::Vector{SimResults}; plot_index=1, date_type=Float64, title=C_PLT_OBS_TTL)
    y = Vector{Vector{Observation}}(undef, length(x))
    y .= [x[i].observations for i in eachindex(x)]
    return plot_observations(y; plot_index=plot_index, date_type=date_type, title=title)
end

## MCMC
# function plot_parameter_trace(sample::MCMCSample, parameter::Int64)
#     return plot_parameter_trace(sample.samples, parameter)
# end

## ARQ
function plot_parameter_trace(sample::ARQMCMCSample, parameter::Int64)
    return ARQMCMC.plot_parameter_trace(sample.rej_sample.samples, parameter)
end
function plot_parameter_trace(sample::ARQMCMCSample)
    return plot_parameter_trace.([sample], [i for i in eachindex(sample.sample_interval)])
end

## all parameters
function plot_parameter_trace(sample::RejectionSample)
    return plot_parameter_trace.([sample], [i for i in eachindex(sample.mu)])
end
# - calls the above
# function plot_parameter_trace(sample::MCMCSample)
#     return plot_parameter_trace(sample.samples)
# end

## marginal
"""
    plot_parameter_marginal(sample, parameter)

Plot the marginal distribution of samples from an MCMC analysis for a given model `parameter` using [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl).

**Parameters**
- `results`     -- Results object, e.g. of type `MCMCSample`.
- `parameter`   -- the index of the model parameter to be plotted.
- `adapt_period`-- Adaptation period to be discarded, only required for `RejectionSample`.
**Optional**
- `use_is`      -- Resample IS rather than using MCMC [re]samples (`ARQMCMCSample` results only.)

"""
function plot_parameter_marginal(sample::RejectionSample, parameter::Int64, adapt_period::Int64, nbins::Int64)
    x = vec(sample.theta[parameter, (adapt_period+1):size(sample.theta, 2), :])
    p = UnicodePlots.histogram(x, nbins = nbins)
    UnicodePlots.ylabel!(p, string("θ", Char(8320 + parameter)))
    UnicodePlots.xlabel!(p, "samples")
    return p
end

## MCMC
function plot_parameter_marginal(sample::MCMCSample, parameter::Int64; nbins = 20)
    return plot_parameter_marginal(sample.samples, parameter, sample.adapt_period, nbins)
end

## resampler
function plot_parameter_marginal(sample::ImportanceSample, parameter::Int64; nbins = 20)
    rs = resample(sample)
    return plot_parameter_marginal(rs, parameter, 0, nbins)
end

## heatmap
"""
    plot_parameter_heatmap(mcmc, x_parameter, y_parameter)

Plot the marginal distribution of samples from an MCMC analysis for two model parameters using [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl).

**Parameters**
- `mcmc`        -- `MCMCResults`, e.g. from a call to `run_met_hastings_mcmc`.
- `x_parameter`   -- the index of the model parameter to be plotted on the x axis.
- `y_parameter`   -- the index of the model parameter to be plotted on the y axis.

"""
function plot_parameter_heatmap(sample::RejectionSample, x_parameter::Int64, y_parameter::Int64, adapt_period::Int64)
    x = vec(sample.theta[x_parameter, (adapt_period+1):size(sample.theta,2), :])
    y = vec(sample.theta[y_parameter, (adapt_period+1):size(sample.theta,2), :])
    p = UnicodePlots.densityplot(x, y, color = :red)
    UnicodePlots.xlabel!(p, string("θ", Char(8320 + x_parameter)))
    UnicodePlots.ylabel!(p, string("θ", Char(8320 + y_parameter)))
    return p
end

# function get_df_lim(theta::Array{Float64,3}, p::Int64)
#     return [floor(minimum(theta[p,:,:]), sigdigits = 1), ceil(maximum(theta[p,:,:]), sigdigits = 1)]
# end

## MCMC
function plot_parameter_heatmap(sample::MCMCSample, x_parameter::Int64, y_parameter::Int64)
    return plot_parameter_heatmap(sample.samples, x_parameter, y_parameter, sample.adapt_period)
end

## resampler
function plot_parameter_heatmap(sample::ImportanceSample, x_parameter::Int64, y_parameter::Int64)
    rs = resample(sample)
    return plot_parameter_heatmap(rs, x_parameter, y_parameter, 0)
end



## model evidence comparison
#- `boxplot`   -- `true` for a series of boxplots, else a simple UnicodePlots.barplot showing only the average BME for each model variant (default.)
"""
    plot_model_comparison(results)

Plot the Bayesian model evidence (BME) from a [multi-] model inference workflow, using [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl).

**Parameters**
- `results`   -- `ModelComparisonResults`, i.e. from a call to `run_model_comparison_analysis`.

"""
function plot_model_comparison(results::Array{SingleModelResults, 1})
    c_plot_title = "Estimated model [log] evidence"
    # this is a HACK - need to handle bad results better...
    try
        names = [r.model.name for r in results]
        ml = [r.ibis.bme[1] for r in results]
        return UnicodePlots.barplot(names, round.(ml; digits = 1), title = c_plot_title, xlabel = C_LBL_BME)
    catch err
        println("ERROR: couldn't produce plot :=\n", err)
        return nothing
    end
end
# function plot_model_comparison(results::ModelComparisonResults, boxplot = false)
#     c_plot_title = "Estimated model evidence"
#     # this is a HACK - need to handle bad results better...
#     try
#         if boxplot
#             return UnicodePlots.boxplot(results.names, [results.bme[:, i] for i in 1:size(results.bme,2)], title = c_plot_title, xlabel = C_LBL_BME)
#         else
#             return UnicodePlots.barplot(results.names, round.(results.mu; digits = 1), title = c_plot_title, xlabel = C_LBL_BME)
#         end
#     catch err
#         println("ERROR: couldn't produce plot :=\n")
#         return err # HACK - FIX THIS ***
#     end
# end

## for priors
function plot_pdf(d::Distributions.Distribution, mx = 1.0, mn = 0.0, np = 1000)
    pd = zeros(np)
    x = zeros(np)
    inc = (mx - mn) / np
    for i in eachindex(pd)
        x[i] = mn + (i * inc)
        pd[i] = Distributions.pdf(d, x[i])
    end
    p = UnicodePlots.lineplot(x, pd, title = "PDF", xlabel = "x", ylabel = "density")
    return p
end
