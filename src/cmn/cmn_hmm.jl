#### common HMM stuff (i.e. not needed for arq)

struct ObservationQuantiles
    t::Vector{Float64}
    y::Array{Float64,2}
    q::Vector{Float64}
end

## compute observation quantiles
function get_observation_quantiles(observations::Vector{Vector{Observation}}, val_index::Int64, quantiles::Vector{Float64})
    t1 = [y.time for y in observations[1]]
    output = zeros(length(observations[1]), length(quantiles))
    for i in eachindex(observations[1])
        y = [yy[i].val[val_index] for yy in observations]
        output[i, :] .= Statistics.quantile!(y, quantiles)
    end
    return ObservationQuantiles(t1, output, quantiles)
end

## choose event type
function choose_event(cum_rates::Array{Float64,1})
    etc = rand() * cum_rates[end]
    for i in 1:(length(cum_rates) - 1)
        cum_rates[i] > etc && return i
    end
    return length(cum_rates)
end

## Gaussian mv parameter proposal
function get_mv_param(propd::Distributions.MvNormal, sclr::Float64, theta_i::Array{Float64, 1})
    output = rand(propd)
    output .*= sclr
    output .+= theta_i
    return output
end

## mcmc proposal density
function get_prop_density(cv::Array{Float64,2}, old)
    ## update proposal density
    tmp = LinearAlgebra.Hermitian(cv)
    if LinearAlgebra.isposdef(tmp)
        return Distributions.MvNormal(Matrix(tmp))
    else
        C_DEBUG && println(" warning: particle degeneracy problem\n  covariance matrix: ", cv)
        return old
    end
end

## get normalised observation date values
function get_norm_yt(res, col = 1)
    dts = zeros(length(res[!,col]))
    for i in eachindex(dts)
        dts[i] = Dates.value(Dates.DateTime(res[i,col], "yyyy-mm-dd"))# / C_MS_DAY
    end
    dts .-= dts[1]
    return dts
end

## get observations data from DataFrame or file location
"""
    get_observations(source)

Return an array of type `Observation`, based on a two-dimensional array, `DataFrame` or file location (i.e. `String`.)

Note that a observation times must be in the first column of the input variable.
"""
function get_observations(df::DataFrames.DataFrame; time_col=1, type_col=0, val_seq=nothing)
    val_seq = isnothing(val_seq) ? (2:size(df,2)) : val_seq
    ## check dates
    if eltype(df[:,time_col]) == Dates.Date
        dts = Dates.value.(df[:,time_col])
    else
        dts = df[:,time_col]
    end
    ## populate output
    obs = Observation[]
    for i in 1:size(df,1)
        obs_type::Int64 = type_col==0 ? 1 : df[i,type_col]
        v = zeros(Int64, length(val_seq))
        v .= values(df[i, val_seq])
        push!(obs, Observation(dts[i], obs_type, 1.0, v))
    end
    sort!(obs)
    return obs
end
function get_observations(fpath::String; time_col=1, type_col=0, val_seq=nothing)
    df = CSV.read(fpath, DataFrames.DataFrame)
    return get_observations(df; time_col=time_col, type_col=type_col, val_seq=val_seq)
end

## save simulation results to file
# NB. function overloaded per docs below
function save_to_file(results::SimResults, dpath::String)
    # check dir
    isdir(dpath) || mkpath(dpath)
    # print sequence
    open(string(dpath, "sim.csv"), "w") do f
        # print headers
        write(f, "time, event")
        for p in 1:size(results.population, 2)
            write(f, ",val$p")
        end
        # print event sequence
        for i in eachindex(results.particle.trajectory)
            tm = results.particle.trajectory[i].time
            tp = results.particle.trajectory[i].event_type
            write(f, "\n$tm,$tp")
            for p in 1:size(results.population, 2)
                write(f, ",$(results.population[i,p])")
            end
        end
    end # end of print sequence
    # print observation
    open(string(dpath, "obs.csv"), "w") do f
        # print headers
        write(f, "time,id,prop")
        for p in eachindex(results.observations[1].val)
            # c = model.name[p]
            write(f, ",val$p")
        end
        # print event sequence
        for i in eachindex(results.observations)
            write(f, "\n$(results.observations[i].time),$(results.observations[i].obs_id),$(results.observations[i].prop)")
            tp = results.observations[i].val
            for p in eachindex(tp)
                write(f, ",$(tp[p])")
            end
        end
    end # end of print observations
end

## y quantiles
function save_to_file(x::ObservationQuantiles, fpath::String)
    open(fpath, "w") do f
        # print headers
        write(f, "t")
        for q in x.q
            write(f, ",$q")
        end
        # print quantiles
        for i in eachindex(x.t)
            write(f, "\n$(x.t[i])")
            for j in eachindex(x.q)
                write(f, ",$(x.y[i,j])")
            end
        end
    end
end

## save n simulation results to file
function save_to_file(results::Vector{SimResults}, dpath::String; obs_quantiles::Vector{Float64}=[0.25, 0.5, 0.75])
    dp = string.(dpath, 1:length(results), "/")
    println("SAVING TO ", dp)
    # check dir
    isdir(dpath) || mkpath(dpath)
    # print metadata
    open(string(dpath, "metadata.csv"), "w") do f
        write(f, "n\n$(length(results))")
    end
    # quantiles
    # NB. expand for all observed val ****
    y = [x.observations for x in results]
    y1 = get_observation_quantiles(y, 1, obs_quantiles)
    save_to_file(y1, string(dp, "y1.csv"))
    # sims
    save_to_file.(results, dp)
end
