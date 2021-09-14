#### resampling ####
C_DFT_N_RESAMPLES = 10000
## IS resampler
"""
    resample(sample; n = 10000 [, discard = 0])

Resample a results type of some kind, e.g. MCMCSample, ImportanceSample or SingleModelResults.

**Parameters**
- `sample`      -- parameter inference results of some kind.
- `n`           -- the index of the model parameter to be plotted on the x axis.
- `discard`     -- number of initial samples to be discarded - applicable to `sample` of type RejectionSamples only.

"""
function resample(sample::ImportanceSample; n::Int64 = C_DFT_N_RESAMPLES)
    rsi = StatsBase.sample(collect(1:length(sample.weight)), StatsBase.Weights(sample.weight), n)
    resamples = zeros(length(sample.mu), n, 1)
    for i in eachindex(rsi)
        resamples[:,i,1] .= sample.theta[:,rsi[i]]
    end
    return resamples
end
## rejection samples
function resample(sample::RejectionSample; n::Int64 = C_DFT_N_RESAMPLES, discard::Int64=0)
    resamples = zeros(length(sample.mu), n, 1)
    ns = size(sample.theta, 2)
    nc = size(sample.theta, 3)
    for i in 1:n
        resamples[:,i,1] .= sample.theta[:,rand((discard+1):ns), rand(1:nc)]
    end
    return resamples
end
## - mcmc
function resample(sample::MCMCSample; n::Int64 = C_DFT_N_RESAMPLES)
    return resample(sample.samples; n=n, discard=sample.adapt_period)
end
# - single model
function resample(sample::SingleModelResults; n::Int64 = C_DFT_N_RESAMPLES)
    return resample(sample.ibis; n=n)
end

#### print samples ####

## move this ********
include("cmn/utils.jl")

"""
    save_to_file

**Parameters**
- `samples`     -- a data structure of type `MCMCSample`, `ImportanceSample` or `ARQMCMCSample`.
- `dpath`       -- the directory where the results will be saved.

Print the results of an inference analysis to file.
"""
## print importance sample results
function save_to_file(results::ImportanceSample, dpath::String)
    # check dir
    isdir(dpath) || mkpath(dpath)
    # print metadata
    open(string(dpath, "metadata.csv"), "w") do f
        write(f, "st,n,run_time,bme\nis,$(length(results.mu)),$(results.run_time),$(results.bme[1])")
    end
    ##
    print_imp_sample(results, dpath)
end

## print arq mcmc results
function save_to_file(results::ARQMCMCSample, dpath::String)
    ARQMCMC.save_to_file(results, dpath)
end

## print MCMC sample
function save_to_file(results::MCMCSample, dpath::String)
    # check dir
    isdir(dpath) || mkpath(dpath)
    # print metadata
    open(string(dpath, "metadata.csv"), "w") do f
        write(f, "st,n,adapt_period,run_time\nmcmc,")
        write(f, "$(length(results.samples.mu)),$(results.adapt_period),$(results.run_time)")
    end
    # print rejection/re samples
    print_rej_sample(results.samples, dpath, results.psrf)
end

# - single model results
function save_to_file(results::SingleModelResults, dpath::String)
    # check dir
    isdir(dpath) || mkpath(dpath)
    # print
    save_to_file(results.ibis, string(dpath, "ibis/"))
    save_to_file(results.mcmc, string(dpath, "mcmc/"))
end
