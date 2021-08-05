#### tabulate stuff ####

## results summary
#- `proposals`   -- display proposal analysis (MCMC only).
"""
    tabulate_results(results)

Display the results of an inference analysis.

The main parameter is `results` -- a data structure of type `MCMCSample`, `ImportanceSample`, `ARQMCMCSample` or `ModelComparisonResults`. When invoking the latter, the named parameter `null_index = 1` by default but can be overridden. This determines the 'null' model, used to compute the Bayes factor.
"""
function tabulate_results(results::MCMCSample)
    ## proposals
    # proposals && tabulate_proposals(results)
    ## samples
    # println("MCMC results:")
    h = ["θ", "E[θ]", ":σ", "PSRF", "PSRF975"]
    d = Matrix(undef, length(results.samples.mu), 5)
    sd = compute_sigma(results.samples.cv)
    d[:,1] .= 1:length(results.samples.mu)
    d[:,2] .= round.(results.samples.mu; sigdigits = C_PR_SIGDIG)
    d[:,3] .= round.(sd; sigdigits = C_PR_SIGDIG)
    d[:,4] .= round.(results.psrf[:,2]; sigdigits = C_PR_SIGDIG + 1)
    d[:,5] .= round.(results.psrf[:,3]; sigdigits = C_PR_SIGDIG + 1)
    PrettyTables.pretty_table(d, h)
end

## importance sample:
function tabulate_results(results::ImportanceSample)
    ## samples
    # println("IBIS results:")
    h = ["θ", "E[θ]", ":σ", C_LBL_BME]
    d = Matrix(undef, length(results.mu), 4)
    sd = compute_sigma(results.cv)
    d[:,1] .= 1:length(results.mu)
    d[:,2] .= round.(results.mu; sigdigits = C_PR_SIGDIG)
    d[:,3] .= round.(sd; sigdigits = C_PR_SIGDIG)
    d[:,4] .= 0
    bme_seq = C_DEBUG ? (1:2) : (1:1)
    d[bme_seq, 4] = round.(results.bme[bme_seq]; digits = 1)
    PrettyTables.pretty_table(d, h)
end

function tabulate_results(results::ARQMCMCSample)
    ARQMCMC.tabulate_results(results)
end

function tabulate_results(results::SingleModelResults)
    println("--- ", results.model.name, " MODEL INFERENCE RESULTS ---")
    println("- IBIS RESULTS:")
    tabulate_results(results.ibis)
    println("- MCMC RESULTS:")
    tabulate_results(results.mcmc)
    ## ADD SAMPLE COMPARISON
end


## IS resampler - artifical RejectionSamples
# function resample_is(sample::ImportanceSample; n = 10000)
#     rsi = StatsBase.sample(collect(1:length(sample.weight)), StatsBase.Weights(sample.weight), n)
#     resamples = zeros(length(sample.mu), n, 1)
#     for i in eachindex(rsi)
#         resamples[:,i,1] .= sample.theta[:,rsi[i]]
#     end
#     return RejectionSample(resamples, sample.mu, sample.cv)
# end
#
# function compute_bayes_factor(ml::Array{Float64,1}, null_index::Int64)
#     output = exp.(-ml)
#     output ./= output[1]
#     return output
# end
#
# ## model evidence comparison
# function tabulate_results(results::ModelComparisonResults; null_index = 1)
#     h = ["Model", string("ln E[p(y)]"), ":σ", "BF"]   # ADD THETA ******************
#     d = Matrix(undef, length(results.mu), length(h))
#     d[:,1] .= results.names
#     d[:,2] .= round.(results.mu; digits = 1)
#     d[:,3] .= round.(results.sigma; sigdigits = C_PR_SIGDIG)
#     d[:,4] .= round.(compute_bayes_factor(results.mu, null_index); digits = 1)
#     PrettyTables.pretty_table(d, h)
# end
