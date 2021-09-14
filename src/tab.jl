#### tabulate stuff ####


## results summary
#- `proposals`   -- display proposal analysis (MCMC only).
"""
    tabulate_results(results; [null_index = 1], [display=true])

Display the results of an inference analysis.

The main parameter is `results` -- a data structure of type `MCMCSample`, `ImportanceSample`, `ARQMCMCSample` or `ModelComparisonResults`. When invoking the latter, the named parameter `null_index = 1` by default but can be overridden. This determines the 'null' model, used to compute the Bayes factor. The `display` flag can be set `=false` to return a single DataFrame for valid types.
"""
function tabulate_results(results::MCMCSample; display=true)
    ## proposals
    # proposals && tabulate_proposals(results)
    ## samples
    d = Matrix(undef, length(results.samples.mu), 5)
    sd = compute_sigma(results.samples.cv)
    d[:,1] .= 1:length(results.samples.mu)
    d[:,2] .= prettify_n.(results.samples.mu)
    d[:,3] .= round.(sd; sigdigits = C_PR_SIGDIG)
    d[:,4] .= round.(results.psrf[:,2]; sigdigits = C_PR_SIGDIG + 1)
    d[:,5] .= round.(results.psrf[:,3]; sigdigits = C_PR_SIGDIG + 1)
    if display
        h = ["θ", "E[θ]", ":σ", "R̂", "R̂97.5"]
        PrettyTables.pretty_table(d, h)
    else
        h = ["x", "e_x", "sd_x", "R̂", "R̂97.5"]
        return DataFrames.DataFrame(d, h)
    end
end

## importance sample:
function tabulate_results(results::ImportanceSample; display=true)
    ## samples
    # println("IBIS results:")
    h = ["θ", "E[θ]", ":σ", C_LBL_BME]
    d = Matrix(undef, length(results.mu), 4)
    sd = compute_sigma(results.cv)
    d[:,1] .= 1:length(results.mu)
    d[:,2] .= prettify_n.(results.mu)
    d[:,3] .= round.(sd; sigdigits = C_PR_SIGDIG)
    d[:,4] .= 0
    bme_seq = C_DEBUG ? (1:2) : (1:1)
    d[bme_seq, 4] = round.(results.bme[bme_seq]; digits = 1)
    if display
        h = ["θ", "E[θ]", ":σ", C_LBL_BME]
        PrettyTables.pretty_table(d, h)
    else
        h = ["x", "e_x", "sd_x", C_LBL_BME]
        return DataFrames.DataFrame(d, h)
    end
end

function tabulate_results(results::ARQMCMCSample; display=true)
    ARQMCMC.tabulate_results(results; display=display)
end

function tabulate_results(results::SingleModelResults)
    println("--- ", results.model.name, " MODEL INFERENCE RESULTS ---")
    println("- IBIS RESULTS:")
    tabulate_results(results.ibis)
    println("- MCMC RESULTS:")
    tabulate_results(results.mcmc)
    ## TBA: SAMPLE COMPARISON
end

# - TBA: include parameter estimates? **
function tabulate_results(results::Array{SingleModelResults, 1}; null_index = 1)
    ## model evidence comparison:
    function compute_bayes_factor(ml::Array{Float64,1}, null_index::Int64)
        output = exp.(-ml)
        output ./= output[1]
        return output
    end
    ## results table:
    h = ["Model", C_LBL_BME, "BF"]
    d = Matrix(undef, length(results), length(h))
    d[:,1] .= [r.model.name for r in results]
    ml = [r.ibis.bme[1] for r in results]
    d[:,2] .= round.(ml; digits = 1)
    d[:,3] .= round.(compute_bayes_factor(ml, null_index); digits = 1)
    PrettyTables.pretty_table(d, h)
end
