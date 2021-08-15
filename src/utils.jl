#### other printing ####

## save simulation results to file REPL BY save_to_file
# NEED TO ADJUST FOR 2D POPULATIONS *************
# function print_sim_results(results::SimResults, dpath::String)
#     #model::HiddenMarkovModel,
#     # check dir
#     isdir(dpath) || mkpath(dpath)
#     # print sequence
#     open(string(dpath, "sim.csv"), "w") do f
#         # print headers
#         write(f, "time, event")
#         for p in 1:size(results.population, 2)
#             write(f, ",$p")
#         end
#         # print event sequence
#         for i in eachindex(results.particle.trajectory)
#             tm = results.particle.trajectory[i].time
#             tp = results.particle.trajectory[i].event_type
#             write(f, "\n$tm,$tp")
#             for p in 1:size(results.population, 2)
#                 write(f, ",$(results.population[i,p])")
#             end
#         end
#     end # end of print sequence
#     # print observation
#     open(string(dpath, "obs.csv"), "w") do f
#         # print headers
#         write(f, "time,id")
#         for p in eachindex(results.observations[1].val)
#             # c = model.model_name[p]
#             write(f, ",$p")
#         end
#         # print event sequence
#         for i in eachindex(results.observations)
#             write(f, "\n$(results.observations[i].time),$(results.observations[i].obs_id)")
#             tp = results.observations[i].val
#             for p in eachindex(tp)
#                 write(f, ",$(tp[p])")
#             end
#         end
#     end # end of print observations
# end

#### print samples ####

## move this ********
include("cmn/utils.jl")

"""
    print_results

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
