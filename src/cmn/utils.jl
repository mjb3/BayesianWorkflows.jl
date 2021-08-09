#### print samples ####

## print theta summary (internal use)
function print_sample_summary(results, dpath::String)
    # print theta summary
    open(string(dpath, "summary.csv"), "w") do f
        # print headers
        write(f, "theta,mu,sigma")
        # print data
        for p in eachindex(results.mu)
            # c = model.model_name[p]
            write(f, "\n$p,$(results.mu[p]),$(sqrt(results.cv[p,p]))")
        end
    end
end

## print rejection sample (just the samples, summary and sre)
function print_rej_sample(samples::RejectionSample, dpath::String, gelman::Array{Float64,2})
    # print rejection/re samples
    open(string(dpath, "samples.csv"), "w") do f
        # print headers
        write(f, "mc,iter")
        for i in 1:size(samples.theta, 1)
            write(f, ",$i")
        end
        # print data
        for mc in 1:size(samples.theta, 3)
            for i in 1:size(samples.theta, 2)
                write(f, "\n$(mc),$i")
                for p in 1:size(samples.theta, 1)
                    write(f, ",$(samples.theta[p,i,mc])")
                end
            end
        end
    end
    # print theta summary
    print_sample_summary(samples, string(dpath, "rj_"))
    # print gelman results (make optional?)
    open(string(dpath, "gelman.csv"), "w") do f
        # print headers
        write(f, "theta,sre_ll,sre,sre_ul")
        # print data
        for p in eachindex(samples.mu)
            write(f, "\n$p,$(gelman[p,1]),$(gelman[p,2]),$(gelman[p,3])")
        end
    end
end

## print importance sample (just the weighed sample and summary)
function print_imp_sample(results::ImportanceSample, dpath::String)
    # check dir
    isdir(dpath) || mkpath(dpath)
    # print importance samples
    open(string(dpath, "theta.csv"), "w") do f
        # print headers
        write(f, "1")
        for i in 2:length(results.mu)
            write(f, ",$i")
        end
        # print data
        for i in 1:size(results.theta, 2)
            write(f, "\n$(results.theta[1,i])")
            for p in 2:length(results.mu)
                write(f, ",$(results.theta[p,i])")
            end
        end
    end
    # print weights
    open(string(dpath, "weight.csv"), "w") do f
        # print headers
        write(f, "i,w")
        for i in eachindex(results.weight)
            write(f, "\n$i,$(results.weight[i])")
        end
    end
    # print theta summary
    print_sample_summary(results, string(dpath, "is_"))
end
