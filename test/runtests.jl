# dummy test result
using BayesianWorkflows
import Random
import Distributions
import Test

Random.seed!(1)

## tests
test_stat(xx, yy) = round(xx; sigdigits=4)==yy
Test.@testset "package_test" begin
    Test.@test true
    # theta = [0.003, 0.1]
    # data_fp = "data/pooley.csv"
    #
    # ## getting started
    # y = get_observations(data_fp) # val_seq=2:3
    # model = generate_model("SIS", [100,1])
    # Test.@test true
    #
    # ## simulation # NB. first define the SIS 'model' variable, per above
    # Test.@testset "simulation" begin
    #     x = gillespie_sim(model, theta)	    # run simulation
    #     println(plot_trajectory(x))			# plot (optional)
    #     Test.@test x.population[end][1]==45
    # end
    #
    # ## ARQMCMC
    # model.prior = Distributions.Product(Distributions.Uniform.(zeros(2), [0.01, 0.5]))
    # Test.@testset "arqmcmc" begin
    #     sample_interval = [0.0005, 0.02]
    #     rs = run_arq_mcmc_analysis(model, y, sample_interval)
    #     tabulate_results(rs)
    #     # println("ARQ: ", round(rs.imp_sample.mu[1]; sigdigits=4))
    #     Test.@test test_stat(rs.imp_sample.mu[1], 0.003217)
    #     println(plot_parameter_trace(rs, 1))
    # end
    #
    # ## DA MCMC
    # Test.@testset "mbpmcmc" begin
    #     rs = run_mcmc_analysis(model, y)
    #     tabulate_results(rs)
    #     Test.@test test_stat(rs.samples.mu[1], 0.003318)
    # end
    # # println(plot_parameter_trace(rs, 1))  # trace plot of contact parameter (optional)
    #
    # ## SMC^2
    # Test.@testset "smc2" begin
    #     results = run_ibis_analysis(model, y)
    #     tabulate_results(results)
    #     Test.@test test_stat(results.bme[1], 19.98)
    # end
    #
    # ## MBP IBIS
    # Test.@testset "mbpibis" begin
    #     results = run_ibis_analysis(model, y; algorithm="MBPI")
    #     tabulate_results(results)
    #     println(results.bme[1])
    # end
    #
    # ## model comparison
    # # define model to compare against
    # # seis_model = generate_model("SEIS", [100,0,1])
    # # seis_model.prior = Distributions.Product(Distributions.Uniform.(zeros(3), [0.1,0.5,0.5]))
    # # seis_model.obs_model = partial_gaussian_obs_model(2.0, seq = 3, y_seq = 2)
    # #
    # # # run comparison
    # # models = [model, seis_model]
    # # mcomp = run_model_comparison_analysis(models, y)
    # # tabulate_results(mcomp; null_index = 1)
    # # println(plot_model_comparison(mcomp))
    #
    # ## custom models
    # Test.@testset "custom_sim" begin
    #     # rate function
    #     function sis_rf!(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
    #         output[1] = parameters[1] * population[1] * population[2]
    #         output[2] = parameters[2] * population[2]
    #     end
    #     # define obs function
    #     function obs_fn(y::Observation, population::Array{Int64, 1}, theta::Array{Float64,1})
    #         y.val .= population
    #     end
    #     # prior
    #     prior = Distributions.Product(Distributions.Uniform.(zeros(2), [0.1, 0.5]))
    #     # obs model
    #     function si_gaussian(y::Observation, population::Array{Int64, 1}, theta::Array{Float64,1})
    #         obs_err = 2
    #         tmp1 = log(1 / (sqrt(2 * pi) * obs_err))
    #         tmp2 = 2 * obs_err * obs_err
    #         obs_diff = y.val[2] - population[2]
    #         return tmp1 - ((obs_diff * obs_diff) / tmp2)
    #     end
    #     tm = [-1 1; 1 -1] # transition matrix
    #     # define model
    #     model = DPOMPModel("SIS", sis_rf!, [100, 1], tm, obs_fn, si_gaussian, prior, 0)
    #     x = gillespie_sim(model, theta)	# run simulation and plot
    #     # println(x.population[end])
    #     Test.@test x.population[end][1]==45
    # end

end # end of test set
