var documenterSearchIndex = {"docs":
[{"location":"models/#DPOMP-models","page":"DPOMP models","title":"DPOMP models","text":"","category":"section"},{"location":"models/","page":"DPOMP models","title":"DPOMP models","text":"Discrete-state space Partially Observed Markov Processes","category":"page"},{"location":"models/#Predefined-models","page":"DPOMP models","title":"Predefined models","text":"","category":"section"},{"location":"models/","page":"DPOMP models","title":"DPOMP models","text":"The package includes the following (mostly) epidemiological models as predefined examples:","category":"page"},{"location":"models/","page":"DPOMP models","title":"DPOMP models","text":"\"SI\"\n\"SIR\"\n\"SIS\"\n\"SEI\"\n\"SEIR\"\n\"SEIS\"\n\"SEIRS\"\n\"PREDPREY\"\n\"ROSSMAC\"","category":"page"},{"location":"models/","page":"DPOMP models","title":"DPOMP models","text":"They can be instantiated using the generate_model function:","category":"page"},{"location":"models/","page":"DPOMP models","title":"DPOMP models","text":"using BayesianWorkflows\ninitial_model_state = [100, 1]\nmodel = generate_model(\"SIS\", initial_model_state)","category":"page"},{"location":"models/","page":"DPOMP models","title":"DPOMP models","text":"NB. this generates a standard configuration of the named model that can be tweaked later.","category":"page"},{"location":"models/#Options","page":"DPOMP models","title":"Options","text":"","category":"section"},{"location":"models/","page":"DPOMP models","title":"DPOMP models","text":"Aside from the model name and initial model state (i.e. the initial 'population',) there are a number of options for generating a default model configuration:","category":"page"},{"location":"models/","page":"DPOMP models","title":"DPOMP models","text":"freq_dep    – epidemiological models only, set to true for frequency-dependent contact rates.\nobs_error   – average observation error (default = 2.)\nt0_index    – index of the parameter that represents the initial time. 0 if fixed at 0.0.","category":"page"},{"location":"models/#Full-model-configuration","page":"DPOMP models","title":"Full model configuration","text":"","category":"section"},{"location":"models/","page":"DPOMP models","title":"DPOMP models","text":"DPOMPModel","category":"page"},{"location":"models/#BayesianWorkflows.DPOMPModel","page":"DPOMP models","title":"BayesianWorkflows.DPOMPModel","text":"DPOMPModel\n\nA mutable struct which represents a DSSCT model (see Models for further details).\n\nFields\n\nname                – string, e,g, \"SIR\".\nn_events            – number of distinct event types.\nrate_function       – event rate function.\nfn_initial_state    – function for sampling initial model state.\ninitial_condition   – initial condition.\nm_transition        – transition matrix.\nobs_model           – observation model likelihood function.\nobs_function        – observation function, use this to add 'noise' to simulated observations.\nt0_index            – index of the parameter that represents the initial time. 0 if fixed at 0.0.\n\n\n\n\n\n","category":"type"},{"location":"manual/#Package-manual","page":"Package manual","title":"Package manual","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"Pages = [\"manual.md\"]\nDepth = 3","category":"page"},{"location":"manual/#Types","page":"Package manual","title":"Types","text":"","category":"section"},{"location":"manual/#Model-types","page":"Package manual","title":"Model types","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"Particle\nEvent\nObservation","category":"page"},{"location":"manual/#BayesianWorkflows.Particle","page":"Package manual","title":"BayesianWorkflows.Particle","text":"Particle\n\nE.g. the main results of a simulation including the initial and final conditions, but not the full state trajectory.\n\nFields\n\ntheta               – e.g. simulation parameters.\ninitial_condition   – initial system state.\nfinal_condition     – final system state.\ntrajectory          – the event history.\nlog_like            – trajectory log likelihood, mainly for internal use.\n\n\n\n\n\n","category":"type"},{"location":"manual/#BayesianWorkflows.Event","page":"Package manual","title":"BayesianWorkflows.Event","text":"Event\n\nRequires no explanation.\n\nFields\n\ntime        – the time of the event.\nevent_type  – indexes the rate function and transition matrix.\n\n\n\n\n\n","category":"type"},{"location":"manual/#BayesianWorkflows.Observation","page":"Package manual","title":"BayesianWorkflows.Observation","text":"Observation\n\nA single observation. Note that by default val has the same size as the model state space. However that is not necessary - it need only be compatible with the observation model.\n\nFields\n\ntime        – similar to Event.time, the time of the observation.\nobs_id      – <1 if not a resampling step.\nprop        – optional information for the observation model.\nval         – the observation value.\n\n\n\n\n\n","category":"type"},{"location":"manual/#Results","page":"Package manual","title":"Results","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"SimResults\nImportanceSample\nRejectionSample\nMCMCSample\nARQMCMCSample","category":"page"},{"location":"manual/#BayesianWorkflows.SimResults","page":"Package manual","title":"BayesianWorkflows.SimResults","text":"SimResults\n\nThe results of a simulation, including the full state trajectory.\n\nFields\n\nmodel_name      – string, e,g, \"SIR\".\nparticle        – the 'trajectory' variable, of type Particle.\npopulation      – records the complete system state over time.\nobservations    – simulated observations data (an Array of Observation types.)\n\n\n\n\n\n","category":"type"},{"location":"manual/#BayesianWorkflows.ImportanceSample","page":"Package manual","title":"BayesianWorkflows.ImportanceSample","text":"ImportanceSample\n\nThe results of an importance sampling analysis, such as iterative batch importance sampling algorithms.\n\nFields\n\nmu              – weighted sample mean.\ncv              – weighted covariance.\ntheta           – two dimensional array of samples, e.g. parameter; iteration.\nweight          – sample weights.\nrun_time        – application run time.\nbme             – Estimate (or approximation) of the Bayesian model evidence.\n\n\n\n\n\n","category":"type"},{"location":"manual/#BayesianWorkflows.RejectionSample","page":"Package manual","title":"BayesianWorkflows.RejectionSample","text":"RejectionSample\n\nEssentially, the main results of an MCMC analysis, consisting of samples, mean, and covariance matrix.\n\nFields\n\nsamples         – three dimensional array of samples, e.g. parameter; iteration; Markov chain.\nmu              – sample mean.\ncv              – sample covariance matrix.\n\n\n\n\n\n","category":"type"},{"location":"manual/#BayesianWorkflows.MCMCSample","page":"Package manual","title":"BayesianWorkflows.MCMCSample","text":"MCMCSample\n\nThe results of an MCMC analysis, mainly consisting of a RejectionSample.\n\nFields\n\nsamples         – samples of type RejectionSample.\nadapt_period    – adaptation (i.e. 'burn in') period.\nsre             – scale reduction factor estimate, i.e. Gelman diagnostic.\nrun_time        – application run time.\n\n\n\n\n\n","category":"type"},{"location":"manual/#BayesianWorkflows.ARQMCMC.ARQMCMCSample","page":"Package manual","title":"BayesianWorkflows.ARQMCMC.ARQMCMCSample","text":"ARQMCMCSample\n\nThe results of an ARQ MCMC analysis including the ImportanceSample and resampled RejectionSample.\n\nThe sre scale factor reduction estimates relate the rejection (re)samples to the underlying importance sample.\n\nFields\n\nimp_sample          – main results, i.e. ImportanceSample.\nsamples             – resamples, of type RejectionSample.\nadapt_period        – adaptation (i.e. 'burn in') period.\nsample_dispersal    – number of distinct [possible] sample values along each dimension in the unit cube.\nsample_limit        – maximum number of samples per theta tupple.\ngrid_range          – bounds of the parameter space.\nsre                 – scale reduction factor estimate, i.e. Gelman diagnostic. NB. only valid for resamples.\nrun_time            – application run time.\nsample_cache        – a link to the underlying likelihood cache - can be reused.\n\n\n\n\n\n","category":"type"},{"location":"manual/#Functions","page":"Package manual","title":"Functions","text":"","category":"section"},{"location":"manual/#Model-helpers","page":"Package manual","title":"Model helpers","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"generate_model\npartial_gaussian_obs_model","category":"page"},{"location":"manual/#BayesianWorkflows.generate_model","page":"Package manual","title":"BayesianWorkflows.generate_model","text":"generate_model(model_name, initial_condition; freq_dep = false, obs_error = 2.0)\n\nGenerates an DPOMPModel instance. Observation models are generated using the partial_gaussian_obs_model function, with `σ = obs_error (see that functions entry for further details.)\n\nParameters\n\nmodel_name          – the model, e.g. \"SI\"; \"SIR\"; \"SEIR\"; etc\ninitial_condition   – initial condition.\n\nNamed parameters\n\nfreq_dep            – epidemiological models only, set to true for frequency-dependent contact rates.\nobs_error           – average observation error (default = 2.)\nt0_index            – index of the parameter that represents the initial time. 0 if fixed at 0.0.\n\nmodel_name options\n\n\"SI\"\n\"SIR\"\n\"SIS\"\n\"SEI\"\n\"SEIR\"\n\"SEIS\"\n\"SEIRS\"\n\"PREDPREY\"\n\"ROSSMAC\"\n\nExamples\n\ngenerate_model(\"SIS\", [100,1])\n\n\n\n\n\n","category":"function"},{"location":"manual/#BayesianWorkflows.partial_gaussian_obs_model","page":"Package manual","title":"BayesianWorkflows.partial_gaussian_obs_model","text":"partial_gaussian_obs_model(σ = 2.0; seq = 2, y_seq = seq)\n\nGenerate a simple Gaussian observation model. So-called because the accuracy of observations is 'known' and [assumed to be] normally distributed according to~N(0, σ), where observation error σ can be specified by the user.\n\nParameters\n\nσ       – observation error.\nseq     – the indexing sequence of the observed state, e.g. 2 for that state only, 3:4 for the third and fourth, etc.\ny_seq   – as above, the corresponding [indexing] values for the observations data, seq unless otherwise specified.\n\ntest latex eqn:\n\n\fracnk(n - k) = inomnk\n\nExamples\n\np = partial_gaussian_obs_model(1.0, seq = 2)\n\n\n\n\n\n","category":"function"},{"location":"manual/#Simulation","page":"Package manual","title":"Simulation","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"gillespie_sim","category":"page"},{"location":"manual/#BayesianWorkflows.gillespie_sim","page":"Package manual","title":"BayesianWorkflows.gillespie_sim","text":"gillespie_sim(model, parameters; tmax = 100.0, num_obs = 5)\n\nRun a Doob-Gillespie (DGA) simulation based on model.\n\nReturns a SimResults type containing the trajectory and observations data, or an array of the same if n_sims > 1.\n\nParameters\n\nmodel       – DPOMPModel (see [DCTMPs.jl models]@ref).\nparameters  – model parameters.\n\nOptional\n\ntmax        – maximum time (default: 100.)\nn_obs       – the number of observations to draw (default: 5.)\nn_sims      – number of simulations to draw (default: 1.)\n\nExample\n\nusing DiscretePOMP\nm = generate_model(\"SIR\", [50, 1, 0])\nx = DiscretePOMP.gillespie_sim(model, [0.005, 0.12])\nprintln(DiscretePOMP.plot_trajectory(x))\n\n\n\n\n\n","category":"function"},{"location":"manual/#Bayesian-inference","page":"Package manual","title":"Bayesian inference","text":"","category":"section"},{"location":"manual/#Workflows","page":"Package manual","title":"Workflows","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"run_inference_workflow","category":"page"},{"location":"manual/#Algorithms","page":"Package manual","title":"Algorithms","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"run_smc2_analysis\nrun_mbp_ibis_analysis\nrun_mcmc_analysis\nrun_arq_mcmc_analysis","category":"page"},{"location":"manual/#BayesianWorkflows.run_smc2_analysis","page":"Package manual","title":"BayesianWorkflows.run_smc2_analysis","text":"run_smc2_analysis(model, obs_data; ... )\n\nRun an SMC^2 (i.e. particle filter IBIS) analysis based on model and obs_data of type Observations.\n\nParameters\n\nmodel               – DPOMPModel (see [DCTMPs.jl models]@ref).\nobs_data            – Observations data.\n\nOptional\n\nnp                  – number of [outer, i.e. theta] particles (default = 2000.)\nnpf                 – number of [inner] particles (default = 200.)\ness_rs_crit         – resampling criteria (default = 0.5.)\nind_prop            – true for independent theta proposals (default = false.)\nalpha               – user-defined, increase for lower acceptance rate targeted (default = 1.002.)\n\nExample\n\n# NB. using 'y' and 'model' as above\nresults = run_smc2_analysis(model, y)   # importance sample\ntabulate_results(results)               # show the results\n\n\n\n\n\n","category":"function"},{"location":"manual/#BayesianWorkflows.run_mbp_ibis_analysis","page":"Package manual","title":"BayesianWorkflows.run_mbp_ibis_analysis","text":"run_mbp_ibis_analysis(model, obs_data; ... )\n\nRun an MBP-IBIS analysis based on model, and obs_data of type Observations.\n\nParameters\n\nmodel               – DPOMPModel (see [DCTMPs.jl models]@ref).\nobs_data            – Observations data.\n\nOptional\n\nnp                  – number of particles (default = 4000.)\ness_rs_crit         – resampling criteria (default = 0.5.)\nn_props             – MBP mutations per step (default = 3.)\nind_prop            – true for independent theta proposals (default = false.)\nalpha               – user-defined, increase for lower acceptance rate targeted (default = 1.002.)\n\nExample\n\n# NB. using 'y' and 'model' as above\nresults = run_mbp_ibis_analysis(model, y)# importance sample\ntabulate_results(results)                # show the results\n\n\n\n\n\n","category":"function"},{"location":"manual/#BayesianWorkflows.run_mcmc_analysis","page":"Package manual","title":"BayesianWorkflows.run_mcmc_analysis","text":"run_mcmc_analysis(model, obs_data; ... )\n\nRun an n_chains-MCMC analysis using the designated algorithm (MBP-MCMC by default.)\n\nThe initial_parameters are sampled from the prior distribution unless otherwise specified by the user. A Gelman-Rubin convergence diagnostic is automatically carried out (for n_chains > 1) and included in the [multi-chain] analysis results.\n\nParameters\n\nmodel               – DPOMPModel (see [DCTMPs.jl models]@ref).\nobs_data            – Observations data.\n\nOptional\n\nn_chains            – number of Markov chains (default: 3.)\ninitial_parameters  – 2d array of initial model parameters. Each column vector correspondes to a single model parameter.\nsteps               – number of iterations.\nadapt_period        – number of discarded samples.\nmbp                 – model based proposals (MBP). Set mbp = false for standard proposals.\nppp                 – the proportion of parameter (vs. trajectory) proposals in Gibbs sampler. Default: 30%. NB. not required for MBP.\nfin_adapt           – finite adaptive algorithm. The default is false, i.e. [fully] adaptive.\nmvp                 – increase for a higher proportion of 'move' proposals. NB. not applicable if MBP = true (default: 2.)\n\nExample\n\ny = x.observations                          # some simulated data\nmodel = generate_model(\"SIR\", [50, 1, 0])   # a model\nresults = run_mcmc_analysis(model, y; fin_adapt = true) # finite-adaptive MCMC\ntabulate_results(results)                   # optionally, show the results\n\n\n\n\n\n","category":"function"},{"location":"manual/#BayesianWorkflows.run_arq_mcmc_analysis","page":"Package manual","title":"BayesianWorkflows.run_arq_mcmc_analysis","text":"run_arq_mcmc_analysis(model, obs_data, theta_range; ... )\n\nRun ARQ-MCMC analysis with n_chains Markov chains.\n\nThe Gelman-Rubin convergence diagnostic is computed automatically.\n\nParameters\n\nmodel               – DPOMPModel (see docs.)\nobs_data            – Observations data.\nsample_interval     – An array specifying the (fixed or fuzzy) interval between samples.\n\nOptional\n\nsample_dispersal   – i.e. the length of each dimension in the importance sample.\nsample_limit        – sample limit, should be increased when the variance of model.pdf is high (default: 1.)\nn_chains            – number of Markov chains (default: 3.)\nsteps               – number of iterations.\nburnin              – number of discarded samples.\ntgt_ar              – acceptance rate (default: 0.33.)\nnp                  – number of SMC particles in PF (default: 200.)\ness_crit            – acceptance rate (default: 0.33.)\nsample_cache        – the underlying model likelihood cache - can be retained and reused for future analyses.\n\n\n\n\n\n","category":"function"},{"location":"manual/#Bayesian-model-analysis","page":"Package manual","title":"Bayesian model analysis","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"run_model_comparison_analysis","category":"page"},{"location":"manual/#Utilities","page":"Package manual","title":"Utilities","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"get_observations\ntabulate_results\nsave_to_file","category":"page"},{"location":"manual/#BayesianWorkflows.get_observations","page":"Package manual","title":"BayesianWorkflows.get_observations","text":"get_observations(source)\n\nReturn an array of type Observation, based on a two-dimensional array, DataFrame or file location (i.e. String.)\n\nNote that a observation times must be in the first column of the input variable.\n\n\n\n\n\n","category":"function"},{"location":"manual/#BayesianWorkflows.tabulate_results","page":"Package manual","title":"BayesianWorkflows.tabulate_results","text":"tabulate_results(results)\n\nDisplay the results of an inference analysis.\n\nThe main parameter is results – a data structure of type MCMCSample, ImportanceSample, ARQMCMCSample or ModelComparisonResults. When invoking the latter, the named parameter null_index = 1 by default but can be overridden. This determines the 'null' model, used to compute the Bayes factor.\n\n\n\n\n\n","category":"function"},{"location":"manual/#Visualisation","page":"Package manual","title":"Visualisation","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"plot_trajectory\nplot_parameter_trace\nplot_parameter_marginal\nplot_parameter_heatmap\nplot_model_comparison","category":"page"},{"location":"manual/#BayesianWorkflows.plot_trajectory","page":"Package manual","title":"BayesianWorkflows.plot_trajectory","text":"plot_trajectory(x; plot_index=[:])\n\nPlot the trajectory of a a DGA simulation using UnicodePlots.jl.\n\nThe only input parameter required is x of type SimResults, i.e. from a call to gillespie_sim. All system states are plotted by default, but a subset can be specified by passing an integer array to the plot_index option, which contains the indices of the desired subset. E.g. [1,2] for the first two 'compartments' only.\n\n\n\n\n\n","category":"function"},{"location":"manual/#BayesianWorkflows.plot_parameter_trace","page":"Package manual","title":"BayesianWorkflows.plot_parameter_trace","text":"plot_parameter_trace(mcmc, [parameter::Int64])\n\nProduce a trace plot of samples using UnicodePlots.jl.\n\nThe mcmc input is of type MCMCSample, ARQMCMCSample or RejectionSample. The parameter index can be optionally specified, else all parameters are plotted and returned as an Array of unicode plots.\n\n\n\n\n\n","category":"function"},{"location":"manual/#BayesianWorkflows.plot_parameter_marginal","page":"Package manual","title":"BayesianWorkflows.plot_parameter_marginal","text":"plot_parameter_marginal(sample, parameter)\n\nPlot the marginal distribution of samples from an MCMC analysis for a given model parameter using UnicodePlots.jl.\n\nParameters\n\nresults     – Results object, e.g. of type MCMCSample.\nparameter   – the index of the model parameter to be plotted.\nadapt_period– Adaptation period to be discarded, only required for RejectionSample.\n\nOptional\n\nuse_is      – Resample IS rather than using MCMC [re]samples (ARQMCMCSample results only.)\n\n\n\n\n\n","category":"function"},{"location":"manual/#BayesianWorkflows.plot_parameter_heatmap","page":"Package manual","title":"BayesianWorkflows.plot_parameter_heatmap","text":"plot_parameter_heatmap(mcmc, x_parameter, y_parameter)\n\nPlot the marginal distribution of samples from an MCMC analysis for two model parameters using UnicodePlots.jl.\n\nParameters\n\nmcmc        – MCMCResults, e.g. from a call to run_met_hastings_mcmc.\nx_parameter   – the index of the model parameter to be plotted on the x axis.\ny_parameter   – the index of the model parameter to be plotted on the y axis.\n\n\n\n\n\n","category":"function"},{"location":"manual/#BayesianWorkflows.plot_model_comparison","page":"Package manual","title":"BayesianWorkflows.plot_model_comparison","text":"plot_model_comparison(results)\n\nPlot the Bayesian model evidence (BME) from a [multi-] model inference workflow, using UnicodePlots.jl.\n\nParameters\n\nresults   – ModelComparisonResults, i.e. from a call to run_model_comparison_analysis.\n\n\n\n\n\n","category":"function"},{"location":"manual/#Index","page":"Package manual","title":"Index","text":"","category":"section"},{"location":"manual/","page":"Package manual","title":"Package manual","text":"","category":"page"},{"location":"workflows/#Bayesian-workflows","page":"Bayesian workflows","title":"Bayesian workflows","text":"","category":"section"},{"location":"workflows/","page":"Bayesian workflows","title":"Bayesian workflows","text":"This page gives a brief overview of the simulation and inference features of the package.","category":"page"},{"location":"workflows/#Model-simulation","page":"Bayesian workflows","title":"Model simulation","text":"","category":"section"},{"location":"workflows/","page":"Bayesian workflows","title":"Bayesian workflows","text":"DPOMP models can be simulated using the Gillespie algorithm, which is invoked as follows:","category":"page"},{"location":"workflows/","page":"Bayesian workflows","title":"Bayesian workflows","text":"using BayesianWorkflows\ninitial_condition = [100, 1]    # define a model\nmodel = generate_model(\"SIS\", initial_condition)\n\ntheta = [0.003, 0.1]            # model parameters\nx = gillespie_sim(model, theta)\t# run simulation\nprintln(plot_trajectory(x))     # plot (optional)","category":"page"},{"location":"workflows/","page":"Bayesian workflows","title":"Bayesian workflows","text":"Simulations also provide a vector of simulated observations (x.observations) that can be used to try out the inference features of the package.","category":"page"},{"location":"workflows/","page":"Bayesian workflows","title":"Bayesian workflows","text":"gillespie_sim","category":"page"},{"location":"workflows/#Inference-workflows","page":"Bayesian workflows","title":"Inference workflows","text":"","category":"section"},{"location":"workflows/","page":"Bayesian workflows","title":"Bayesian workflows","text":"The package provides workflows for both single-model [i.e. paramerter-] inference and multi-model inference (or 'model comparison'.)","category":"page"},{"location":"workflows/#[Single-model]-parameter-inference","page":"Bayesian workflows","title":"[Single-model] parameter inference","text":"","category":"section"},{"location":"workflows/#Model-comparison","page":"Bayesian workflows","title":"Model comparison","text":"","category":"section"},{"location":"workflows/#Multiple-candidate-prior-distributions","page":"Bayesian workflows","title":"Multiple candidate prior distributions","text":"","category":"section"},{"location":"#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"BayesianWorkflows.jl is a package for:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Bayesian parameter inference, and\nSimulation of [mainly,]\nDiscrete-state-space Partially Observed Markov Processes (DPOMP,) in Julia.\nIt also includes tools for things like automated convergence analysis; model comparison and visualisation.","category":"page"},{"location":"#What-are-DPOMP-models?","page":"Introduction","title":"What are DPOMP models?","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Discrete-state-space (DSS) models are used throughout ecology and other scientific domains to represent systems comprised of interacting components (e.g. people or molecules.)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"A well-known example is the Kermack-McKendrick susceptible-infectious-susceptible (SIR) model:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"<img src=\"https://raw.githubusercontent.com/mjb3/BayesianWorkflows.jl/master/docs/img/sir.png\" alt=\"SIR model\" style=\"height: 80px;\"/>","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"See the Simple example for a brief primer on DSS.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"In applied situations (e.g. like a scientific study) such systems are often difficult to directly observe, and so they are referred to in context as being Partially Observed.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The dynamics (how the system changes over time) of the SIR, and other DSS models, can be represented in continuous time by [a set of coupled] Markov Processes. More specifically, we can define a probability density (a 'likelihood function' in Bayesian parlance) that governs the time-evolution of the system under study.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Combining these concepts, we have a general class of statistical model: Discrete-state-space Partially Observed Markov Processes, or Discrete POMP.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Furthermore, given some applicable [partially complete] scientific data, they yield a paradigm for (in this case, Bayesian) statistical inference based on that model class. That is, we can infer [the likely value of] unknown quantities, such as the unknown time of a known event (like the introduction of a pathogen,) or a model parameter that characterises the infectiousness of that pathogen.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"To summarise, DPOMP models and associated methods allow us to learn about a given system of interest (e.g. an ecosystem, pandemic, chemical reaction, and so on,) even in when the available data is limited ['partial'].","category":"page"},{"location":"#Scientific-applications","page":"Introduction","title":"Scientific applications","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Discrete POMP, and discrete-state-space models in general, have a wide range of applications including:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Epidemiological modelling (e.g. SEIR models)\nEcology (e.g. predator-prey dynamics)\nMany other potential use cases, e.g. physics; chemical reactions; social media.","category":"page"},{"location":"#Package-features","page":"Introduction","title":"Package features","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"The algorithms implemented by the package for simulation and inference include:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The Gillespie direct method algorithm\nData-augmented Markov chain Monte Carlo (MCMC)\nThe model-based-proposal (MBP) algorithm[1]\nParticle filters (i.e. Sequential Monte Carlo)\nSMC^2[2], or iterative-batch-importance sampling (IBIS)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"[1]: C. M. Pooley, S. C. Bishop, and G. Marion. Using model-based proposals for fast parameter inference on discrete state space, continuous-time Markov processes. Journal of The Royal Society Interface, 12(107):20150225–20150225, May 2015.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"[2]: N. Chopin, P. E. Jacob, and O. Papaspiliopoulos. SMC^2 : an efficient algorithm for sequential analysis of state space models: Sequential Analysis of State Space Models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 75(3):397–426, June 2013.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"A number of well-known models are provided as predefined examples:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"SIR, SEIR, and other epidemiological model\nThe Lotka-Voltera predator-prey model\nRoss-MacDonald two-species malaria model","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The package code was initially developed during the course of a postgraduate research project in infectious disease modelling at Biostatistics Scotland, and there is a heavy emphasis on epidemiology and epidemiological modelling throughout.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"In practice though, this affects only the applied examples and naming conventions of the predefined models available with the package. Otherwise, the models and methods are applicable to many situations entirely outwith the field of ecology (such as chemical reactions.)","category":"page"},{"location":"#Installation","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"As a prerequisite, the package naturally requires a working installation of the Julia programming language. The package is not yet registered but can nonetheless must be added via the package manager Pkg in the usual way.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"From the Julia REPL, run:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using Pkg\nPkg.add(url=\"https://github.com/ScottishCovidResponse/DataRegistryUtils.jl\")","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"See the package code repository to inspect the source code.","category":"page"},{"location":"#Documentation","page":"Introduction","title":"Documentation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Pages = [\n    \"models.md\",\n    \"examples.md\",\n    \"manual.md\",\n]\nDepth = 2","category":"page"}]
}
