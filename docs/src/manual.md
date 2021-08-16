# Package manual
```@contents
Pages = ["manual.md"]
Depth = 3
```

## Types

### Model types
```@docs
DPOMPModel
Particle
Event
Observation
ARQModel
```

### Results
```@docs
SimResults
ImportanceSample
RejectionSample
MCMCSample
ARQMCMCSample
```

## Functions

### Model helpers
```@docs
generate_model
partial_gaussian_obs_model
```

### Simulation
```@docs
gillespie_sim
```

### Bayesian inference

#### Workflows

```@docs
run_inference_workflow
```

#### Algorithms

```@docs
run_smc2_analysis
run_mbp_ibis_analysis
run_mcmc_analysis
run_arq_mcmc_analysis
```

### Bayesian model analysis

```@docs
run_model_comparison_analysis
```

### Utilities
```@docs
get_observations
tabulate_results
save_to_file
```

### Visualisation

```@docs
plot_trajectory
plot_parameter_trace
plot_parameter_marginal
plot_parameter_heatmap
plot_model_comparison
```

## Index
```@index
```
