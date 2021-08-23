# Bayesian workflows
This page gives a brief overview of the simulation and inference features of the package.

## Model simulation

DPOMP models can be simulated using the Gillespie algorithm, which is invoked as follows:

```
using BayesianWorkflows
initial_condition = [100, 1]    # define a model
model = generate_model("SIS", initial_condition)

theta = [0.003, 0.1]            # model parameters
x = gillespie_sim(model, theta)	# run simulation
println(plot_trajectory(x))     # plot (optional)
```

See the [gillespie_sim](@ref) entry in the package manual for more details.

Simulations also provide a vector of simulated observations (`x.observations`) that can be used to try out the inference features of the package.

## Inference workflows

The package implements three automated Bayesian inference workflows, covering both single-model [i.e. paramerter-] inference and multi-model inference (or 'model comparison'.)

### [Single-model] parameter inference
The purpose of the first workflow is to 'infer' the [likely] model parameters, given a set of observations data `y`.

```
y = x.observations            # using the simulated observations (see above)
results = run_inference_workflow(model, prior, y)
tabulate_results(results)
```

### Model comparison
The second workflow is for situations where we have multiple candidate models and wish to formally evaluate them with respect to some observations data `y`.

```
# define alternative model
seis_model = generate_model("SEIS", [100, 0, 1])
seis_model.obs_model = partial_gaussian_obs_model(2.0, seq = 3, y_seq = 2)
seis_prior = Distributions.Product(Distributions.Uniform.(zeros(3), [0.1,0.5,0.5]))

# run model comparison workflow
models::Array{DPOMPModel, 1} = [model, seis_model]
priors::Array{Distributions.Distribution, 1} = [prior, seis_prior]
results = run_inference_workflow(models, priors, y)
tabulate_results(results)
```

### Multiple candidate prior distributions
WIP

## Inference algorithms
TBA
