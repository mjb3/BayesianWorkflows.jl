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

Simulations also provide a vector of simulated observations (`x.observations`) that can be used to try out the inference features of the package.

```@docs
gillespie_sim
```

## Inference workflows

The package provides workflows for both single-model [i.e. paramerter-] inference and multi-model inference (or 'model comparison'.)

### [Single-model] parameter inference



### Model comparison

### Multiple candidate prior distributions
