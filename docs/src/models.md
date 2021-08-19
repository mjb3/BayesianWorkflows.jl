# DPOMP models
**Discrete-state space Partially Observed Markov Processes**

## Predefined models

The package includes the following (mostly) epidemiological models as predefined examples:
- `"SI"`
- `"SIR"`
- `"SIS"`
- `"SEI"`
- `"SEIR"`
- `"SEIS"`
- `"SEIRS"`
- `"PREDPREY"`
- `"ROSSMAC"`

They can be instantiated using the `generate_model` function:

``` julia
using BayesianWorkflows
initial_model_state = [100, 1]
model = generate_model("SIS", initial_model_state)
```

NB. this generates a standard configuration of the named model that can be tweaked later.

### Options
Aside from the model name and initial model state (i.e. the initial 'population',) there are a number of options for generating a default model configuration:

1. `freq_dep`    -- epidemiological models only, set to `true` for frequency-dependent contact rates.
1. `obs_error`   -- average observation error (default = 2.)
1. `t0_index`    -- index of the parameter that represents the initial time. `0` if fixed at `0.0`.

## Full model configuration

```@docs
DPOMPModel
```
