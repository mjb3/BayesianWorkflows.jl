using Documenter, BayesianWorkflows

makedocs(sitename="BayesianWorkflows.jl docs", pages = ["index.md", "manual.md", "models.md"])#, "workflows.md"
#, "examples.md", "models.md"

## nb. called by GitHub Actions wf
# - local version deploys to build dir
deploydocs(
    repo = "github.com/mjb3/BayesianWorkflows.jl.git",
    devbranch = "main",
)
