# ContinuousNormalizingFlows.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://impICNF.github.io/ContinuousNormalizingFlows.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://impICNF.github.io/ContinuousNormalizingFlows.jl/dev)
[![Version](https://juliahub.com/docs/General/ContinuousNormalizingFlows/stable/version.svg)](https://juliahub.com/ui/Packages/General/ContinuousNormalizingFlows)
[![Deps](https://juliahub.com/docs/General/ContinuousNormalizingFlows/stable/deps.svg)](https://juliahub.com/ui/Packages/General/ContinuousNormalizingFlows?t=2)
[![Build Status](https://github.com/impICNF/ContinuousNormalizingFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/impICNF/ContinuousNormalizingFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Codecov Coverage](https://codecov.io/gh/impICNF/ContinuousNormalizingFlows.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/impICNF/ContinuousNormalizingFlows.jl)
[![Coveralls Coverage](https://coveralls.io/repos/github/impICNF/ContinuousNormalizingFlows.jl/badge.svg?branch=main)](https://coveralls.io/github/impICNF/ContinuousNormalizingFlows.jl?branch=main)
[![PkgEval](https://juliahub.com/docs/General/ContinuousNormalizingFlows/stable/pkgeval.svg)](https://juliahub.com/ui/Packages/General/ContinuousNormalizingFlows)
[![Pkg Eval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/C/ContinuousNormalizingFlows.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/C/ContinuousNormalizingFlows.html)
[![Monthly Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FContinuousNormalizingFlows&query=total_requests&suffix=%2Fmonth&label=Monthly%20Downloads)](https://juliapkgstats.com/pkg/ContinuousNormalizingFlows)
[![Total Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FContinuousNormalizingFlows&query=total_requests&suffix=%2Fmonth&label=Total%20Downloads)](https://juliapkgstats.com/pkg/ContinuousNormalizingFlows)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Implementations of Infinitesimal Continuous Normalizing Flows Algorithms in Julia

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).

## Installation

```julia
using Pkg
Pkg.add("ContinuousNormalizingFlows")
```

## Usage

```julia
# Enable Logging
using Logging, TerminalLoggers
global_logger(TerminalLogger())

# Parameters
nvars = 1
naugs = nvars
# n_in = nvars # without augmentation
n_in = nvars + naugs # with augmentation
n = 1024

# Model
using ContinuousNormalizingFlows, Lux, ADTypes #, Enzyme, CUDA, ComputationalResources
nn = Chain(Dense(n_in => 3 * n_in, tanh), Dense(3 * n_in => n_in, tanh))
icnf = construct(
    RNODE,
    nn,
    nvars, # number of variables
    naugs; # number of augmented dimensions
    # compute_mode = DIVecJacMatrixMode(AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation = Enzyme.Const)), # process data in batches and use Enzyme
    # inplace = true, # use the inplace version of functions
    # resource = CUDALibs(), # process data by GPU
    tspan = (0.0f0, 13.0f0), # have bigger time span
    steer_rate = 1.0f-1, # add random noise to end of the time span
    λ₁ = 1.0f-2, # regulate flow
    λ₂ = 1.0f-2, # regulate volume change
    λ₃ = 1.0f-2, # regulate augmented dimensions
    sol_kwargs = (;
        progress = true,
        save_everystep = false,
        reltol = sqrt(eps(one(Float32))),
        abstol = eps(one(Float32)),
        maxiters = typemax(Int32),
    ), # pass to the solver
)

# Data
using Distributions
data_dist = Beta{Float32}(2.0f0, 4.0f0)
r = rand(data_dist, nvars, n)
r = convert.(Float32, r)

# Fit It
using DataFrames, MLJBase #, Zygote, ADTypes, OptimizationOptimisers
df = DataFrame(transpose(r), :auto)
model = ICNFModel(
    icnf;
    # optimizers = (Lion(),),
    # n_epochs = 300,
    # adtype = AutoZygote(),
    # use_batch = true,
    # batch_size = 32,
    sol_kwargs = (; progress = true,), # pass to the solver
)
mach = machine(model, df)
fit!(mach)
ps, st = fitted_params(mach)

# Store It
using JLD2, UnPack
jldsave("fitted.jld2"; ps, st) # save
@unpack ps, st = load("fitted.jld2") # load

# Use It
d = ICNFDist(icnf, TestMode(), ps, st) # direct way
# d = ICNFDist(mach, TestMode()) # alternative way
actual_pdf = pdf.(data_dist, vec(r))
estimated_pdf = pdf(d, r)
new_data = rand(d, n)

# Evaluate It
using Distances
mad_ = meanad(estimated_pdf, actual_pdf)
msd_ = msd(estimated_pdf, actual_pdf)
tv_dis = totalvariation(estimated_pdf, actual_pdf) / n
res_df = DataFrame(; mad_, msd_, tv_dis)
display(res_df)

# Plot It
using CairoMakie
f = Figure()
ax = Makie.Axis(f[1, 1]; title = "Result")
lines!(ax, 0.0f0 .. 1.0f0, x -> pdf(data_dist, x); label = "actual")
lines!(ax, 0.0f0 .. 1.0f0, x -> pdf(d, vcat(x)); label = "estimated")
axislegend(ax)
save("result-fig.svg", f)
save("result-fig.png", f)
```
