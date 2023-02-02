module ICNF

using AbstractDifferentiation,
    CUDA,
    ComponentArrays,
    ComputationalResources,
    DataFrames,
    Distributions,
    DistributionsAD,
    FillArrays,
    IterTools,
    Lux,
    LuxCore,
    MLJBase,
    MLJModelInterface,
    MLUtils,
    NNlib,
    NNlibCUDA,
    Optimisers,
    Optimization,
    OptimizationOptimisers,
    SciMLBase,
    ScientificTypes,
    Zygote,
    LinearAlgebra,
    Random,
    Statistics,
    Base.Iterators

include("types.jl")
include("defaults.jl")

include("planar_layer.jl")

include("base_icnf.jl")
include("base_cond_icnf.jl")
include("core_icnf.jl")
include("core_cond_icnf.jl")

include("rnode.jl")
include("ffjord.jl")
include("planar.jl")

include("cond_rnode.jl")
include("cond_ffjord.jl")
include("cond_planar.jl")

end
