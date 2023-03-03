using ICNF,
    AbstractDifferentiation,
    Aqua,
    CUDA,
    ComponentArrays,
    ComputationalResources,
    DataFrames,
    Distributions,
    FiniteDiff,
    FiniteDifferences,
    Flux,
    ForwardDiff,
    Logging,
    Lux,
    MLJBase,
    Optimization,
    Random,
    ReverseDiff,
    Test,
    Tracker,
    Zygote

debuglogger = ConsoleLogger(Logging.Debug)
global_logger(debuglogger)

include("core.jl")

GROUP = get(ENV, "GROUP", "All")
SMALL = get(ENV, "SMALL", "0") == "1"

@testset "Overall" begin
    CUDA.allowscalar() do
        include("smoke_tests.jl")
    end
end

if GROUP == "All" || GROUP == "Quality"
    include("quality_tests.jl")
end
