using ICNF,
    AbstractDifferentiation,
    Aqua,
    CUDA,
    DataFrames,
    DifferentialEquations,
    Distributions,
    FiniteDiff,
    FiniteDifferences,
    Flux,
    ForwardDiff,
    Logging,
    Optimization,
    ReverseDiff,
    MLJBase,
    SciMLBase,
    SciMLSensitivity,
    Test,
    Tracker,
    Zygote

debuglogger = ConsoleLogger(Logging.Debug)
global_logger(debuglogger)

include("core.jl")

GROUP = get(ENV, "GROUP", "All")

@testset "Overall" begin
    CUDA.allowscalar() do
        include("smoke_tests.jl")
    end
end

if GROUP == "All" || GROUP == "Quality"
    include("quality_tests.jl")
end
