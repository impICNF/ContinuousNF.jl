import ADTypes,
    Aqua,
    ComponentArrays,
    ComputationalResources,
    DataFrames,
    DifferentiationInterface,
    Distances,
    Distributions,
    Enzyme,
    JET,
    Logging,
    Lux,
    MLJBase,
    SciMLBase,
    StableRNGs,
    TerminalLoggers,
    Test,
    ContinuousNormalizingFlows

Enzyme.API.fast_math!(false)
Enzyme.API.inlineall!(true)
Enzyme.API.instname!(true)
Enzyme.API.looseTypeAnalysis!(false)
Enzyme.API.maxtypedepth!(typemax(Int))
Enzyme.API.maxtypeoffset!(typemax(Int))
Enzyme.API.memmove_warning!(true)
Enzyme.API.strictAliasing!(false)
Enzyme.API.strong_zero!(false)
Enzyme.API.typeWarning!(true)

GROUP = get(ENV, "GROUP", "All")

if (GROUP == "All")
    GC.enable_logging(true)

    debuglogger = TerminalLoggers.TerminalLogger(stderr, Logging.Debug)
    Logging.global_logger(debuglogger)
end

Test.@testset "Overall" begin
    if GROUP == "All" ||
       GROUP in ["RNODE", "FFJORD", "Planar", "CondRNODE", "CondFFJORD", "CondPlanar"]
        include("smoke_tests.jl")
    end

    if GROUP == "All" || GROUP == "Quality"
        include("quality_tests.jl")
    end

    if GROUP == "All" || GROUP == "Instability"
        include("instability_tests.jl")
    end

    if GROUP == "All" || GROUP == "Regression"
        include("regression_tests.jl")
    end
end
