mutable struct ICNFModel{AICNF <: AbstractICNF} <: MLJICNF{AICNF}
    m::AICNF
    loss::Function

    optimizers::Tuple
    n_epochs::Int
    adtype::ADTypes.AbstractADType

    use_batch::Bool
    batch_size::Int
    sol_kwargs::NamedTuple
end

function ICNFModel(
    m::AbstractICNF,
    loss::Function = loss;
    optimizers::Tuple = (Optimisers.Lion(),),
    n_epochs::Int = 300,
    adtype::ADTypes.AbstractADType = ADTypes.AutoZygote(),
    use_batch::Bool = true,
    batch_size::Int = 32,
    sol_kwargs::NamedTuple = (;),
)
    ICNFModel(m, loss, optimizers, n_epochs, adtype, use_batch, batch_size, sol_kwargs)
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    x = collect(transpose(MLJModelInterface.matrix(X)))
    tdev = if model.m.resource isa ComputationalResources.CUDALibs
        Lux.gpu_device()
    else
        Lux.cpu_device()
    end
    ps, st = LuxCore.setup(model.m.rng, model.m)
    ps = ComponentArrays.ComponentArray(ps)
    x = tdev(x)
    ps = tdev(ps)
    st = tdev(st)
    data = if model.m.compute_mode isa VectorMode
        MLUtils.DataLoader((x,); batchsize = -1, shuffle = true, partial = true)
    elseif model.m.compute_mode isa MatrixMode
        MLUtils.DataLoader(
            (x,);
            batchsize = if model.use_batch
                model.batch_size
            else
                size(x, 2)
            end,
            shuffle = true,
            partial = true,
        )
    else
        error("Not Implemented")
    end
    data = tdev(data)
    optfunc = SciMLBase.OptimizationFunction(
        make_opt_loss(model.m, TrainMode(), st, model.loss),
        model.adtype,
    )
    optprob = SciMLBase.OptimizationProblem(optfunc, ps, data)

    tst_overall = @timed for opt in model.optimizers
        optprob_re = SciMLBase.remake(optprob; u0 = ps)
        tst_epochs = @timed res = SciMLBase.solve(
            optprob_re,
            opt;
            epochs = model.n_epochs,
            model.sol_kwargs...,
        )
        ps .= res.u
        @info(
            "Fitting (all epochs) - $(typeof(opt).name.name)",
            "elapsed time (seconds)" = tst_epochs.time,
            "garbage collection time (seconds)" = tst_epochs.gctime,
            "allocated (bytes)" = tst_epochs.bytes,
        )
    end
    @info(
        "Fitting - Overall",
        "elapsed time (seconds)" = tst_overall.time,
        "garbage collection time (seconds)" = tst_overall.gctime,
        "allocated (bytes)" = tst_overall.bytes,
    )

    fitresult = (ps, st)
    cache = nothing
    report = (stats = tst_overall,)
    (fitresult, cache, report)
end

function MLJModelInterface.transform(model::ICNFModel, fitresult, Xnew)
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    tdev = if model.m.resource isa ComputationalResources.CUDALibs
        Lux.gpu_device()
    else
        Lux.cpu_device()
    end
    xnew = tdev(xnew)
    (ps, st) = fitresult

    tst = @timed if model.m.compute_mode isa VectorMode
        logp̂x = broadcast(function (x)
                first(inference(model.m, TestMode(), x, ps, st))
            end, eachcol(xnew))
    elseif model.m.compute_mode isa MatrixMode
        logp̂x = first(inference(model.m, TestMode(), xnew, ps, st))
    else
        error("Not Implemented")
    end

    @info(
        "Transforming",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
        "allocated (bytes)" = tst.bytes,
    )

    DataFrames.DataFrame(; px = exp.(logp̂x))
end

MLJBase.metadata_pkg(
    ICNFModel;
    package_name = "ContinuousNormalizingFlows",
    package_uuid = "00b1973d-5b2e-40bf-8604-5c9c1d8f50ac",
    package_url = "https://github.com/impICNF/ContinuousNormalizingFlows.jl",
    is_pure_julia = true,
    package_license = "MIT",
    is_wrapper = false,
)
MLJBase.metadata_model(
    ICNFModel;
    input_scitype = ScientificTypesBase.Table{
        AbstractVector{ScientificTypesBase.Continuous},
    },
    target_scitype = ScientificTypesBase.Table{
        AbstractVector{ScientificTypesBase.Continuous},
    },
    output_scitype = ScientificTypesBase.Table{
        AbstractVector{ScientificTypesBase.Continuous},
    },
    supports_weights = false,
    load_path = "ContinuousNormalizingFlows.ICNFModel",
)
