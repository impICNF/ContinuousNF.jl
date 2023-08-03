export CondICNFModel, CondICNFDist

# SciML interface

function loss_f(icnf::AbstractCondICNF, loss::Function, st::Any)
    @inline function f(ps, θ, xs, ys)
        loss(icnf, TrainMode(), xs, ys, ps, st)
    end
    f
end

# MLJ interface

mutable struct CondICNFModel <: MLJICNF
    m::AbstractCondICNF
    loss::Function

    optimizers::AbstractVector
    n_epochs::Integer
    adtype::ADTypes.AbstractADType

    use_batch::Bool
    batch_size::Integer
    have_callback::Bool

    data_type::Type{<:AbstractFloat}
    compute_mode::Type{<:ComputeMode}
end

function CondICNFModel(
    m::AbstractCondICNF{T, CM},
    loss::Function = loss;
    optimizers::AbstractVector = Any[Optimisers.Lion(),],
    n_epochs::Integer = 300,
    adtype::ADTypes.AbstractADType = AutoZygote(),
    use_batch::Bool = true,
    batch_size::Integer = 32,
    have_callback::Bool = true,
) where {T <: AbstractFloat, CM <: ComputeMode}
    CondICNFModel(
        m,
        loss,
        optimizers,
        n_epochs,
        adtype,
        use_batch,
        batch_size,
        have_callback,
        T,
        CM,
    )
end

function MLJModelInterface.fit(model::CondICNFModel, verbosity, XY)
    rng = Random.default_rng()
    X, Y = XY
    x = collect(transpose(MLJModelInterface.matrix(X)))
    y = collect(transpose(MLJModelInterface.matrix(Y)))
    ps, st = LuxCore.setup(rng, model.m)
    if !(model.m isa FluxCompatLayer)
        ps = ComponentArray(ps)
    end
    if model.m.resource isa CUDALibs
        gdev = gpu_device()
        x = gdev(x)
        y = gdev(y)
        ps = gdev(ps)
        st = gdev(st)
    end
    _loss = loss_f(model.m, model.loss, st)
    optfunc = OptimizationFunction(_loss, model.adtype)
    optprob = OptimizationProblem(optfunc, ps)
    tst_overall = @timed for opt in model.optimizers
        tst_epochs = @timed for ep in 1:(model.n_epochs)
            if model.use_batch
                if model.compute_mode <: VectorMode
                    data = DataLoader(
                        (x, y);
                        batchsize = -1,
                        shuffle = true,
                        partial = true,
                        parallel = false,
                        buffer = false,
                    )
                elseif model.compute_mode <: MatrixMode
                    data = DataLoader(
                        (x, y);
                        batchsize = model.batch_size,
                        shuffle = true,
                        partial = true,
                        parallel = false,
                        buffer = false,
                    )
                else
                    error("Not Implemented")
                end
            else
                data = [(x, y)]
            end
            optprob_re = remake(optprob; u0 = ps)
            if model.have_callback
                prgr = Progress(
                    length(data);
                    desc = "Fitting (epoch: $ep of $(model.n_epochs)): ",
                    showspeed = true,
                )
                _callback = callback_f(model.m, prgr)
                tst_one =
                    @timed res = solve(optprob_re, opt, data; callback = _callback)
                ProgressMeter.finish!(prgr)

            else
                tst_one = @timed res = solve(optprob_re, opt, data)
            end
            ps .= res.u
            @info(
                "Fitting (epoch: $ep of $(model.n_epochs)) - $(typeof(opt).name.name)",
                "elapsed time (seconds)" = tst_one.time,
                "garbage collection time (seconds)" = tst_one.gctime,
            )
        end
        @info(
            "Fitting (all epochs) - $(typeof(opt).name.name)",
            "elapsed time (seconds)" = tst_epochs.time,
            "garbage collection time (seconds)" = tst_epochs.gctime,
        )
    end
    @info(
        "Fitting - Overall",
        "elapsed time (seconds)" = tst_overall.time,
        "garbage collection time (seconds)" = tst_overall.gctime,
    )

    fitresult = (ps, st)
    cache = nothing
    report = (stats = tst_overall,)
    fitresult, cache, report
end

function MLJModelInterface.transform(model::CondICNFModel, fitresult, XYnew)
    Xnew, Ynew = XYnew
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    ynew = collect(transpose(MLJModelInterface.matrix(Ynew)))
    if model.m.resource isa CUDALibs
        gdev = gpu_device()
        xnew = gdev(xnew)
        ynew = gdev(ynew)
    end
    (ps, st) = fitresult

    if model.compute_mode <: VectorMode
        tst = @timed logp̂x = broadcast(
            ((x, y),) -> first(inference(model.m, TestMode(), x, y, ps, st)),
            zip(eachcol(xnew), eachcol(ynew)),
        )
    elseif model.compute_mode <: MatrixMode
        tst = @timed logp̂x = first(inference(model.m, TestMode(), xnew, ynew, ps, st))
    else
        error("Not Implemented")
    end
    @info(
        "Transforming",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
    )

    DataFrame(; px = exp.(logp̂x))
end

MLJBase.metadata_pkg(
    CondICNFModel;
    package_name = "ContinuousNormalizingFlows",
    package_uuid = "00b1973d-5b2e-40bf-8604-5c9c1d8f50ac",
    package_url = "https://github.com/impICNF/ContinuousNormalizingFlows.jl",
    is_pure_julia = true,
    package_license = "MIT",
    is_wrapper = false,
)
MLJBase.metadata_model(
    CondICNFModel;
    input_scitype = Tuple{
        Table{AbstractVector{ScientificTypes.Continuous}},
        Table{AbstractVector{ScientificTypes.Continuous}},
    },
    target_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    output_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    supports_weights = false,
    load_path = "ContinuousNormalizingFlows.CondICNFModel",
)

# Distributions interface

struct CondICNFDist <: ICNFDistribution
    m::AbstractCondICNF
    mode::Mode
    ys::AbstractVecOrMat{<:Real}
    ps::Any
    st::Any
end

function CondICNFDist(
    mach::Machine{<:CondICNFModel},
    mode::Mode,
    ys::AbstractVecOrMat{<:Real},
)
    (ps, st) = fitted_params(mach)
    CondICNFDist(mach.model.m, mode, ys, ps, st)
end

function Base.length(d::CondICNFDist)
    d.m.nvars
end
function Base.eltype(d::CondICNFDist)
    typeof(d.m).parameters[1]
end
function Distributions._logpdf(d::CondICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractCondICNF{<:AbstractFloat, <:VectorMode}
        first(inference(d.m, d.mode, x, d.ys, d.ps, d.st))
    elseif d.m isa AbstractCondICNF{<:AbstractFloat, <:MatrixMode}
        first(Distributions._logpdf(d, hcat(x)))
    else
        error("Not Implemented")
    end
end
function Distributions._logpdf(d::CondICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractCondICNF{<:AbstractFloat, <:VectorMode}
        broadcast(x -> Distributions._logpdf(d, x), eachcol(A))
    elseif d.m isa AbstractCondICNF{<:AbstractFloat, <:MatrixMode}
        first(inference(d.m, d.mode, A, d.ys[:, begin:size(A, 2)], d.ps, d.st))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::CondICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractCondICNF{<:AbstractFloat, <:VectorMode}
        x .= generate(d.m, d.mode, d.ys, d.ps, d.st; rng)
    elseif d.m isa AbstractCondICNF{<:AbstractFloat, <:MatrixMode}
        x .= Distributions._rand!(rng, d, hcat(x))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::CondICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractCondICNF{<:AbstractFloat, <:VectorMode}
        A .= hcat(broadcast(x -> Distributions._rand!(rng, d, x), eachcol(A))...)
    elseif d.m isa AbstractCondICNF{<:AbstractFloat, <:MatrixMode}
        A .= generate(d.m, d.mode, d.ys[:, begin:size(A, 2)], d.ps, d.st, size(A, 2); rng)
    else
        error("Not Implemented")
    end
end
