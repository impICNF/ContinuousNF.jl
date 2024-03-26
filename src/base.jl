export construct

function construct(
    aicnf::Type{<:AbstractFlows},
    nn::LuxCore.AbstractExplicitLayer,
    nvars::Int,
    naugmented::Int = 0;
    data_type::Type{<:AbstractFloat} = Float32,
    compute_mode::Type{<:ComputeMode} = ADVecJacVectorMode,
    inplace::Bool = false,
    cond::Bool = aicnf <: Union{CondRNODE, CondFFJORD, CondPlanar},
    resource::AbstractResource = CPU1(),
    basedist::Distribution = MvNormal(
        Zeros{data_type}(nvars + naugmented),
        Eye{data_type}(nvars + naugmented),
    ),
    tspan::NTuple{2} = (zero(data_type), one(data_type)),
    steer_rate::AbstractFloat = zero(data_type),
    epsdist::Distribution = MvNormal(
        Zeros{data_type}(nvars + naugmented),
        Eye{data_type}(nvars + naugmented),
    ),
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    autodiff_backend::ADTypes.AbstractADType = ifelse(
        compute_mode <: SDJacVecMatrixMode,
        AutoForwardDiff(),
        AutoZygote(),
    ),
    sol_kwargs::NamedTuple = (
        save_everystep = false,
        alg = Tsit5(; thread = OrdinaryDiffEq.True()),
    ),
    rng::AbstractRNG = rng_AT(resource),
    λ₁::AbstractFloat = if aicnf <: Union{RNODE, CondRNODE}
        convert(data_type, 1e-2)
    else
        zero(data_type)
    end,
    λ₂::AbstractFloat = if aicnf <: Union{RNODE, CondRNODE}
        convert(data_type, 1e-2)
    else
        zero(data_type)
    end,
    λ₃::AbstractFloat = zero(data_type),
)
    steerdist = Uniform{data_type}(-steer_rate, steer_rate)

    ICNF{
        data_type,
        compute_mode,
        inplace,
        cond,
        !iszero(naugmented),
        !iszero(steer_rate),
        !iszero(λ₁),
        !iszero(λ₂),
        !iszero(λ₃),
        typeof(nn),
        typeof(nvars),
        typeof(resource),
        typeof(basedist),
        typeof(tspan),
        typeof(steerdist),
        typeof(epsdist),
        typeof(differentiation_backend),
        typeof(autodiff_backend),
        typeof(sol_kwargs),
        typeof(rng),
    }(
        nn,
        nvars,
        naugmented,
        resource,
        basedist,
        tspan,
        steerdist,
        epsdist,
        differentiation_backend,
        autodiff_backend,
        sol_kwargs,
        rng,
        λ₁,
        λ₂,
        λ₃,
    )
end

@inline function n_augment(::AbstractFlows, ::Mode)
    0
end

# pretty-printing

function Base.show(io::IO, icnf::AbstractFlows)
    print(
        io,
        typeof(icnf),
        "\n\tNumber of Variables: ",
        icnf.nvars,
        "\n\tNumber of Augmentations: ",
        n_augment_input(icnf),
        "\n\tTime Span: ",
        icnf.tspan,
    )
end

@inline function n_augment_input(
    icnf::AbstractFlows{<:AbstractFloat, <:ComputeMode, INPLACE, COND, true},
) where {INPLACE, COND}
    icnf.naugmented
end

@inline function n_augment_input(::AbstractFlows)
    0
end

@inline function steer_tspan(
    icnf::AbstractFlows{T, <:ComputeMode, INPLACE, COND, AUGMENTED, true},
    ::TrainMode,
) where {T <: AbstractFloat, INPLACE, COND, AUGMENTED}
    t₀, t₁ = icnf.tspan
    Δt = abs(t₁ - t₀)
    r = convert(T, rand(icnf.rng, icnf.steerdist))
    t₁_new = muladd(Δt, r, t₁)
    (t₀, t₁_new)
end

@inline function steer_tspan(icnf::AbstractFlows, ::Mode)
    icnf.tspan
end

@inline function rng_AT(::AbstractResource)
    Random.default_rng()
end

@inline function base_AT(
    ::AbstractResource,
    ::AbstractFlows{T},
    dims...,
) where {T <: AbstractFloat}
    Array{T}(undef, dims...)
end

@non_differentiable base_AT(::Any...)

function inference_sol(
    icnf::AbstractFlows{T, <:VectorMode, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    sol = solve(prob; icnf.sol_kwargs...)
    fsol = get_fsol(sol)
    z = fsol[begin:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    augs = fsol[(end - n_aug + 1):end]
    logpz = oftype(Δlogp, logpdf(icnf.basedist, z))
    logp̂x = logpz - Δlogp
    Ȧ = if (NORM_Z_AUG && AUGMENTED)
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end]
        norm(z_aug)
    else
        zero(T)
    end
    (logp̂x, vcat(augs, Ȧ))
end

function inference_sol(
    icnf::AbstractFlows{T, <:MatrixMode, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    sol = solve(prob; icnf.sol_kwargs...)
    fsol = get_fsol(sol)
    z = fsol[begin:(end - n_aug - 1), :]
    Δlogp = fsol[(end - n_aug), :]
    augs = fsol[(end - n_aug + 1):end, :]
    logpz = oftype(Δlogp, logpdf(icnf.basedist, z))
    logp̂x = logpz - Δlogp
    Ȧ = transpose(if (NORM_Z_AUG && AUGMENTED)
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end, :]
        norm.(eachcol(z_aug))
    else
        zrs_aug = similar(augs, size(augs, 2))
        @ignore_derivatives fill!(zrs_aug, zero(T))
        zrs_aug
    end)
    (logp̂x, eachrow(vcat(augs, Ȧ)))
end

function generate_sol(
    icnf::AbstractFlows{T, <:VectorMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob; icnf.sol_kwargs...)
    fsol = get_fsol(sol)
    z = fsol[begin:(end - n_aug_input - n_aug - 1)]
    z
end

function generate_sol(
    icnf::AbstractFlows{T, <:MatrixMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob; icnf.sol_kwargs...)
    fsol = get_fsol(sol)
    z = fsol[begin:(end - n_aug_input - n_aug - 1), :]
    z
end

@inline function get_fsol(sol::SciMLBase.AbstractODESolution)
    last(sol.u)
end

@inline function get_fsol(sol::AbstractArray{T, N}) where {T, N}
    selectdim(sol, N, lastindex(sol, N))
end
