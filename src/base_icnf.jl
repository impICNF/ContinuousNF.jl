export inference, generate, loss

function inference(
    icnf::AbstractICNF{T, AT},
    mode::Mode,
    xs::AbstractVector{<:Real},
    p::AbstractVector{<:Real} = icnf.p,
    args...;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractVector{<:Real} where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode)
    zrs = convert(AT, zeros(T, n_aug + 1))
    f_aug = augmented_f(icnf, mode; differentiation_backend, rng)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, p)
    sol = solve(prob, args...; kwargs...)
    fsol = sol[:, end]
    z = fsol[1:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    vcat(logp̂x, fsol[(end - n_aug + 1):end])
end

function generate(
    icnf::AbstractICNF{T, AT},
    mode::Mode,
    p::AbstractVector{<:Real} = icnf.p,
    args...;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractVector{<:Real} where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode)
    new_xs = convert(AT, rand(rng, icnf.basedist))
    zrs = convert(AT, zeros(T, n_aug + 1))
    f_aug = augmented_f(icnf, mode; differentiation_backend, rng)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, args...; kwargs...)
    fsol = sol[:, end]
    z = fsol[1:(end - n_aug - 1)]
    z
end

function loss(
    icnf::AbstractICNF{T, AT},
    xs::AbstractVector{<:Real},
    p::AbstractVector{<:Real} = icnf.p;
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, = inference(icnf, TrainMode(), xs, p; rng)
    -logp̂x
end

function loss(
    icnf::AbstractICNF{T, AT},
    xs::AbstractMatrix{<:Real},
    p::AbstractVector{<:Real} = icnf.p;
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    Folds.sum(x -> loss(icnf, x, p; rng), eachcol(xs)) / size(xs, 2)
end

function n_augment(
    icnf::AbstractICNF{T, AT},
    mode::Mode,
)::Integer where {T <: AbstractFloat, AT <: AbstractArray}
    0
end

# pretty-printing
function Base.show(io::IO, icnf::AbstractICNF)
    print(
        io,
        typeof(icnf),
        "\n\tNumber of Variables: ",
        icnf.nvars,
        "\n\tTime Span: ",
        icnf.tspan,
    )
end
