export Planar, PlanarNN

struct PlanarNN
    u::AbstractVector{<:Real}
    w::AbstractVector{<:Real}
    b::AbstractVector{<:Real}

    h::Function
end

function PlanarNN(
    nvars::Integer,
    h::Function = tanh;
    cond = false,
    rng::AbstractRNG = Random.default_rng(),
)
    u = randn(rng, nvars)
    w = randn(rng, cond ? nvars * 2 : nvars)
    b = randn(rng, 1)
    PlanarNN(u, w, b, h)
end

@functor PlanarNN (u, w, b)

function (m::PlanarNN)(z::AbstractVecOrMat)::AbstractVecOrMat
    u, w, b = m.u, m.w, only(m.b)
    h = NNlib.fast_act(m.h, z)
    u * h.(muladd(transpose(w), z, b))
end

function pl_h(m::PlanarNN, z::AbstractVecOrMat)::AbstractVecOrMat
    u, w, b = m.u, m.w, only(m.b)
    h = NNlib.fast_act(m.h, z)
    h.(muladd(transpose(w), z, b))
end

"""
Implementation of Planar Flows from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct Planar{T <: AbstractFloat, AT <: AbstractArray} <: AbstractICNF{T, AT}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    # trace_test
    # trace_train
end

function Planar{T, AT}(
    nn::PlanarNN,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, default_tspan),
    rng::AbstractRNG = Random.default_rng(),
) where {T <: AbstractFloat, AT <: AbstractArray}
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    Planar{T, AT}(re, convert(AT{T}, p), nvars, basedist, tspan)
end

function augmented_f(
    icnf::Planar{T, AT},
    mode::Mode;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:(end - n_aug)]
        mz = m(z)
        trace_J = transpose(m.u) * only(AbstractDifferentiation.jacobian(differentiation_backend, x -> pl_h(m, x), z))
        vcat(mz, -trace_J)
    end
    f_aug
end

@functor Planar (p,)
