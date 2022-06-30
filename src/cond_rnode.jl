export CondRNODE

"""
Implementation of RNODE (Conditional Version)
"""
struct CondRNODE{T <: AbstractFloat} <: AbstractCondICNF{T}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    differentiation_backend_test::AbstractDifferentiation.AbstractBackend
    differentiation_backend_train::AbstractDifferentiation.AbstractBackend

    solver_test::SciMLBase.AbstractODEAlgorithm
    solver_train::SciMLBase.AbstractODEAlgorithm

    sensealg_test::SciMLBase.AbstractSensitivityAlgorithm
    sensealg_train::SciMLBase.AbstractSensitivityAlgorithm

    acceleration::AbstractResource
    array_mover::Function

    # trace_test
    # trace_train
end

function CondRNODE{T}(
        nn,
        nvars::Integer,
        ;
        basedist::Distribution=MvNormal(zeros(T, nvars), Diagonal(ones(T, nvars))),
        tspan::Tuple{T, T}=convert(Tuple{T, T}, (0, 1)),

        differentiation_backend_test::AbstractDifferentiation.AbstractBackend=default_differentiation_backend(),
        differentiation_backend_train::AbstractDifferentiation.AbstractBackend=default_differentiation_backend(),

        solver_test::SciMLBase.AbstractODEAlgorithm=default_solver,
        solver_train::SciMLBase.AbstractODEAlgorithm=default_solver,

        sensealg_test::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,
        sensealg_train::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,

        acceleration::AbstractResource=default_acceleration,
        ) where {T <: AbstractFloat}
    array_mover = make_mover(acceleration, T)
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    CondRNODE{T}(
        re, p |> array_mover, nvars, basedist, tspan,
        differentiation_backend_test, differentiation_backend_train,
        solver_test, solver_train,
        sensealg_test, sensealg_train,
        acceleration, array_mover,
    )
end

function augmented_f(icnf::CondRNODE{T}, mode::TestMode, ys::AbstractMatrix{T})::Function where {T <: AbstractFloat}

    function f_aug(u, p, t)
        m = Chain(
            x -> vcat(x, ys),
            icnf.re(p),
        )
        z = u[1:end - 1, :]
        ż = m(z)
        l̇ = transpose(div_batch_full(m, z, icnf.differentiation_backend_test))
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(l̇)))
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

function augmented_f(icnf::CondRNODE{T}, mode::TrainMode, ys::AbstractMatrix{T}, sz::Tuple{T2, T2}; rng::AbstractRNG=Random.default_rng())::Function where {T <: AbstractFloat, T2 <: Integer}
    ϵ = randn(rng, T, icnf.nvars) |> icnf.array_mover

    function f_aug(u, p, t)
        m = Chain(
            x -> vcat(x, ys),
            icnf.re(p),
        )
        z = u[1:end - 3, :]
        ż = m(z)
        l̇ = transpose(div_batch_est(m, z, icnf.differentiation_backend_train, ϵ))
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(l̇)))
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

function inference(icnf::CondRNODE{T}, mode::TestMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractVector where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    ys = ys |> icnf.array_mover
    zrs = zeros(T, 3, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false, true}(func, vcat(xs, zrs), icnf.tspan, p; alg=icnf.solver_test, sensealg=icnf.sensealg_test)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 3, :]
    Δlogp = fsol[end - 2, :]
    Ė = fsol[end - 1, :]
    ṅ = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x, Ė, ṅ
end

function inference(icnf::CondRNODE{T}, mode::TrainMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::Tuple where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    ys = ys |> icnf.array_mover
    zrs = zeros(T, 3, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys; rng)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false, true}(func, vcat(xs, zrs), icnf.tspan, p; alg=icnf.solver_train, sensealg=icnf.sensealg_train)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 3, :]
    Δlogp = fsol[end - 2, :]
    Ė = fsol[end - 1, :]
    ṅ = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x, Ė, ṅ
end

function generate(icnf::CondRNODE{T}, mode::TestMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix{T} where {T <: AbstractFloat}
    ys = ys |> icnf.array_mover
    new_xs = rand(rng, icnf.basedist, n) |> icnf.array_mover
    zrs = zeros(T, 3, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false, true}(func, vcat(new_xs, zrs), reverse(icnf.tspan), p; alg=icnf.solver_test, sensealg=icnf.sensealg_test)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 3, :]
    z
end

function generate(icnf::CondRNODE{T}, mode::TrainMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix{T} where {T <: AbstractFloat}
    ys = ys |> icnf.array_mover
    new_xs = rand(rng, icnf.basedist, n) |> icnf.array_mover
    zrs = zeros(T, 3, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys; rng)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false, true}(func, vcat(new_xs, zrs), reverse(icnf.tspan), p; alg=icnf.solver_train, sensealg=icnf.sensealg_train)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 3, :]
    z
end

Flux.@functor CondRNODE (p,)

function loss(icnf::CondRNODE{T}, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p, λ₁::T=convert(T, 1e-2), λ₂::T=convert(T, 1e-2); agg::Function=mean) where {T <: AbstractFloat}
    logp̂x, Ė, ṅ = inference(icnf, TrainMode(), xs, ys, p)
    agg(-logp̂x + λ₁*Ė + λ₂*ṅ)
end
