export inference, generate, loss

@views function inference_prob(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = zeros_T_AT(resource, icnf, n_aug_input + n_aug + 1)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            ϵ,
            st;
            resource,
            differentiation_backend,
            autodiff_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        cat(xs, zrs; dims = 1),
        steer_tspan(icnf, mode; tspan, steerdist, rng),
        ps;
        sol_kwargs...,
    )
    prob
end

@views function inference(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    prob = inference_prob(
        icnf,
        mode,
        xs,
        ps,
        st;
        tspan,
        steerdist,
        basedist,
        differentiation_backend,
        autodiff_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    n_aug = n_augment(icnf, mode)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = sol[:, end]
    z = fsol[begin:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    logp̂x = logpdf(basedist, z) - Δlogp
    if iszero(n_aug)
        (logp̂x,)
    else
        augs = fsol[(end - n_aug + 1):end]
        (logp̂x, augs...)
    end
end

@views function inference_prob(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = zeros_T_AT(resource, icnf, n_aug_input + n_aug + 1, size(xs, 2))
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input, size(xs, 2))
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            ϵ,
            st;
            resource,
            differentiation_backend,
            autodiff_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        cat(xs, zrs; dims = 1),
        steer_tspan(icnf, mode; tspan, steerdist, rng),
        ps;
        sol_kwargs...,
    )
    prob
end

@views function inference(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    prob = inference_prob(
        icnf,
        mode,
        xs,
        ps,
        st;
        tspan,
        steerdist,
        basedist,
        differentiation_backend,
        autodiff_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    n_aug = n_augment(icnf, mode)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = sol[:, :, end]
    z = fsol[begin:(end - n_aug - 1), :]
    Δlogp = fsol[(end - n_aug), :]
    logp̂x = logpdf(basedist, z) - Δlogp
    if iszero(n_aug)
        (logp̂x,)
    else
        augs = fsol[(end - n_aug + 1):end, :]
        (logp̂x, eachrow(augs)...)
    end
end

@views function generate_prob(
    icnf::AbstractICNF{T, <:VectorMode},
    mode::Mode,
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = rand_cstm_AT(resource, icnf, basedist, rng)
    zrs = zeros_T_AT(resource, icnf, n_aug + 1)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            ϵ,
            st;
            resource,
            differentiation_backend,
            autodiff_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        cat(new_xs, zrs; dims = 1),
        reverse(steer_tspan(icnf, mode; tspan, steerdist, rng)),
        ps;
        sol_kwargs...,
    )
    prob
end
@views function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    prob = generate_prob(
        icnf,
        mode,
        ps,
        st;
        tspan,
        steerdist,
        basedist,
        differentiation_backend,
        autodiff_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = sol[:, end]
    z = fsol[begin:(end - n_aug_input - n_aug - 1)]
    z
end

@views function generate_prob(
    icnf::AbstractICNF{T, <:MatrixMode},
    mode::Mode,
    ps::Any,
    st::Any,
    n::Int;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = rand_cstm_AT(resource, icnf, basedist, rng, n)
    zrs = zeros_T_AT(resource, icnf, n_aug + 1, size(new_xs, 2))
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input, size(new_xs, 2))
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            ϵ,
            st;
            resource,
            differentiation_backend,
            autodiff_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        cat(new_xs, zrs; dims = 1),
        reverse(steer_tspan(icnf, mode; tspan, steerdist, rng)),
        ps;
        sol_kwargs...,
    )
    prob
end

@views function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ps::Any,
    st::Any,
    n::Int;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    prob = generate_prob(
        icnf,
        mode,
        ps,
        st,
        n;
        tspan,
        steerdist,
        basedist,
        differentiation_backend,
        autodiff_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = sol[:, :, end]
    z = fsol[begin:(end - n_aug_input - n_aug - 1), :]
    z
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    -first(
        inference(
            icnf,
            mode,
            xs,
            ps,
            st;
            resource,
            tspan,
            steerdist,
            basedist,
            differentiation_backend,
            autodiff_backend,
            rng,
            sol_args,
            sol_kwargs,
        ),
    )
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    -mean(
        first(
            inference(
                icnf,
                mode,
                xs,
                ps,
                st;
                resource,
                tspan,
                steerdist,
                basedist,
                differentiation_backend,
                autodiff_backend,
                rng,
                sol_args,
                sol_kwargs,
            ),
        ),
    )
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{<:AbstractFloat, <:ADVectorMode},
    mode::TestMode,
    ϵ::AbstractVector{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz, J = AbstractDifferentiation.value_and_jacobian(
        differentiation_backend,
        x -> icnf._fnn(x, p, st),
        z,
    )
    trace_J = tr(only(J))
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::TestMode,
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz, J = jacobian_batched(icnf, x -> icnf._fnn(x, p, st), z; resource, autodiff_backend)
    trace_J = transpose(tr.(eachslice(J; dims = 3)))
    cat(mz, -trace_J; dims = 1)
end

@inline function (icnf::AbstractICNF)(xs::Any, ps::Any, st::Any)
    first(inference(icnf, TrainMode(), xs, ps, st))
end
