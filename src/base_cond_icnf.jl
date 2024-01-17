export inference, generate, loss

@views function inference_prob(
    icnf::AbstractCondICNF{T, <:VectorMode, INPLACE},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1)
    @ignore_derivatives fill!(zrs, zero(T))
    ϵ = randn(icnf.rng, T, icnf.nvars + n_aug_input)
    ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        ifelse(
            INPLACE,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ys, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
            end,
        ),
        vcat(xs, zrs),
        steer_tspan(icnf, mode),
        ps,
    )
end

@views function inference_prob(
    icnf::AbstractCondICNF{T, <:MatrixMode, INPLACE},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1, size(xs, 2))
    @ignore_derivatives fill!(zrs, zero(T))
    ϵ = randn(icnf.rng, T, icnf.nvars + n_aug_input, size(xs, 2))
    ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        ifelse(
            INPLACE,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ys, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
            end,
        ),
        vcat(xs, zrs),
        steer_tspan(icnf, mode),
        ps,
    )
end

@views function generate_prob(
    icnf::AbstractCondICNF{T, <:VectorMode, INPLACE},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    ϵ = randn(icnf.rng, T, icnf.nvars + n_aug_input)
    new_xs = oftype(ϵ, rand(icnf.rng, icnf.basedist))
    zrs = similar(new_xs, n_aug + 1)
    @ignore_derivatives fill!(zrs, zero(T))
    ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        ifelse(
            INPLACE,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ys, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
            end,
        ),
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode)),
        ps,
    )
end

@views function generate_prob(
    icnf::AbstractCondICNF{T, <:MatrixMode, INPLACE},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
    n::Int,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    ϵ = randn(icnf.rng, T, icnf.nvars + n_aug_input, n)
    new_xs = oftype(ϵ, rand(icnf.rng, icnf.basedist, n))
    zrs = similar(new_xs, n_aug + 1, n)
    @ignore_derivatives fill!(zrs, zero(T))
    ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        ifelse(
            INPLACE,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ys, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
            end,
        ),
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode)),
        ps,
    )
end

@inline function inference(
    icnf::AbstractCondICNF,
    mode::Mode,
    xs::AbstractVecOrMat{<:Real},
    ys::AbstractVecOrMat{<:Real},
    ps::Any,
    st::Any,
)
    inference_sol(icnf, mode, inference_prob(icnf, mode, xs, ys, ps, st))
end

@inline function generate(
    icnf::AbstractCondICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)
    generate_sol(icnf, mode, generate_prob(icnf, mode, ys, ps, st))
end

@inline function generate(
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
    n::Int,
)
    generate_sol(icnf, mode, generate_prob(icnf, mode, ys, ps, st, n))
end

@inline function loss(
    icnf::AbstractCondICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)
    -first(inference(icnf, mode, xs, ys, ps, st))
end

@inline function loss(
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
)
    -mean(first(inference(icnf, mode, xs, ys, ps, st)))
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ADVectorMode, false},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = AbstractDifferentiation.value_and_jacobian(
        icnf.differentiation_backend,
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z,
    )
    l̇ = -tr(only(J))
    vcat(ż, l̇)
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ADVectorMode, true},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = AbstractDifferentiation.value_and_jacobian(
        icnf.differentiation_backend,
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z,
    )
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -tr(only(J))
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ZygoteVectorMode, false},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = Zygote.withjacobian(let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    l̇ = -tr(only(J))
    vcat(ż, l̇)
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ZygoteVectorMode, true},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = Zygote.withjacobian(let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -tr(only(J))
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:MatrixMode, false},
    mode::TestMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = jacobian_batched(icnf, let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    l̇ = -transpose(tr.(eachslice(J; dims = 3)))
    vcat(ż, l̇)
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:MatrixMode, true},
    mode::TestMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = jacobian_batched(icnf, let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -(tr.(eachslice(J; dims = 3)))
    nothing
end

@inline function (icnf::AbstractCondICNF)(xs_ys::Any, ps::Any, st::Any)
    xs, ys = xs_ys
    first(inference(icnf, TrainMode(), xs, ys, ps, st))
end
