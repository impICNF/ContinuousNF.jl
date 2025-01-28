@inline function jacobian_batched(
    icnf::AbstractICNF{T, <:DIVecJacMatrixMode},
    f::Lux.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
) where {T}
    y = f(xs)
    z = similar(xs)
    ChainRulesCore.@ignore_derivatives fill!(z, zero(T))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        ChainRulesCore.@ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] =
            only(DifferentiationInterface.pullback(f, icnf.compute_mode.adback, xs, (z,)))
        ChainRulesCore.@ignore_derivatives z[i, :] .= zero(T)
    end
    y, eachslice(copy(res); dims = 3)
end

@inline function jacobian_batched(
    icnf::AbstractICNF{T, <:DIJacVecMatrixMode},
    f::Lux.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
) where {T}
    y = f(xs)
    z = similar(xs)
    ChainRulesCore.@ignore_derivatives fill!(z, zero(T))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        ChainRulesCore.@ignore_derivatives z[i, :] .= one(T)
        res[:, i, :] = only(
            DifferentiationInterface.pushforward(f, icnf.compute_mode.adback, xs, (z,)),
        )
        ChainRulesCore.@ignore_derivatives z[i, :] .= zero(T)
    end
    y, eachslice(copy(res); dims = 3)
end

@inline function jacobian_batched(
    icnf::AbstractICNF{T, <:DIMatrixMode},
    f::Lux.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
) where {T}
    y, J = DifferentiationInterface.value_and_jacobian(f, icnf.compute_mode.adback, xs)
    y, split_jac(J, size(xs, 1))
end

@inline function split_jac(x::AbstractMatrix{<:Real}, sz::Integer)
    (
        x[i:j, i:j] for (i, j) in zip(
            firstindex(x, 1):sz:lastindex(x, 1),
            (firstindex(x, 1) + sz - 1):sz:lastindex(x, 1),
        )
    )
end
