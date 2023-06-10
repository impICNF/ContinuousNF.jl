export inference, generate, loss

for arr_tp in (:Array, :CuArray, :AbstractArray)
    @eval begin
        function inference(
            icnf::AbstractCondICNF{T, $arr_tp, <:VectorMode},
            mode::Mode,
            xs::$arr_tp{<:Real, 1},
            ys::$arr_tp{<:Real, 1},
            ps::Any,
            st::Any;
            differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
            rng::AbstractRNG = Random.default_rng(),
            sol_args::Tuple = icnf.sol_args,
            sol_kwargs::Dict = icnf.sol_kwargs,
        )::Tuple{Vararg{Real}} where {T <: AbstractFloat}
            n_aug = n_augment(icnf, mode)
            zrs::$arr_tp = zeros(T, n_aug + 1)
            f_aug = augmented_f(icnf, mode, ys, st; differentiation_backend, rng)
            func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
            prob = ODEProblem{false, SciMLBase.FullSpecialize}(
                func,
                vcat(xs, zrs),
                icnf.tspan,
                ps,
            )
            sol = solve(prob, sol_args...; sol_kwargs...)
            fsol = sol[:, end]
            z = fsol[1:(end - n_aug - 1)]
            Δlogp = fsol[(end - n_aug)]
            logp̂x = logpdf(icnf.basedist, z) - Δlogp
            iszero(n_aug) ? (logp̂x,) : (logp̂x, fsol[(end - n_aug + 1):end]...)
        end

        function inference(
            icnf::AbstractCondICNF{T, $arr_tp, <:MatrixMode},
            mode::Mode,
            xs::$arr_tp{<:Real, 2},
            ys::$arr_tp{<:Real, 2},
            ps::Any,
            st::Any;
            differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
            rng::AbstractRNG = Random.default_rng(),
            sol_args::Tuple = icnf.sol_args,
            sol_kwargs::Dict = icnf.sol_kwargs,
        )::Tuple{Vararg{AbstractArray{<:Real, 1}}} where {T <: AbstractFloat}
            n_aug = n_augment(icnf, mode)
            zrs::$arr_tp = zeros(T, n_aug + 1, size(xs, 2))
            f_aug =
                augmented_f(icnf, mode, ys, st, size(xs, 2); differentiation_backend, rng)
            func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
            prob = ODEProblem{false, SciMLBase.FullSpecialize}(
                func,
                vcat(xs, zrs),
                icnf.tspan,
                ps,
            )
            sol = solve(prob, sol_args...; sol_kwargs...)
            fsol = sol[:, :, end]
            z = fsol[1:(end - n_aug - 1), :]
            Δlogp = fsol[(end - n_aug), :]
            logp̂x = logpdf(icnf.basedist, z) - Δlogp
            iszero(n_aug) ? (logp̂x,) : (logp̂x, eachrow(fsol[(end - n_aug + 1):end, :])...)
        end

        function generate(
            icnf::AbstractCondICNF{T, $arr_tp, <:VectorMode},
            mode::Mode,
            ys::$arr_tp{<:Real, 1},
            ps::Any,
            st::Any;
            differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
            rng::AbstractRNG = Random.default_rng(),
            sol_args::Tuple = icnf.sol_args,
            sol_kwargs::Dict = icnf.sol_kwargs,
        )::$arr_tp{<:Real, 1} where {T <: AbstractFloat}
            n_aug = n_augment(icnf, mode)
            new_xs::$arr_tp = rand(rng, icnf.basedist)
            zrs::$arr_tp = zeros(T, n_aug + 1)
            f_aug = augmented_f(icnf, mode, ys, st; differentiation_backend, rng)
            func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
            prob = ODEProblem{false, SciMLBase.FullSpecialize}(
                func,
                vcat(new_xs, zrs),
                reverse(icnf.tspan),
                ps,
            )
            sol = solve(prob, sol_args...; sol_kwargs...)
            fsol = sol[:, end]
            z = fsol[1:(end - n_aug - 1)]
            z
        end

        function generate(
            icnf::AbstractCondICNF{T, $arr_tp, <:MatrixMode},
            mode::Mode,
            ys::$arr_tp{<:Real, 2},
            ps::Any,
            st::Any,
            n::Integer;
            differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
            rng::AbstractRNG = Random.default_rng(),
            sol_args::Tuple = icnf.sol_args,
            sol_kwargs::Dict = icnf.sol_kwargs,
        )::$arr_tp{<:Real, 2} where {T <: AbstractFloat}
            n_aug = n_augment(icnf, mode)
            new_xs::$arr_tp = rand(rng, icnf.basedist, n)
            zrs::$arr_tp = zeros(T, n_aug + 1, size(new_xs, 2))
            f_aug = augmented_f(
                icnf,
                mode,
                ys,
                st,
                size(new_xs, 2);
                differentiation_backend,
                rng,
            )
            func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
            prob = ODEProblem{false, SciMLBase.FullSpecialize}(
                func,
                vcat(new_xs, zrs),
                reverse(icnf.tspan),
                ps,
            )
            sol = solve(prob, sol_args...; sol_kwargs...)
            fsol = sol[:, :, end]
            z = fsol[1:(end - n_aug - 1), :]
            z
        end

        function loss(
            icnf::AbstractCondICNF{T, $arr_tp, <:VectorMode},
            xs::$arr_tp{<:Real, 1},
            ys::$arr_tp{<:Real, 1},
            ps::Any,
            st::Any;
            differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
            mode::Mode = TrainMode(),
            rng::AbstractRNG = Random.default_rng(),
            sol_args::Tuple = icnf.sol_args,
            sol_kwargs::Dict = icnf.sol_kwargs,
        )::Real where {T <: AbstractFloat}
            logp̂x, = inference(
                icnf,
                mode,
                xs,
                ys,
                ps,
                st;
                differentiation_backend,
                rng,
                sol_args,
                sol_kwargs,
            )
            -logp̂x
        end

        function loss(
            icnf::AbstractCondICNF{T, $arr_tp, <:MatrixMode},
            xs::$arr_tp{<:Real, 2},
            ys::$arr_tp{<:Real, 2},
            ps::Any,
            st::Any;
            differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
            mode::Mode = TrainMode(),
            rng::AbstractRNG = Random.default_rng(),
            sol_args::Tuple = icnf.sol_args,
            sol_kwargs::Dict = icnf.sol_kwargs,
        )::Real where {T <: AbstractFloat}
            logp̂x, = inference(
                icnf,
                mode,
                xs,
                ys,
                ps,
                st;
                differentiation_backend,
                rng,
                sol_args,
                sol_kwargs,
            )
            mean(-logp̂x)
        end
    end
end
