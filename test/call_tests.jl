@testset "Call Tests" begin
    if GROUP == "RNODE"
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[RNODE]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[CondRNODE]
    elseif GROUP == "FFJORD"
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[FFJORD]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[CondFFJORD]
    elseif GROUP == "Planar"
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[Planar]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[CondPlanar]
    else
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[RNODE, FFJORD, Planar]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[
            CondRNODE,
            CondFFJORD,
            CondPlanar,
        ]
    end
    resources = ComputationalResources.AbstractResource[CPU1()]
    if CUDA.has_cuda_gpu() && USE_GPU
        push!(resources, ComputationalResources.CUDALibs())
    end
    data_types = Type{<:AbstractFloat}[Float32]
    cmodes = Type{<:ContinuousNormalizingFlows.ComputeMode}[
        ZygoteMatrixMode,
        SDVecJacMatrixMode,
        # SDJacVecMatrixMode,
    ]
    omodes = ContinuousNormalizingFlows.Mode[TrainMode(), TestMode()]
    nvars_ = (1:2)
    adb_list = AbstractDifferentiation.AbstractBackend[
        AbstractDifferentiation.ZygoteBackend(),
        AbstractDifferentiation.ReverseDiffBackend(),
        AbstractDifferentiation.ForwardDiffBackend(),
    ]
    rng = Random.default_rng()
    aug_steers = Bool[false, true]

    @testset "$resource | $data_type | $(typeof(adb_u).name.name) | $nvars Vars | $mt" for resource in
                                                                                           resources,
        data_type in data_types,
        adb_u in adb_list,
        aug_steer in aug_steers,
        nvars in nvars_,
        omode in omodes,
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        r = convert(at{data_type}, rand(data_dist, nvars))
        r_arr = convert(at{data_type}, hcat(r))

        if mt <: Planar
            nn = aug_steer ? PlanarLayer(nvars * 2, tanh) : PlanarLayer(nvars, tanh)
        else
            nn =
                aug_steer ? Lux.Dense(nvars * 2 => nvars * 2, tanh) :
                Lux.Dense(nvars => nvars, tanh)
        end
        icnf =
            aug_steer ?
            construct(
                mt,
                nn,
                nvars,
                nvars;
                data_type,
                resource,
                augmented = true,
                steer = true,
                steer_rate = convert(data_type, 0.1),
                differentiation_backend = adb_u,
            ) :
            construct(mt, nn, nvars; data_type, resource, differentiation_backend = adb_u)
        ps, st = Lux.setup(rng, icnf)
        ps = ComponentArrays.ComponentArray(map(at{data_type}, ps))

        @test !isnothing(ContinuousNormalizingFlows.n_augment(icnf, omode))
        @test !isnothing(ContinuousNormalizingFlows.augmented_f(icnf, omode, st))
        @test !isnothing(ContinuousNormalizingFlows.zeros_T_AT(resource, icnf, 1))
        @test !isnothing(ContinuousNormalizingFlows.rand_T_AT(resource, icnf, rng, 1))
        @test !isnothing(ContinuousNormalizingFlows.randn_T_AT(resource, icnf, rng, 1))
        @test !isnothing(ContinuousNormalizingFlows.inference_prob(icnf, omode, r, ps, st))
        @test !isnothing(ContinuousNormalizingFlows.generate_prob(icnf, omode, ps, st))

        @test !isnothing(inference(icnf, omode, r, ps, st))
        @test !isnothing(generate(icnf, omode, ps, st))

        @test !isnothing(loss(icnf, omode, r, ps, st))

        diff_loss(x) = loss(icnf, omode, r, x, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff_loss, ps),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, ps))
            @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, ps))
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, ps))
        end

        @test !isnothing(Zygote.gradient(diff_loss, ps))
        @test !isnothing(Zygote.jacobian(diff_loss, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss, ps))
        diff_loss2(x) = Zygote.checkpointed(diff_loss, x)
        @test !isnothing(Zygote.gradient(diff_loss2, ps))
        @test !isnothing(Zygote.jacobian(diff_loss2, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss2, ps))
        # @test !isnothing(Zygote.hessian(diff_loss2, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss2, ps))
        # diff_loss3(x) = Zygote.forwarddiff(diff_loss, x)
        # @test !isnothing(Zygote.gradient(diff_loss3, ps))
        # @test !isnothing(Zygote.jacobian(diff_loss3, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss3, ps))
        # @test !isnothing(Zygote.hessian(diff_loss3, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss3, ps))
        # diff_loss4(x) = Zygote.forwarddiff(diff_loss2, x)
        # @test !isnothing(Zygote.gradient(diff_loss4, ps))
        # @test !isnothing(Zygote.jacobian(diff_loss4, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss4, ps))
        # @test !isnothing(Zygote.hessian(diff_loss4, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss4, ps))

        @test !isnothing(ReverseDiff.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ReverseDiff.hessian(diff_loss, ps))

        @test !isnothing(ForwardDiff.gradient(diff_loss, ps))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ForwardDiff.hessian(diff_loss, ps))

        diff2_loss(x) = loss(icnf, omode, x, ps, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff2_loss, r),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff2_loss, r))
            @test !isnothing(AbstractDifferentiation.jacobian(adb, diff2_loss, r))
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff2_loss, r))
        end

        @test !isnothing(Zygote.gradient(diff2_loss, r))
        @test !isnothing(Zygote.jacobian(diff2_loss, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss, r))
        # @test !isnothing(Zygote.hessian(diff2_loss, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss, r))
        diff2_loss2(x) = Zygote.checkpointed(diff2_loss, x)
        @test !isnothing(Zygote.gradient(diff2_loss2, r))
        @test !isnothing(Zygote.jacobian(diff2_loss2, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss2, r))
        # @test !isnothing(Zygote.hessian(diff2_loss2, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss2, r))
        diff2_loss3(x) = Zygote.forwarddiff(diff2_loss, x)
        @test !isnothing(Zygote.gradient(diff2_loss3, r))
        @test !isnothing(Zygote.jacobian(diff2_loss3, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss3, r))
        # @test !isnothing(Zygote.hessian(diff2_loss3, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss3, r))
        diff2_loss4(x) = Zygote.forwarddiff(diff2_loss2, x)
        @test !isnothing(Zygote.gradient(diff2_loss4, r))
        @test !isnothing(Zygote.jacobian(diff2_loss4, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss4, r))
        # @test !isnothing(Zygote.hessian(diff2_loss4, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss4, r))

        @test !isnothing(ReverseDiff.gradient(diff2_loss, r))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff2_loss, r))
        # @test !isnothing(ReverseDiff.hessian(diff2_loss, r))

        @test !isnothing(ForwardDiff.gradient(diff2_loss, r))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff2_loss, r))
        # @test !isnothing(ForwardDiff.hessian(diff2_loss, r))

        d = ICNFDist(icnf, omode, ps, st)

        @test !isnothing(Distributions.logpdf(d, r))
        @test !isnothing(Distributions.logpdf(d, r_arr))
        @test !isnothing(Distributions.pdf(d, r))
        @test !isnothing(Distributions.pdf(d, r_arr))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, 1))
    end
    @testset "$resource | $data_type | $cmode | $nvars Vars | $mt" for resource in
                                                                       resources,
        data_type in data_types,
        cmode in cmodes,
        aug_steer in aug_steers,
        nvars in nvars_,
        omode in omodes,
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        r = convert(at{data_type}, rand(data_dist, nvars))
        r_arr = convert(at{data_type}, hcat(r))

        if mt <: Planar
            nn = aug_steer ? PlanarLayer(nvars * 2, tanh) : PlanarLayer(nvars, tanh)
        else
            nn =
                aug_steer ? Lux.Dense(nvars * 2 => nvars * 2, tanh) :
                Lux.Dense(nvars => nvars, tanh)
        end
        icnf =
            aug_steer ?
            construct(
                mt,
                nn,
                nvars,
                nvars;
                data_type,
                resource,
                augmented = true,
                steer = true,
                steer_rate = convert(data_type, 0.1),
                compute_mode = cmode,
            ) : construct(mt, nn, nvars; data_type, resource, compute_mode = cmode)
        ps, st = Lux.setup(rng, icnf)
        ps = ComponentArrays.ComponentArray(map(at{data_type}, ps))

        @test !isnothing(ContinuousNormalizingFlows.n_augment(icnf, omode))
        @test !isnothing(
            ContinuousNormalizingFlows.augmented_f(icnf, omode, st, size(r_arr, 2)),
        )
        @test !isnothing(ContinuousNormalizingFlows.zeros_T_AT(resource, icnf, 1))
        @test !isnothing(ContinuousNormalizingFlows.rand_T_AT(resource, icnf, rng, 1))
        @test !isnothing(ContinuousNormalizingFlows.randn_T_AT(resource, icnf, rng, 1))
        @test !isnothing(
            ContinuousNormalizingFlows.inference_prob(icnf, omode, r_arr, ps, st),
        )
        @test !isnothing(ContinuousNormalizingFlows.generate_prob(icnf, omode, ps, st, 1))

        @test !isnothing(inference(icnf, omode, r_arr, ps, st))
        @test !isnothing(generate(icnf, omode, ps, st, 1))

        @test !isnothing(loss(icnf, omode, r_arr, ps, st))

        diff_loss(x) = loss(icnf, omode, r_arr, x, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff_loss, ps),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, ps))
            @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, ps))
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, ps))
        end

        @test !isnothing(Zygote.gradient(diff_loss, ps))
        @test !isnothing(Zygote.jacobian(diff_loss, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss, ps))
        diff_loss2(x) = Zygote.checkpointed(diff_loss, x)
        @test !isnothing(Zygote.gradient(diff_loss2, ps))
        @test !isnothing(Zygote.jacobian(diff_loss2, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss2, ps))
        # @test !isnothing(Zygote.hessian(diff_loss2, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss2, ps))
        # diff_loss3(x) = Zygote.forwarddiff(diff_loss, x)
        # @test !isnothing(Zygote.gradient(diff_loss3, ps))
        # @test !isnothing(Zygote.jacobian(diff_loss3, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss3, ps))
        # @test !isnothing(Zygote.hessian(diff_loss3, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss3, ps))
        # diff_loss4(x) = Zygote.forwarddiff(diff_loss2, x)
        # @test !isnothing(Zygote.gradient(diff_loss4, ps))
        # @test !isnothing(Zygote.jacobian(diff_loss4, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss4, ps))
        # @test !isnothing(Zygote.hessian(diff_loss4, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss4, ps))

        @test !isnothing(ReverseDiff.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ReverseDiff.hessian(diff_loss, ps))

        @test !isnothing(ForwardDiff.gradient(diff_loss, ps))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ForwardDiff.hessian(diff_loss, ps))

        diff2_loss(x) = loss(icnf, omode, hcat(x), ps, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff2_loss, r),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff2_loss, r))
            @test !isnothing(AbstractDifferentiation.jacobian(adb, diff2_loss, r))
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff2_loss, r))
        end

        @test !isnothing(Zygote.gradient(diff2_loss, r))
        @test !isnothing(Zygote.jacobian(diff2_loss, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss, r))
        # @test !isnothing(Zygote.hessian(diff2_loss, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss, r))
        diff2_loss2(x) = Zygote.checkpointed(diff2_loss, x)
        @test !isnothing(Zygote.gradient(diff2_loss2, r))
        @test !isnothing(Zygote.jacobian(diff2_loss2, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss2, r))
        # @test !isnothing(Zygote.hessian(diff2_loss2, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss2, r))
        diff2_loss3(x) = Zygote.forwarddiff(diff2_loss, x)
        @test !isnothing(Zygote.gradient(diff2_loss3, r))
        @test !isnothing(Zygote.jacobian(diff2_loss3, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss3, r))
        # @test !isnothing(Zygote.hessian(diff2_loss3, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss3, r))
        diff2_loss4(x) = Zygote.forwarddiff(diff2_loss2, x)
        @test !isnothing(Zygote.gradient(diff2_loss4, r))
        @test !isnothing(Zygote.jacobian(diff2_loss4, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss4, r))
        # @test !isnothing(Zygote.hessian(diff2_loss4, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss4, r))

        @test !isnothing(ReverseDiff.gradient(diff2_loss, r))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff2_loss, r))
        # @test !isnothing(ReverseDiff.hessian(diff2_loss, r))

        @test !isnothing(ForwardDiff.gradient(diff2_loss, r))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff2_loss, r))
        # @test !isnothing(ForwardDiff.hessian(diff2_loss, r))

        d = ICNFDist(icnf, omode, ps, st)

        @test !isnothing(Distributions.logpdf(d, r))
        @test !isnothing(Distributions.logpdf(d, r_arr))
        @test !isnothing(Distributions.pdf(d, r))
        @test !isnothing(Distributions.pdf(d, r_arr))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, 1))
    end
    @testset "$resource | $data_type | $(typeof(adb_u).name.name) | $nvars Vars | $mt" for resource in
                                                                                           resources,
        data_type in data_types,
        adb_u in adb_list,
        aug_steer in aug_steers,
        nvars in nvars_,
        omode in omodes,
        mt in cmts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        r = convert(at{data_type}, rand(data_dist, nvars))
        r_arr = convert(at{data_type}, hcat(r))
        r2 = convert(at{data_type}, rand(data_dist, nvars))
        r2_arr = convert(at{data_type}, hcat(r2))

        if mt <: CondPlanar
            nn =
                aug_steer ? PlanarLayer(nvars * 2, tanh; cond = true, n_cond = nvars) :
                PlanarLayer(nvars, tanh; cond = true, n_cond = nvars)
        else
            nn =
                aug_steer ? Lux.Dense(nvars * 3 => nvars * 2, tanh) :
                Lux.Dense(nvars * 2 => nvars, tanh)
        end
        icnf =
            aug_steer ?
            construct(
                mt,
                nn,
                nvars,
                nvars;
                data_type,
                resource,
                augmented = true,
                steer = true,
                steer_rate = convert(data_type, 0.1),
                differentiation_backend = adb_u,
            ) :
            construct(mt, nn, nvars; data_type, resource, differentiation_backend = adb_u)
        ps, st = Lux.setup(rng, icnf)
        ps = ComponentArrays.ComponentArray(map(at{data_type}, ps))

        @test !isnothing(ContinuousNormalizingFlows.n_augment(icnf, omode))
        @test !isnothing(ContinuousNormalizingFlows.augmented_f(icnf, omode, r2, st))
        @test !isnothing(ContinuousNormalizingFlows.zeros_T_AT(resource, icnf, 1))
        @test !isnothing(ContinuousNormalizingFlows.rand_T_AT(resource, icnf, rng, 1))
        @test !isnothing(ContinuousNormalizingFlows.randn_T_AT(resource, icnf, rng, 1))
        @test !isnothing(
            ContinuousNormalizingFlows.inference_prob(icnf, omode, r, r2, ps, st),
        )
        @test !isnothing(ContinuousNormalizingFlows.generate_prob(icnf, omode, r2, ps, st))

        @test !isnothing(inference(icnf, omode, r, r2, ps, st))
        @test !isnothing(generate(icnf, omode, r2, ps, st))

        @test !isnothing(loss(icnf, omode, r, r2, ps, st))

        diff_loss(x) = loss(icnf, omode, r, r2, x, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff_loss, ps),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, ps))
            @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, ps))
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, ps))
        end

        @test !isnothing(Zygote.gradient(diff_loss, ps))
        @test !isnothing(Zygote.jacobian(diff_loss, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss, ps))
        diff_loss2(x) = Zygote.checkpointed(diff_loss, x)
        @test !isnothing(Zygote.gradient(diff_loss2, ps))
        @test !isnothing(Zygote.jacobian(diff_loss2, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss2, ps))
        # @test !isnothing(Zygote.hessian(diff_loss2, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss2, ps))
        # diff_loss3(x) = Zygote.forwarddiff(diff_loss, x)
        # @test !isnothing(Zygote.gradient(diff_loss3, ps))
        # @test !isnothing(Zygote.jacobian(diff_loss3, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss3, ps))
        # @test !isnothing(Zygote.hessian(diff_loss3, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss3, ps))
        # diff_loss4(x) = Zygote.forwarddiff(diff_loss2, x)
        # @test !isnothing(Zygote.gradient(diff_loss4, ps))
        # @test !isnothing(Zygote.jacobian(diff_loss4, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss4, ps))
        # @test !isnothing(Zygote.hessian(diff_loss4, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss4, ps))

        @test !isnothing(ReverseDiff.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ReverseDiff.hessian(diff_loss, ps))

        @test !isnothing(ForwardDiff.gradient(diff_loss, ps))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ForwardDiff.hessian(diff_loss, ps))

        diff2_loss(x) = loss(icnf, omode, x, r2, ps, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff2_loss, r),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff2_loss, r))
            @test !isnothing(AbstractDifferentiation.jacobian(adb, diff2_loss, r))
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff2_loss, r))
        end

        @test !isnothing(Zygote.gradient(diff2_loss, r))
        @test !isnothing(Zygote.jacobian(diff2_loss, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss, r))
        # @test !isnothing(Zygote.hessian(diff2_loss, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss, r))
        diff2_loss2(x) = Zygote.checkpointed(diff2_loss, x)
        @test !isnothing(Zygote.gradient(diff2_loss2, r))
        @test !isnothing(Zygote.jacobian(diff2_loss2, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss2, r))
        # @test !isnothing(Zygote.hessian(diff2_loss2, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss2, r))
        diff2_loss3(x) = Zygote.forwarddiff(diff2_loss, x)
        @test !isnothing(Zygote.gradient(diff2_loss3, r))
        @test !isnothing(Zygote.jacobian(diff2_loss3, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss3, r))
        # @test !isnothing(Zygote.hessian(diff2_loss3, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss3, r))
        diff2_loss4(x) = Zygote.forwarddiff(diff2_loss2, x)
        @test !isnothing(Zygote.gradient(diff2_loss4, r))
        @test !isnothing(Zygote.jacobian(diff2_loss4, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss4, r))
        # @test !isnothing(Zygote.hessian(diff2_loss4, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss4, r))

        @test !isnothing(ReverseDiff.gradient(diff2_loss, r))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff2_loss, r))
        # @test !isnothing(ReverseDiff.hessian(diff2_loss, r))

        @test !isnothing(ForwardDiff.gradient(diff2_loss, r))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff2_loss, r))
        # @test !isnothing(ForwardDiff.hessian(diff2_loss, r))

        d = CondICNFDist(icnf, omode, r2, ps, st)

        @test !isnothing(Distributions.logpdf(d, r))
        @test !isnothing(Distributions.logpdf(d, r_arr))
        @test !isnothing(Distributions.pdf(d, r))
        @test !isnothing(Distributions.pdf(d, r_arr))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, 1))
    end
    @testset "$resource | $data_type | $cmode | $nvars Vars | $mt" for resource in
                                                                       resources,
        data_type in data_types,
        cmode in cmodes,
        aug_steer in aug_steers,
        nvars in nvars_,
        omode in omodes,
        mt in cmts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        r = convert(at{data_type}, rand(data_dist, nvars))
        r_arr = convert(at{data_type}, hcat(r))
        r2 = convert(at{data_type}, rand(data_dist, nvars))
        r2_arr = convert(at{data_type}, hcat(r2))

        if mt <: CondPlanar
            nn =
                aug_steer ? PlanarLayer(nvars * 2, tanh; cond = true, n_cond = nvars) :
                PlanarLayer(nvars, tanh; cond = true, n_cond = nvars)
        else
            nn =
                aug_steer ? Lux.Dense(nvars * 3 => nvars * 2, tanh) :
                Lux.Dense(nvars * 2 => nvars, tanh)
        end
        icnf =
            aug_steer ?
            construct(
                mt,
                nn,
                nvars,
                nvars;
                data_type,
                resource,
                augmented = true,
                steer = true,
                steer_rate = convert(data_type, 0.1),
                compute_mode = cmode,
            ) : construct(mt, nn, nvars; data_type, resource, compute_mode = cmode)
        ps, st = Lux.setup(rng, icnf)
        ps = ComponentArrays.ComponentArray(map(at{data_type}, ps))

        @test !isnothing(ContinuousNormalizingFlows.n_augment(icnf, omode))
        @test !isnothing(
            ContinuousNormalizingFlows.augmented_f(icnf, omode, r2_arr, st, size(r_arr, 2)),
        )
        @test !isnothing(ContinuousNormalizingFlows.zeros_T_AT(resource, icnf, 1))
        @test !isnothing(ContinuousNormalizingFlows.rand_T_AT(resource, icnf, rng, 1))
        @test !isnothing(ContinuousNormalizingFlows.randn_T_AT(resource, icnf, rng, 1))
        @test !isnothing(
            ContinuousNormalizingFlows.inference_prob(icnf, omode, r_arr, r2_arr, ps, st),
        )
        @test !isnothing(
            ContinuousNormalizingFlows.generate_prob(icnf, omode, r2_arr, ps, st, 1),
        )

        @test !isnothing(inference(icnf, omode, r_arr, r2_arr, ps, st))
        @test !isnothing(generate(icnf, omode, r2_arr, ps, st, 1))

        @test !isnothing(loss(icnf, omode, r_arr, r2_arr, ps, st))

        diff_loss(x) = loss(icnf, omode, r_arr, r2_arr, x, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff_loss, ps),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, ps))
            @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, ps))
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, ps))
        end

        @test !isnothing(Zygote.gradient(diff_loss, ps))
        @test !isnothing(Zygote.jacobian(diff_loss, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss, ps))
        diff_loss2(x) = Zygote.checkpointed(diff_loss, x)
        @test !isnothing(Zygote.gradient(diff_loss2, ps))
        @test !isnothing(Zygote.jacobian(diff_loss2, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss2, ps))
        # @test !isnothing(Zygote.hessian(diff_loss2, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss2, ps))
        # diff_loss3(x) = Zygote.forwarddiff(diff_loss, x)
        # @test !isnothing(Zygote.gradient(diff_loss3, ps))
        # @test !isnothing(Zygote.jacobian(diff_loss3, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss3, ps))
        # @test !isnothing(Zygote.hessian(diff_loss3, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss3, ps))
        # diff_loss4(x) = Zygote.forwarddiff(diff_loss2, x)
        # @test !isnothing(Zygote.gradient(diff_loss4, ps))
        # @test !isnothing(Zygote.jacobian(diff_loss4, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss4, ps))
        # @test !isnothing(Zygote.hessian(diff_loss4, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss4, ps))

        @test !isnothing(ReverseDiff.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ReverseDiff.hessian(diff_loss, ps))

        @test !isnothing(ForwardDiff.gradient(diff_loss, ps))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ForwardDiff.hessian(diff_loss, ps))

        diff2_loss(x) = loss(icnf, omode, hcat(x), r2_arr, ps, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff2_loss, r),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff2_loss, r))
            @test !isnothing(AbstractDifferentiation.jacobian(adb, diff2_loss, r))
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff2_loss, r))
        end

        @test !isnothing(Zygote.gradient(diff2_loss, r))
        @test !isnothing(Zygote.jacobian(diff2_loss, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss, r))
        # @test !isnothing(Zygote.hessian(diff2_loss, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss, r))
        diff2_loss2(x) = Zygote.checkpointed(diff2_loss, x)
        @test !isnothing(Zygote.gradient(diff2_loss2, r))
        @test !isnothing(Zygote.jacobian(diff2_loss2, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss2, r))
        # @test !isnothing(Zygote.hessian(diff2_loss2, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss2, r))
        diff2_loss3(x) = Zygote.forwarddiff(diff2_loss, x)
        @test !isnothing(Zygote.gradient(diff2_loss3, r))
        @test !isnothing(Zygote.jacobian(diff2_loss3, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss3, r))
        # @test !isnothing(Zygote.hessian(diff2_loss3, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss3, r))
        diff2_loss4(x) = Zygote.forwarddiff(diff2_loss2, x)
        @test !isnothing(Zygote.gradient(diff2_loss4, r))
        @test !isnothing(Zygote.jacobian(diff2_loss4, r))
        # @test !isnothing(Zygote.diaghessian(diff2_loss4, r))
        # @test !isnothing(Zygote.hessian(diff2_loss4, r))
        # @test !isnothing(Zygote.hessian_reverse(diff2_loss4, r))

        @test !isnothing(ReverseDiff.gradient(diff2_loss, r))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff2_loss, r))
        # @test !isnothing(ReverseDiff.hessian(diff2_loss, r))

        @test !isnothing(ForwardDiff.gradient(diff2_loss, r))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff2_loss, r))
        # @test !isnothing(ForwardDiff.hessian(diff2_loss, r))

        d = CondICNFDist(icnf, omode, r2_arr, ps, st)

        @test !isnothing(Distributions.logpdf(d, r))
        @test !isnothing(Distributions.logpdf(d, r_arr))
        @test !isnothing(Distributions.pdf(d, r))
        @test !isnothing(Distributions.pdf(d, r_arr))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, 1))
    end
end
