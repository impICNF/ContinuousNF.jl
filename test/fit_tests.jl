@testset "Fit Tests" begin
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
    resources = ComputationalResources.AbstractResource[ComputationalResources.CPU1()]
    if CUDA.has_cuda_gpu() && USE_GPU
        push!(resources, ComputationalResources.CUDALibs())
    end
    data_types = Type{<:AbstractFloat}[Float32]
    cmodes = Type{<:ContinuousNormalizingFlows.ComputeMode}[
        ZygoteMatrixMode,
        SDVecJacMatrixMode,
        # SDJacVecMatrixMode,
    ]
    nvars_ = (1:2)
    adb_list = AbstractDifferentiation.AbstractBackend[
        AbstractDifferentiation.ZygoteBackend(),
        AbstractDifferentiation.ReverseDiffBackend(),
        AbstractDifferentiation.ForwardDiffBackend(),
    ]
    go_ads = ADTypes.AbstractADType[
        ADTypes.AutoZygote(),
        ADTypes.AutoReverseDiff(),
        ADTypes.AutoForwardDiff(),
    ]

    @testset "$resources | $data_type | $(typeof(adb_u).name.name) for internal | $go_ad for fitting | $nvars Vars | $mt" for resource in
                                                                                                                              resources,
        data_type in data_types,
        adb_u in adb_list,
        go_ad in go_ads,
        nvars in nvars_,
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        r = convert(at{data_type}, rand(data_dist, nvars, 1))
        df = DataFrames.DataFrame(transpose(r), :auto)
        if mt <: Planar
            nn = PlanarLayer(nvars, tanh)
        else
            nn = Lux.Dense(nvars => nvars, tanh)
        end
        icnf =
            construct(mt, nn, nvars; data_type, resource, differentiation_backend = adb_u)
        model = ICNFModel(icnf; n_epochs = 2, adtype = go_ad)
        mach = MLJBase.machine(model, df)
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, df))
        @test !isnothing(MLJBase.fitted_params(mach))

        @test !isnothing(ICNFDist(mach, TrainMode()))
        @test !isnothing(ICNFDist(mach, TestMode()))
    end
    @testset "$resource | $data_type | $cmode | $go_ad for fitting | $nvars Vars | $mt" for resource in
                                                                                            resources,
        data_type in data_types,
        cmode in cmodes,
        go_ad in go_ads,
        nvars in nvars_,
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        r = convert(at{data_type}, rand(data_dist, nvars, 1))
        df = DataFrames.DataFrame(transpose(r), :auto)
        if mt <: Planar
            nn = PlanarLayer(nvars, tanh)
        else
            nn = Lux.Dense(nvars => nvars, tanh)
        end
        icnf = construct(mt, nn, nvars; data_type, resource, compute_mode = cmode)
        model = ICNFModel(icnf; n_epochs = 2, adtype = go_ad)
        mach = MLJBase.machine(model, df)
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, df))
        @test !isnothing(MLJBase.fitted_params(mach))

        @test !isnothing(ICNFDist(mach, TrainMode()))
        @test !isnothing(ICNFDist(mach, TestMode()))
    end
    @testset "$at | $data_type | $(typeof(adb_u).name.name) for internal | $go_ad for fitting | $nvars Vars | $mt" for resource in
                                                                                                                       resources,
        data_type in data_types,
        adb_u in adb_list,
        go_ad in go_ads,
        nvars in nvars_,
        mt in cmts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        r = convert(at{data_type}, rand(data_dist, nvars, 1))
        r2 = convert(at{data_type}, rand(data_dist, nvars, 1))
        df = DataFrames.DataFrame(transpose(r), :auto)
        df2 = DataFrames.DataFrame(transpose(r2), :auto)
        if mt <: CondPlanar
            nn = PlanarLayer(nvars, tanh; cond = true)
        else
            nn = Lux.Dense(2 * nvars => nvars, tanh)
        end
        icnf =
            construct(mt, nn, nvars; data_type, resource, differentiation_backend = adb_u)
        model = CondICNFModel(icnf; n_epochs = 2, adtype = go_ad)
        mach = MLJBase.machine(model, (df, df2))
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        @test !isnothing(MLJBase.fitted_params(mach))

        @test !isnothing(CondICNFDist(mach, TrainMode(), r2))
        @test !isnothing(CondICNFDist(mach, TestMode(), r2))
    end
    @testset "$resources | $data_type | $cmode | $go_ad for fitting | $nvars Vars | $mt" for resource in
                                                                                             resources,
        data_type in data_types,
        cmode in cmodes,
        go_ad in go_ads,
        nvars in nvars_,
        mt in cmts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        r = convert(at{data_type}, rand(data_dist, nvars, 1))
        r2 = convert(at{data_type}, rand(data_dist, nvars, 1))
        df = DataFrames.DataFrame(transpose(r), :auto)
        df2 = DataFrames.DataFrame(transpose(r2), :auto)
        if mt <: CondPlanar
            nn = PlanarLayer(nvars, tanh; cond = true)
        else
            nn = Lux.Dense(2 * nvars => nvars, tanh)
        end
        icnf = construct(mt, nn, nvars; data_type, resources, compute_mode = cmode)
        model = CondICNFModel(icnf; n_epochs = 2, adtype = go_ad)
        mach = MLJBase.machine(model, (df, df2))
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        @test !isnothing(MLJBase.fitted_params(mach))

        @test !isnothing(CondICNFDist(mach, TrainMode(), r2))
        @test !isnothing(CondICNFDist(mach, TestMode(), r2))
    end
end
