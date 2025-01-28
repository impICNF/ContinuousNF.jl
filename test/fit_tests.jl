Test.@testset "Fit Tests" begin
    mts = if GROUP == "RNODE"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.RNODE]
    elseif GROUP == "FFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.FFJORD]
    elseif GROUP == "Planar"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.Planar]
    elseif GROUP == "CondRNODE"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.CondRNODE]
    elseif GROUP == "CondFFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.CondFFJORD]
    elseif GROUP == "CondPlanar"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.CondPlanar]
    else
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[
            ContinuousNormalizingFlows.RNODE,
            ContinuousNormalizingFlows.FFJORD,
            ContinuousNormalizingFlows.Planar,
            ContinuousNormalizingFlows.CondRNODE,
            ContinuousNormalizingFlows.CondFFJORD,
            ContinuousNormalizingFlows.CondPlanar,
        ]
    end
    n_epochs_ = Int[2]
    ndata_ = Int[4]
    nvars_ = Int[2]
    aug_steers = Bool[false, true]
    inplaces = Bool[false, true]
    adtypes = ADTypes.AbstractADType[ADTypes.AutoZygote(),
    # ADTypes.AutoEnzyme(;
    #     mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    #     function_annotation = Enzyme.Const,
    # ),
    # ADTypes.AutoEnzyme(;
    #     mode = Enzyme.set_runtime_activity(Enzyme.Forward),
    #     function_annotation = Enzyme.Const,
    # ),
    ]
    compute_modes = ContinuousNormalizingFlows.ComputeMode[
        ContinuousNormalizingFlows.DIVecJacVectorMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIVecJacMatrixMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIVecJacVectorMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                function_annotation = Enzyme.Const,
            ),
        ),
        ContinuousNormalizingFlows.DIJacVecVectorMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                function_annotation = Enzyme.Const,
            ),
        ),
        ContinuousNormalizingFlows.DIVecJacMatrixMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                function_annotation = Enzyme.Const,
            ),
        ),
        ContinuousNormalizingFlows.DIJacVecMatrixMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                function_annotation = Enzyme.Const,
            ),
        ),
    ]
    data_types = Type{<:AbstractFloat}[Float32]
    resources = ComputationalResources.AbstractResource[ComputationalResources.CPU1()]

    Test.@testset "$resource | $data_type | $compute_mode | $adtype | inplace = $inplace | aug & steer = $aug_steer | nvars = $nvars | $mt" for resource in
                                                                                                                                                resources,
        data_type in data_types,
        compute_mode in compute_modes,
        adtype in adtypes,
        inplace in inplaces,
        aug_steer in aug_steers,
        nvars in nvars_,
        ndata in ndata_,
        n_epochs in n_epochs_,
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        r = convert.(data_type, rand(data_dist, nvars, ndata))
        r2 = convert.(data_type, rand(data_dist2, nvars, ndata))
        df = DataFrames.DataFrame(transpose(r), :auto)
        df2 = DataFrames.DataFrame(transpose(r2), :auto)

        nn = ifelse(
            mt <: Union{
                ContinuousNormalizingFlows.CondRNODE,
                ContinuousNormalizingFlows.CondFFJORD,
                ContinuousNormalizingFlows.CondPlanar,
            },
            ifelse(
                mt <: ContinuousNormalizingFlows.CondPlanar,
                ifelse(
                    aug_steer,
                    Lux.Chain(
                        ContinuousNormalizingFlows.PlanarLayer(
                            nvars * 2,
                            tanh;
                            n_cond = nvars,
                        ),
                    ),
                    Lux.Chain(
                        ContinuousNormalizingFlows.PlanarLayer(nvars, tanh; n_cond = nvars),
                    ),
                ),
                ifelse(
                    aug_steer,
                    Lux.Chain(Lux.Dense(nvars * 3 => nvars * 2, tanh)),
                    Lux.Chain(Lux.Dense(nvars * 2 => nvars, tanh)),
                ),
            ),
            ifelse(
                mt <: ContinuousNormalizingFlows.Planar,
                ifelse(
                    aug_steer,
                    Lux.Chain(ContinuousNormalizingFlows.PlanarLayer(nvars * 2, tanh)),
                    Lux.Chain(ContinuousNormalizingFlows.PlanarLayer(nvars, tanh)),
                ),
                ifelse(
                    aug_steer,
                    Lux.Chain(Lux.Dense(nvars * 2 => nvars * 2, tanh)),
                    Lux.Chain(Lux.Dense(nvars => nvars, tanh)),
                ),
            ),
        )
        icnf = ifelse(
            aug_steer,
            ContinuousNormalizingFlows.construct(
                mt,
                nn,
                nvars,
                nvars;
                data_type,
                compute_mode,
                inplace,
                resource,
                steer_rate = convert(data_type, 1.0e-1),
                λ₃ = convert(data_type, 1.0e-2),
            ),
            ContinuousNormalizingFlows.construct(
                mt,
                nn,
                nvars;
                data_type,
                compute_mode,
                inplace,
                resource,
            ),
        )
        if mt <: Union{
            ContinuousNormalizingFlows.CondRNODE,
            ContinuousNormalizingFlows.CondFFJORD,
            ContinuousNormalizingFlows.CondPlanar,
        }
            model = ContinuousNormalizingFlows.CondICNFModel(icnf; n_epochs, adtype)
            mach = MLJBase.machine(model, (df, df2))

            Test.@test !isnothing(MLJBase.fit!(mach))
            Test.@test !isnothing(MLJBase.transform(mach, (df, df2)))
            Test.@test !isnothing(MLJBase.fitted_params(mach))

            Test.@test !isnothing(
                ContinuousNormalizingFlows.CondICNFDist(
                    mach,
                    ContinuousNormalizingFlows.TrainMode(),
                    r2,
                ),
            )
            Test.@test !isnothing(
                ContinuousNormalizingFlows.CondICNFDist(
                    mach,
                    ContinuousNormalizingFlows.TestMode(),
                    r2,
                ),
            )
        else
            model = ContinuousNormalizingFlows.ICNFModel(icnf; n_epochs, adtype)
            mach = MLJBase.machine(model, df)

            Test.@test !isnothing(MLJBase.fit!(mach))
            Test.@test !isnothing(MLJBase.transform(mach, df))
            Test.@test !isnothing(MLJBase.fitted_params(mach))

            Test.@test !isnothing(
                ContinuousNormalizingFlows.ICNFDist(
                    mach,
                    ContinuousNormalizingFlows.TrainMode(),
                ),
            )
            Test.@test !isnothing(
                ContinuousNormalizingFlows.ICNFDist(
                    mach,
                    ContinuousNormalizingFlows.TestMode(),
                ),
            )
        end
    end
end
