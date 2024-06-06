Test.@testset "Regression Tests" begin
    rng = StableRNGs.StableRNG(12345)
    nvars = 2^3
    naugs = nvars
    n_in = nvars + naugs
    n = 2^10
    nn = Chain(Dense(n_in => 3 * n_in, tanh), Dense(3 * n_in => n_in, tanh))

    icnf = ContinuousNormalizingFlows.construct(
        ContinuousNormalizingFlows.RNODE,
        nn,
        nvars,
        naugs;
        compute_mode = ContinuousNormalizingFlows.DIVecJacMatrixMode(ADTypes.AutoZygote()),
        tspan = (0.0f0, 13.0f0),
        steer_rate = 1.0f-1,
        λ₃ = 1.0f-2,
        rng,
    )
    ps, st = Lux.setup(icnf.rng, icnf)
    ps = ComponentArrays.ComponentArray(ps)

    data_dist = Distributions.Beta{Float32}(2.0f0, 4.0f0)
    r = rand(icnf.rng, data_dist, nvars, n)
    r = convert.(Float32, r)

    df = DataFrames.DataFrame(transpose(r), :auto)
    model = ICNFModel(icnf) # use defaults

    mach = machine(model, df)
    fit!(mach)

    d = ICNFDist(mach, TestMode()) # alternative way
    actual_pdf = pdf.(data_dist, r)
    estimated_pdf = pdf(d, r)

    mad_ = Distances.meanad(estimated_pdf, actual_pdf)
    msd_ = Distances.msd(estimated_pdf, actual_pdf)
    tv_dis = Distances.totalvariation(estimated_pdf, actual_pdf) / n

    Test.@test mad_ <= 1.0f-1
    Test.@test msd_ <= 1.0f-1
    Test.@test tv_dis <= 1.0f-1
end