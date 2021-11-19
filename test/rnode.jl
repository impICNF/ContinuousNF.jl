@testset "RNODE" begin
    @testset "$cr-$tp-$nvars-Var-Smoke-Test" for
            cr in [CPU1(), CUDALibs()],
            tp in [Float64, Float32, Float16],
            nvars in 1:3
        rnode = RNODE{tp}(Dense(nvars, nvars), nvars; acceleration=cr)
        ufd = copy(rnode.p)
        n = 8
        r = rand(tp, nvars, n)

        @test !isnothing(inference(rnode, TestMode(), r))
        @test !isnothing(inference(rnode, TrainMode(), r))

        @test !isnothing(generate(rnode, TestMode(), n))
        @test !isnothing(generate(rnode, TrainMode(), n))

        @test !isnothing(rnode(r))
        @test !isnothing(loss_f(rnode)(r))

        d = ICNFDistribution(; m=rnode)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(pdf(d, r))
        @test !isnothing(rand(d, n))

        df = DataFrame(r', :auto)
        model = ICNFModel(; m=rnode, n_epochs=8)
        mach = machine(model, df)
        fit!(mach)
        fd = MLJBase.fitted_params(mach).learned_parameters

        @test !isnothing(MLJBase.transform(mach, df))
        @test fd != ufd
    end
end
