function test_UniformContext_errors()
    colerror = ArgumentError("Bounds must have exactly two columns.")
    faulty_bounds = rand(2, 3)
    @test_throws colerror UniformContext(faulty_bounds)
end


function test_UniformContext()
    test_UniformContext_errors()
    cdist = UniformContext(2, [0.0, 1.0])
    @test get_dim(cdist) == 2
    @test get_bounds(cdist) == repeat([0.0 1.0], 2, 1)
    Random.seed!(100)
    x = gen_context(cdist)
    Random.seed!(100)
    @test x == rand(Uniform(0.0, 1.0), 2)
end


function test_all_cdist()
    test_UniformContext()
end


test_all_cdist()
