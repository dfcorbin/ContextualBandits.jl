function test_Gaussian_errors(means::Vector{Function})
    SD = [1.0]
    SD_error = DimensionMismatch("SD must be same length as means")
    @test_throws SD_error GaussianArms(means, SD)
end


function test_Gaussian_rdist()
    means = [x -> x[1], x -> 2 * x[1]]
    test_Gaussian_errors(means)
    rdist = GaussianArms(means, 1.0)
    @test get_A(rdist) == 2
    @test get_SD(rdist) == [1.0, 1.0]
    @test get_SD(rdist, 1) == 1.0
    @test get_mfun(rdist, 1)(0.0) == 0.0
    Random.seed!(100)
    x = [1.0]
    r = gen_reward(rdist, x, 1)
    Random.seed!(100)
    @test r == rand(Normal(1.0, 1.0))
    @test compute_regret(rdist, [1.0], 1) == 1.0
end


function test_all_rdist()
    test_Gaussian_rdist()
end

test_all_rdist()
