using ContextualBandits
using Test, Random, Distributions

@testset "Context distribution" begin
    include("test_cdist.jl")
end

@testset "Reward distribution" begin
    include("test_rdist.jl")
end

@testset "Policies" begin
    include("policies/test_KNN.jl")
end