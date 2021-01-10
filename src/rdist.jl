abstract type RewardDist end


struct GaussianArms <: RewardDist
    means::Vector{Function}
    SD::Vector{Float64}
    function GaussianArms(means, SD)
        error = DimensionMismatch("SD must be same length as means")
        if length(SD) != length(means)
            throw(error)
        end
        return new(means, SD)
    end
end


function GaussianArms(means::Vector{Function}, SD::Float64)
    A = length(means)
    SDv = fill(SD, A)
    return GaussianArms(means, SDv)
end


get_A(rdist::GaussianArms) = length(rdist.means)
get_SD(rdist::GaussianArms) = rdist.SD
get_SD(rdist::GaussianArms, a::Int64) = rdist.SD[a]
get_mfun(rdist::GaussianArms, a::Int64) = rdist.means[a]


function gen_reward(rdist::GaussianArms, x::Vector{Float64}, a::Int64)
    mfun = get_mfun(rdist, a)
    μ = mfun(x)
    SD = get_SD(rdist, a)
    dist = Normal(μ, SD)
    return rand(dist)
end


function compute_regret(rdist::GaussianArms, x::Vector{Float64}, a::Int64)
    A = get_A(rdist)
    mu = Vector{Float64}(undef, A)
    for i = 1:A
        mu[i] = get_mfun(rdist, i)(x)
    end
    return maximum(mu) - mu[a]
end
