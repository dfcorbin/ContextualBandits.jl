module ContextualBandits


using Distributions: Uniform, Normal


# Getters and setters - cdist
export get_bounds, get_dim

export UniformContext, gen_context, ContextDist
include("cdist.jl")


# Getters and setters - rdist
export get_A, get_SD, get_mfun

export GaussianArms, gen_reward, compute_regret, RewardDist
include("rdist.jl")


# Getters and setters - policy
export get_A

export BanditPolicy, RandPol, choose, update!
include("policy.jl")

export simulate, BanditResults
include("sim.jl")

end
