module ContextualBandits

using RecursivePartition:
    auto_partition_polyblm,
    identity_hyper,
    PartitionModel,
    PolyBLM,
    auto_partition_polyblm,
    which_subset,
    get_lm,
    polymod_pcbmat,
    get_shapepost,
    get_scalepost,
    get_covpost,
    get_coeffpost,
    fit!
using Distributions: Uniform, Normal, InverseGamma, MvNormal
using LinearAlgebra: Symmetric

include("utils.jl")

# Getters and setters - cdist
export get_bounds, get_dim

export UniformContext, gen_context, ContextDist
include("contexts/contexts.jl")


# Getters and setters - rdist
export get_A, get_SD, get_mfun

export GaussianArms, gen_reward, compute_regret, RewardDist
include("rewards/rewards.jl")


# Getters and setters - policy
export get_A

export BanditPolicy, RandPol, choose, update!, RecursiveTSArgs, recursive_partition_TS
include("policies/policies.jl")

export simulate, BanditResults
include("sim.jl")


end
