abstract type BanditPolicy end

include("RandPol.jl")
include("ThompsonSampling/ThompsonSampling.jl")
include("ThompsonSampling/recursive_partition_TS.jl")
include("KNN_UCB.jl")
