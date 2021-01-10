module ContextualBandits

using Distributions: Uniform

# Getters and setters
export get_bounds, get_dim

export UniformContext, gen_context
include("cdist.jl")

end
