struct UniformContext <: ContextDist
    bounds::Matrix{Float64}
    function UniformContext(bounds::Matrix{Float64})
        error = ArgumentError("Bounds must have exactly two columns.")
        size(bounds, 2) == 2 ? new(bounds) : throw(error)
    end
end


get_dim(cdist::UniformContext) = size(cdist.bounds, 1)
get_bounds(cdist::UniformContext) = cdist.bounds


function UniformContext(dim::Int64, bounds::Vector{Float64})
    if length(bounds) != 2
        throw(ArgumentError("Bounds must be of length 2."))
    end
    b1 = reshape(bounds, (1, 2))
    bmat = repeat(b1, dim, 1)
    return UniformContext(bmat)
end


function gen_context!(cdist::UniformContext, x::Vector{Float64})
    dim = get_dim(cdist)
    bounds = get_bounds(cdist)
    for d = 1:dim
        x[d] = rand(Uniform(bounds[d, 1], bounds[d, 2]))
    end
    return x
end


function gen_context(cdist::UniformContext)
    dim = get_dim(cdist)
    x = Vector{Float64}(undef, dim)
    gen_context!(cdist, x)
    return x
end
