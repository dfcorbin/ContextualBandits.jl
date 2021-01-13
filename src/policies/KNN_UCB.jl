mutable struct NearestneighborsUCB <: BanditPolicy
    X::Matvec
    r::Vecvec
    t::Int64
    theta::Float64
end

get_A(pol::NearestneighborsUCB) = length(pol.X)
get_X(pol::NearestneighborsUCB, a::Int64) = pol.X[a]
get_theta(pol::NearestneighborsUCB) = pol.theta
get_t(pol::NearestneighborsUCB) = pol.t
get_r(pol::NearestneighborsUCB, a::Int64) = pol.r[a]

inc_t!(pol::NearestneighborsUCB, val) = begin pol.t += val end

function NearestneighborsUCB(A::Int64, dim::Int64, theta::Float64)
    X = [Matrix{Float64}(undef, 0, dim) for a in 1:A]
    r = [Vector{Float64}(undef, 0) for a in 1:A]
    return NearestneighborsUCB(X, r, 1, theta)
end


function sort_nearest(pol::NearestneighborsUCB, x::Vector{Float64}, a::Int64)
    dist = Euclidean()
    xvec = reshape(x, (1, :))
    X = get_X(pol, a)
    r = get_r(pol, a)
    pairwise_distances = pairwise(dist, X, xvec; dims=1)
    pairwise_distances_flat = reshape(pairwise_distances, (:))
    inds = sortperm(pairwise_distances_flat)
    return pairwise_distances_flat[inds], r[inds]
end


function compute_upper_bound(pol::NearestneighborsUCB, x::Vector{Float64}, a::Int64)
    t = get_t(pol::NearestneighborsUCB)
    distances, y_nearest = sort_nearest(pol, x, a)
    theta = get_theta(pol)
    uvals = Vector{Float64}(undef, length(y_nearest))
    for k in 1:length(y_nearest)
        radius = distances[k]
        uvals[k] = sqrt(theta * log(t) / k) + radius
    end
    min_uncertainty, min_k = findmin(uvals)
    # println("min k: ", min_k)
    # println("mean: ", mean(y_nearest[1:min_k]))
    # println("min U: ", min_uncertainty)
    return mean(y_nearest[1:min_k]) + min_uncertainty
end


function choose(pol::NearestneighborsUCB, x::Vector{Float64})
    t = get_t(pol::NearestneighborsUCB)
    A = get_A(pol)
    if t <= A
        return t
    end
    # Construct upper confidence bounds
    upper_bounds = Vector{Float64}(undef, A)
    for a in 1:A
        upper_bounds[a] = compute_upper_bound(pol, x, a)
    end
    return findmax(upper_bounds)[2]
end


function append_X!(pol::NearestneighborsUCB, a::Int64, val::Vector{Float64})
    val1 = reshape(val, (1, :))
    pol.X[a] = vcat(pol.X[a], val1)
end


function append_r!(pol::NearestneighborsUCB, a::Int64, val::Float64)
    push!(pol.r[a], val)
end


function update!(pol::NearestneighborsUCB, x::Vector{Float64}, a::Int64, r::Float64)
    append_X!(pol, a, x)
    append_r!(pol, a, r)
    inc_t!(pol, 1)
end