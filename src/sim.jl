struct BanditResults
    cdist::ContextDist
    rdist::RewardDist
    pol::BanditPolicy
    context::Matrix{Float64}
    action::Vector{Int64}
    reward::Vector{Float64}
    total_regret::Vector{Float64}
end


function simulate(
    cdist::ContextDist,
    rdist::RewardDist,
    pol::BanditPolicy,
    T::Int64;
    verbose::Bool = false,
)
    pol1 = deepcopy(pol)
    dim = get_dim(cdist)
    context = Matrix{Float64}(undef, T, dim)
    action = Vector{Int64}(undef, T)
    reward = Vector{Float64}(undef, T)
    total_regret = zeros(Float64, T + 1)
    x = Vector{Float64}(undef, dim)
    for t = 1:T
        if verbose
            reg = total_regret[t]
            print("\rTime step: $t, regret: $reg")
        end
        gen_context!(cdist, x)
        a = choose(pol1, x)
        r = gen_reward(rdist, x, a)
        update!(pol1, x, a, r)
        total_regret[t+1] = total_regret[t] + compute_regret(rdist, x, a)
        context[t, :] = x
        action[t] = a
        reward[t] = r
    end
    return BanditResults(cdist, rdist, pol1, context, action, reward, total_regret[2:end])
end
