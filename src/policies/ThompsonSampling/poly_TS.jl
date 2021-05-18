struct PolyTSArgs
    degmax::Int64
    bounds::Matrix{Float64}
    maxparam::Int64
    shape::Float64
    scale::Float64
    priorgen::Function
    retrain::Vector{Int64}
end

get_degmax(args::PolyTSArgs) = args.degmax
get_bounds(args::PolyTSArgs) = args.bounds
get_maxparam(args::PolyTSArgs) = args.maxparam
get_shape(args::PolyTSArgs) = args.shape
get_scale(args::PolyTSArgs) = args.scale
get_priorgen(args::PolyTSArgs) = args.priorgen
get_retrain(args::PolyTSArgs) = args.retrain

function init_arm_model(
    X::Matrix{Float64},
    y::Vector{Float64},
    args::PolyTSArgs
)
    return PolyBLM(
        X,
        y,
        get_degmax(args),
        get_bounds(args);
        maxparam=get_maxparam(args),
        shape=get_shape(args),
        scale=get_scale(args),
        priorgen=get_priorgen(args)
    )
end

function update_arm_model!(
    pol::ThompsonSampling{PolyBLM, PolyTSArgs},
    x::Vector{Float64},
    a::Int64,
    r::Float64
)
    t = get_t(pol)
    burnin = get_burnin(pol)
    args = get_model_args(pol)
    if t == burnin || (t in get_retrain(args))
        A = get_A(pol)
        for arm = 1:A
            X = get_X(pol, arm)
            y = get_y(pol, arm)
            args = get_model_args(pol)
            new_model = init_arm_model(X, y, args)
            set_arm_models!(pol, arm, new_model)
        end
    elseif t > burnin
        model = get_arm_models(pol, a)
        x1 = reshape(x, (1, :))
        fit!(model, x1, [r])
    end
end

function simulate_arm_mean(mod::PolyBLM, x::Vector{Float64})
    x1 = polymod_pcbmat(mod, x; intercept=true)
    shape = get_shapepost(mod)
    scale = get_scalepost(mod)
    varsim = rand(InverseGamma(shape, scale))
    covpost = get_covpost(mod)
    covar = Symmetric(varsim * covpost)
    coeffpost = get_coeffpost(mod)
    coeffsim = rand(MvNormal(coeffpost, covar))
    out::Float64 = x1' * coeffsim
    return out
end

function simulate_means(
    arm_models::Vector{PolyBLM},
    x::Vector{Float64}
)
    A = length(arm_models)
    out = Vector{Float64}(undef, A)
    for a = 1:A
        out[a] = simulate_arm_mean(arm_models[a], x)
    end
    return out
end

function poly_TS(
    A::Int64,
    dim::Int64,
    degmax::Int64,
    bounds::Matrix{Float64},
    retrain::Vector{Int64},
    burnin::Int64;
    maxparam::Int64=200,
    shape::Float64=0.001,
    scale::Float64=0.001,
    priorgen::Function=identity_hyper
)
    args = PolyTSArgs(degmax, bounds, maxparam, shape, scale, priorgen, retrain)
    return ThompsonSampling(A, dim, PolyBLM, args, burnin)
end

function poly_TS(
    A::Int64,
    dim::Int64,
    degmax::Int64,
    bounds::Vector{Float64},
    retrain::Vector{Int64},
    burnin::Int64;
    maxparam::Int64 = 200,
    shape::Float64 = 0.001,
    scale::Float64 = 0.001,
    priorgen::Function = identity_hyper
)
    bounds1 = reshape(bounds, (1, :))
    bounds1 = repeat(bounds1, dim, 1)
    return poly_TS(
        A,
        dim,
        degmax,
        bounds1,
        retrain,
        burnin;
        maxparam = maxparam,
        shape = shape,
        scale = scale,
        priorgen = priorgen,
    )
end