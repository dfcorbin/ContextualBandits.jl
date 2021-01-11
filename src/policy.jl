abstract type BanditPolicy end
const Matvec = Vector{Matrix{Float64}}
const Vecvec = Vector{Vector{Float64}}


struct RandPol <: BanditPolicy
    A::Int64
end


get_A(pol::RandPol) = pol.A


function choose(pol::RandPol, x::Vector{Float64})
    A = get_A(pol)
    return rand(1:A)
end


function update!(pol::RandPol, x::Vector{Float64}, a::Int64, r::Float64)
    nothing
end


# Generic Thompson Sampling framework. User is responsible for implementing mean simulation and
# updates according to the needs of their policy. 


mutable struct ThompsonSampling{T1,T2} <: BanditPolicy
    t::Int64
    A::Int64
    arm_models::Vector{T1}
    model_args::T2
    X::Matvec
    r::Vecvec
    burnin::Int64
end


function ThompsonSampling(A::Int64, dim::Int64, model_type::Type, model_args, burnin::Int64)
    t = 1
    arm_models = Vector{model_type}(undef, A)
    X = [Matrix{Float64}(undef, 0, dim) for a = 1:A]
    r = [Vector{Float64}(undef, 0) for a = 1:A]
    return ThompsonSampling(t, A, arm_models, model_args, X, r, burnin)
end


get_t(pol::ThompsonSampling) = pol.t
get_A(pol::ThompsonSampling) = length(pol.arm_models)
get_arm_models(pol::ThompsonSampling) = pol.arm_models
get_arm_models(pol::ThompsonSampling, a::Int64) = pol.arm_models[a]
get_X(pol::ThompsonSampling, a::Int64) = pol.X[a]
get_y(pol::ThompsonSampling, a::Int64) = pol.r[a]
get_model_args(pol::ThompsonSampling) = pol.model_args
get_burnin(pol::ThompsonSampling) = pol.burnin


function inc_t!(pol::ThompsonSampling, val::Int64)
    pol.t += val
end

function set_arm_models!(pol::ThompsonSampling, a::Int64, val)
    pol.arm_models[a] = val
end

function append_X!(pol::ThompsonSampling, a::Int64, val::Vector{Float64})
    val1 = reshape(val, (1, :))
    pol.X[a] = vcat(pol.X[a], val1)
end

function append_r!(pol::ThompsonSampling, a::Int64, val::Float64)
    push!(pol.r[a], val)
end


function choose(pol::ThompsonSampling, x::Vector{Float64})
    t = get_t(pol)
    A = get_A(pol)
    burnin = get_burnin(pol)
    if t <= burnin
        return 1 + t % A
    else
        arm_models = get_arm_models(pol)
        msim = simulate_means(arm_models, x)
        return findmax(msim)[2]
    end
end


function update!(pol::ThompsonSampling, x::Vector{Float64}, a::Int64, r::Float64)
    append_X!(pol, a, x)
    append_r!(pol, a, r)
    update_arm_model!(pol, x, a, r)
    inc_t!(pol, 1)
end


# Recursively partitioned Thompson Sampling


struct RecursiveTSArgs
    bounds::Matrix{Float64}
    degmax::Int64
    maxparam::Int64
    priorgen::Function
    shape::Float64
    scale::Float64
    mindat::Union{Nothing,Int64}
    Kmax::Int64
    retrain::Vector{Int64}
end


get_bounds(args::RecursiveTSArgs) = args.bounds
get_degmax(args::RecursiveTSArgs) = args.degmax
get_maxparam(args::RecursiveTSArgs) = args.maxparam
get_priorgen(args::RecursiveTSArgs) = args.priorgen
get_shape(args::RecursiveTSArgs) = args.shape
get_scale(args::RecursiveTSArgs) = args.scale
get_mindat(args::RecursiveTSArgs) = args.mindat
get_Kmax(args::RecursiveTSArgs) = args.Kmax
get_retrain(args::RecursiveTSArgs) = args.retrain


function update_arm_model!(
    pol::ThompsonSampling{PartitionModel{PolyBLM},RecursiveTSArgs},
    x::Vector{Float64},
    a::Int64,
    r::Float64,
)
    t = get_t(pol)
    burnin = get_burnin(pol)
    args = get_model_args(pol)
    if (t == burnin) || (t in get_retrain(args))
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
        fit!(model, x, r)
    end
end


function init_arm_model(X::Matrix{Float64}, y::Vector{Float64}, args::RecursiveTSArgs)
    return auto_partition_polyblm(
        X,
        y,
        get_bounds(args::RecursiveTSArgs);
        degmax = get_degmax(args::RecursiveTSArgs),
        maxparam = get_maxparam(args::RecursiveTSArgs),
        priorgen = get_priorgen(args::RecursiveTSArgs),
        shape = get_shape(args::RecursiveTSArgs),
        scale = get_scale(args::RecursiveTSArgs),
        mindat = get_mindat(args::RecursiveTSArgs),
        Kmax = get_Kmax(args::RecursiveTSArgs),
    )
end


function simulate_arm_mean(mod::PartitionModel{PolyBLM}, x::Vector{Float64})
    k = which_subset(mod, x)
    local_mod = get_lm(mod, k)
    x1 = polymod_pcbmat(local_mod, x; intercept = true)
    shape = get_shapepost(mod)
    scale = get_scalepost(mod)
    varsim = rand(InverseGamma(shape, scale))
    covpost = get_covpost(local_mod)
    covar = Symmetric(varsim * covpost)
    coeffpost = get_coeffpost(local_mod)
    coeffsim = rand(MvNormal(coeffpost, covar))
    out::Float64 = x1' * coeffsim
    return out
end


function simulate_means(arm_models::Vector{PartitionModel{PolyBLM}}, x::Vector{Float64})
    A = length(arm_models)
    out = Vector{Float64}(undef, A)
    for a = 1:A
        out[a] = simulate_arm_mean(arm_models[a], x)
    end
    return out
end


function recursive_partition_TS(
    A::Int64,
    dim::Int64,
    bounds::Matrix{Float64},
    burnin::Int64,
    retrain::Vector{Int64};
    degmax::Int64 = 3,
    maxparam::Int64 = 200,
    priorgen::Function = identity_hyper,
    shape::Float64 = 0.001,
    scale::Float64 = 0.001,
    mindat::Union{Nothing,Int64} = nothing,
    Kmax::Int64 = 200,
)
    args = RecursiveTSArgs(
        bounds,
        degmax,
        maxparam,
        priorgen,
        shape,
        scale,
        mindat,
        Kmax,
        retrain,
    )
    return ThompsonSampling(A, dim, PartitionModel{PolyBLM}, args, burnin)
end


function recursive_partition_TS(
    A::Int64,
    dim::Int64,
    bounds::Vector{Float64},
    burnin::Int64,
    retrain::Vector{Int64};
    degmax::Int64 = 3,
    maxparam::Int64 = 200,
    priorgen::Function = identity_hyper,
    shape::Float64 = 0.001,
    scale::Float64 = 0.001,
    mindat::Union{Nothing,Int64} = nothing,
    Kmax::Int64 = 200,
)
    bounds1 = reshape(bounds, (1, :))
    bounds1 = repeat(bounds1, dim, 1)
    return recursive_partition_TS(
        A,
        dim,
        bounds,
        burnin,
        retrain;
        degmax = degmax,
        maxparam = maxparam,
        priorgen = priogen,
        shape = shape,
        scale = scale,
        mindat = mindat,
        Kmax = Kmax
    )
end
