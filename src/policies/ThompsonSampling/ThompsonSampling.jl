mutable struct ThompsonSampling{T1,T2} <: BanditPolicy
    t::Int64
    A::Int64
    arm_models::Vector{T1}
    model_args::T2
    X::Matvec
    r::Vecvec
    burnin::Int64
end


function ThompsonSampling(
    A::Int64,
    dim::Int64,
    model_type::Type,
    model_args,
    burnin::Int64,
)
    t = 1
    arm_models = Vector{model_type}(undef, A)
    X = fill(Matrix{Float64}(undef, 0, dim), A)
    r = fill(Vector{Float64}(undef, 0), A)
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

function update!(
    pol::ThompsonSampling,
    x::Vector{Float64},
    a::Int64,
    r::Float64,
)
    append_X!(pol, a, x)
    append_r!(pol, a, r)
    # This is implemented custom for the conctrete type
    update_arm_model!(pol, x, a, r)
    inc_t!(pol, 1)
end