abstract type BanditPolicy end


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
