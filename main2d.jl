####################################################################
#           computes krylov complexity
#               by   Prabhakar
#                  8/JAN/2026
####################################################################
using Random
# using MPI
using Printf
using Dates
using JLD2
using Statistics
using LinearAlgebra


include("src/Basis.jl")
include("src/Hamiltonian.jl")
include("src/Lanczos.jl")
include("src/krylov_time_evolution.jl")

using .Basis
using .Hamiltonian
using .Lanczos
using .KrylovTimeEvolution

function disorder(rng, L::Int, W::Float64)
    return rand(rng,L) .* W .- W/2
end

function neel_state(L)
    state1 = 0
    for i in 1:L
        if isodd(i)
            state1 |= (1 << (L - i))
        end
    end
    return state1
end

function read_input_file(filename)
    values = Float64[]

    for line in eachline(filename)
        s=strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        val = split(s)[1]
        push!(values, parse(Float64, val))
    end

    return values
end

vals=read_input_file("input.dat")

L=Int(vals[1])
d=Int(vals[2])
t=vals[3]          # Jx,Jy
U =vals[4]          # Jz
m=Int(vals[5])
W=vals[6]            #disoredr stregth
seed=Int(vals[7])
init_flag =Int(vals[8])
Nup=(L*d)÷2              

println("L X d             =",L,",",d)
println("t                 =",t)
println("U                 =",U)
println("Lanczos vectors m =",m)
println(" W                =",W)
println("Seed              =",seed)
println("Nup               =",Nup)
println("Sz                =",(Nup-(L*d-Nup))/2.0)
println("initial state     =",init_flag)


rng=MersenneTwister(seed)
h=disorder(rng,L,W)


basis=generate_basis(L,d,Nup)
index=state_index(basis)
dim=length(basis)

println("dimension = ",dim)


state = basis[1]
occ = zeros(Int,L,d)      
state_bits_2d!(occ,state,L,d)

println(occ)

pairs = NN_2d(L,d;periodic=false)




H=build_hamiltonian(basis,index,pairs,t,U)



# applyH!(out, v) = apply_xxz!(out,v,basis,index,L,J,delta,h)

# if init_flag==194264
#     eigvals,kry_ham=lanczos(applyH!,dim;m=m,rng=rng,init=:random)
#     evolve_krylov(kry_ham;tmin=0.0,tmax=100.0,Nt=400,prefix="random",seed,L,J,delta)

# elseif init_flag==121212
#     psi00 = zeros(Float64,dim)
#     ino=index[neel_state(L)]
#     # println(ino)
#     psi00[ino]=1.0

#     eigvals,kry_ham=lanczos(applyH!,dim;m=m,rng=rng,init=:neel,v0=psi00)
#     evolve_krylov(kry_ham;tmin=0.0,tmax=100.0,Nt=400,prefix="neel",seed,L,J,delta)
# end

# println("Ground-state energy = ",minimum(eigvals))
# println(basis)                    
# state = basis[2]
# println(state_bits(682, L))








