####################################################################
#             written  by Prabhakar
#               IIT Bombay   
#           
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
include("src/HtimesV.jl")
include("src/krylov_time_evolution.jl")
include("src/imbalance.jl")

using .Basis
using .Hamiltonian
using .Lanczos
using .KrylovTimeEvolution
using .HXV
using .imbalance

function disorder(rng, L::Int, W::Float64)
    return rand(rng,L) .* W .- W/2
end

function neel_state(Lx, Ly)
    state=0
for y in 1:Ly
    for x in 1:Lx
        if isodd(x + y)
            site = (y - 1) * Lx + x
            state |= 1 << (site - 1)
           
        end
    end
end
    return state
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


pairs = NN_2d(L,d;periodic=false)





H=build_hamiltonian(basis,index,pairs,t,U)
















#-------------build hamiltonina in csr format--------------
# H=build_hamiltonian(basis,index,pairs,t,U)
#------------------------------------------------------------

#--------------------------full dense mat---------------------
# H_dense=Matrix(H)       
# F=eigen(H_dense)  
#-------------------------------------------------------------


#-------------------------------krylov comlexity-------------------------------------

#applyH!(out, v) = apply_ham!(out,v,basis,index,pairs,t,U)

# eigvals,kry_ham=lanczos(applyH!,dim;m=m,rng=rng,init=:random)
# evolve_krylov(kry_ham;tmin=0.0,tmax=100.0,Nt=400,prefix="random",seed,L,d,t,U)

#--------------------------neel------------------------------------------------

# psi00 = zeros(Float64,dim)
# state = neel_state(L, d)

# occ = zeros(Int, L, d)
# state_bits_2d!(occ, state, L, d)

# println(occ)
# for y in 1:d
#     println("y=",occ[:, y])
# end
# ino=index[neel_state(L,d)]
# psi00[ino]=1.0


# eigvals,kry_ham=lanczos(applyH!,dim;m=m,rng=rng,init=:neel,v0=psi00)
# evolve_krylov(kry_ham;tmin=0.0,tmax=100.0,Nt=400,prefix="neel",seed,L,d,t,U)


#--------------------------------------------------------------------------------



