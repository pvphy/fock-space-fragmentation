module imbalance

using LinearAlgebra
using SparseArrays
using KrylovKit

import ..Basis: site_coords, occ

export krylov_step,imbalance_operator_2d,imbalance_vs_time

# --------------------------------------------------
# Krylov time evolution
# --------------------------------------------------
function krylov_step(H::SparseMatrixCSC{<:Complex},ψ::Vector{ComplexF64},dt::Real)

    ψnew,info = expv(-1im*dt,H,ψ)
    return ψnew
end

# --------------------------------------------------
# 2D checkerboard imbalance operator
# --------------------------------------------------
function imbalance_operator_2d(basis::Vector{Int},Lx::Int, Ly::Int)

    dim=length(basis)
    L=Lx*Ly
    diag = zeros(Float64,dim)

    @inbounds for (α, state) in enumerate(basis)
        s=0.0
        for site in 1:L
            x,y=site_coords(site, Lx)
            s +=(-1)^(x + y)*occ(state, site)
        end
        diag[α]=s/L
    end

    return Diagonal(diag)
end

# --------------------------------------------------
# Imbalance vs time
# --------------------------------------------------
function imbalance_vs_time(H::SparseMatrixCSC{<:Complex},ψ0::Vector{ComplexF64},Iop::Diagonal,times::Vector{Float64})

    @assert issorted(times)

    ψ=copy(ψ0)
    ψ./=norm(ψ)

    Ivals=zeros(Float64, length(times))
    Ivals[1]=real(dot(ψ,Iop*ψ))

    for k in 2:length(times)
        dt=times[k]-times[k-1]
        ψ=krylov_step(H,ψ,dt)
        Ivals[k]=real(dot(ψ,Iop*ψ))
    end

    return Ivals
end

end 
