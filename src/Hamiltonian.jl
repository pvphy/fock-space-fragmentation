module Hamiltonian

using SparseArrays
include("src/basis.jl")
using .Basis

export build_hamiltonian_sparse

"""
    build_hamiltonian_sparse(basis, index, pairs, t, V)

Construct sparse Hamiltonian matrix for

H = -t ∑⟨ij⟩ (c†ᵢ cⱼ + h.c.) + V ∑⟨ij⟩ nᵢ nⱼ
"""
function build_hamiltonian_sparse(basis::Vector{Int},
                                  index::Dict{Int,Int},
                                  pairs::Vector{Tuple{Int,Int}},
                                  t::Float64,
                                  V::Float64)

    dim = length(basis)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    sizehint!(rows, 10dim)
    sizehint!(cols, 10dim)
    sizehint!(vals, 10dim)

    @inbounds for α in 1:dim
        state = basis[α]

        # ---------- diagonal interaction ----------
        diag = 0.0
        for (i, j) in pairs
            diag += V * occ(state, i) * occ(state, j)
        end
        if diag != 0.0
            push!(rows, α)
            push!(cols, α)
            push!(vals, diag)
        end

        # ---------- hopping ----------
        for (i, j) in pairs

            # i → j
            if occ(state, i) == 1 && occ(state, j) == 0
                sign = fermionic_sign(state, i, j)
                new_state = state ⊻ (1 << (i-1)) ⊻ (1 << (j-1))
                β = index[new_state]

                push!(rows, β)
                push!(cols, α)
                push!(vals, -t * sign)
            end

            # j → i
            if occ(state, j) == 1 && occ(state, i) == 0
                sign = fermionic_sign(state, j, i)
                new_state = state ⊻ (1 << (j-1)) ⊻ (1 << (i-1))
                β = index[new_state]

                push!(rows, β)
                push!(cols, α)
                push!(vals, -t * sign)
            end
        end
    end

    return sparse(rows, cols, vals, dim, dim)
end

end # module
