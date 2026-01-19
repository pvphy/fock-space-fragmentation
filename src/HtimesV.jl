module HXV
using ..Basis   

export apply_ham!


function apply_ham!(out::Vector{Float64}, v::Vector{Float64},basis::Vector{Int},index::Dict{Int,Int},pairs::Vector{Tuple{Int,Int}},t::Float64,U::Float64)

    fill!(out, 0.0)
    dim = length(basis)

    @inbounds for s_index_i in 1:dim
        state = basis[s_index_i]
        amp   = v[s_index_i]
        amp == 0.0 && continue

        # ---------- diagonal interaction ----------
        diag = 0.0
        for (i,j) in pairs
            diag += U*occ(state,i)*occ(state, j)
        end
        out[s_index_i] += diag * amp

        # ---------- hopping ----------
        for (i,j) in pairs

            # i to j
            if occ(state,i)==1 && occ(state,j) == 0
                sign=fermionic_sign(state,i,j)
                new_state=state ⊻ (1 << (i-1)) ⊻ (1 << (j-1))
                s_index_new=index[new_state]
                out[s_index_new]+=-t*sign*amp
            end

            # j to i
            if occ(state,j)==1 && occ(state,i) == 0
                sign = fermionic_sign(state,j,i)
                new_state = state ⊻ (1 << (j-1)) ⊻ (1 << (i-1))
                s_index_new = index[new_state]
                out[s_index_new] += -t*sign*amp
            end
        end
    end

    return nothing
end

end 
