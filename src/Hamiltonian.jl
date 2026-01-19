module Hamiltonian
    using SparseArrays
    include("Basis.jl")
    using .Basis

    export build_hamiltonian

    @inline function fij(state::Int, i::Int, j::Int)   #number of fermions between i and j
        i == j && return 0
        lo = min(i, j)
        hi = max(i, j)

        # bits strictly between lo and hi
        mask = ((1 << (hi - lo - 1)) - 1) << lo
        return count_ones(state & mask)
    end

    @inline function fermionic_sign(state::Int, i::Int, j::Int)
        iseven(fij(state, i, j)) ? 1.0 : -1.0
    end


    function build_hamiltonian(basis::Vector{Int},index::Dict{Int,Int},pairs::Vector{Tuple{Int,Int}},t::Float64,U::Float64)

        dim=length(basis)
        rows=Int[]
        cols=Int[]
        vals=Float64[]

        sizehint!(rows,10dim)
        sizehint!(cols,10dim)
        sizehint!(vals,10dim)

        @inbounds for s_index_i in 1:dim
            state = basis[s_index_i]

            # ---------- diagonal term ----------
            diag = 0.0
            for (i,j) in pairs
                diag+=U*occ(state,i)*occ(state,j)
            end
            if diag !=0.0
                push!(rows,s_index_i)
                push!(cols,s_index_i)
                push!(vals,diag)
            end

            # ---------- hopping terms----------
            for(i,j) in pairs

                # i to j
                if occ(state,i)==1 && occ(state,j)==0
                    sign=fermionic_sign(state, i, j)
                    new_state=state ⊻ (1<<(i-1)) ⊻ (1<<(j-1))
                    s_index_new=index[new_state]

                    push!(rows,s_index_new)
                    push!(cols,s_index_i)
                    push!(vals,-t*sign)
                end

                # j to i
                if occ(state, j)==1 && occ(state,i)==0
                    sign=fermionic_sign(state, j, i)
                    new_state=state ⊻ (1<<(j-1)) ⊻ (1<<(i-1))
                    s_index_new=index[new_state]

                    push!(rows,s_index_new)
                    push!(cols,s_index_i)
                    push!(vals,-t*sign)
                end
            end
        end

        return sparse(rows,cols,vals,dim,dim)
    end



end