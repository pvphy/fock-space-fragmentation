module Basis

export generate_basis,state_index,site_index,site_coords,occ,state_bits_2d!,NN_2d,fij,fermionic_sign

    @inline function site_index(x::Int, y::Int, Lx::Int)
        return (y - 1) * Lx + x
    end


    @inline function site_coords(site::Int, Lx::Int)
        y = (site - 1) ÷ Lx + 1
        x = (site - 1) % Lx + 1
        return x, y
    end

    @inline occ(state::Int, site::Int) = (state >> (site - 1)) & 1


    function generate_basis(Lx::Int,Ly::Int,N::Int)   #Gosper’s hack
        L = Lx * Ly
        N > L && error("N >>L")

        basis = Vector{Int}()
        state =(1<<N) - 1
        limit =1<< L
        while state<limit
            push!(basis, state)
            c=state & -state
            r=state + c
            state=(((r ⊻ state) >> 2) ÷ c) | r
        end
        return basis
    end

    function state_index(basis::Vector{Int})
        idx = Dict{Int,Int}()
        sizehint!(idx, length(basis))   # pre-allocate hash table

        @inbounds for i in eachindex(basis)
            idx[basis[i]] = i
        end
        return idx
    end


    function state_bits_2d!(occ_mat::Matrix{Int},state::Int,Lx::Int,Ly::Int)
        @inbounds for y in 1:Ly, x in 1:Lx
            site = (y - 1) * Lx + x
            occ_mat[x, y] = (state >> (site - 1)) & 1
        end
        return nothing
    end

    function NN_2d(Lx::Int, Ly::Int; periodic::Bool=false)
        pairs = Tuple{Int,Int}[]

        for y in 1:Ly, x in 1:Lx
            i = (y - 1) * Lx + x

            # right neighbor
            if x < Lx
                push!(pairs, (i, i + 1))
            elseif periodic
                push!(pairs, (i, (y - 1) * Lx + 1))
            end

            # up neighbor
            if y < Ly
                push!(pairs, (i, i + Lx))
            elseif periodic
                push!(pairs, (x))
            end
        end

        return pairs
    end

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
end 



