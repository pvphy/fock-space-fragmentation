module Lanczos
    using LinearAlgebra
    export lanczos
    function lanczos(applyH!, dim::Int; m::Int=50,rng,init::Symbol=:random,v0 = nothing)

        if init== :random
            v=randn(rng, dim)

        elseif init == :neel
            v0===nothing && error("For init=:neel, provide psi00")
            v=copy(v0)

        else
            error("Unknown init = $init")
        end


 
        v./= norm(v)

        w=zeros(dim)
        v_old=zeros(dim)

        alpha=zeros(m)
        beta=zeros(m-1)

        for j in 1:m
            applyH!(w,v)
            alpha[j] = dot(v,w)

            if j > 1
                w .-= beta[j-1] .*v_old
            end

            w .-= alpha[j] .*v

            if j < m
                beta[j]=norm(w)
                beta[j] < 1e-12 && break
                v_old .= v
                v .= w ./ beta[j]
            end
        end

        T = SymTridiagonal(alpha,beta)
        return eigen(T).values,T
    end

end 
