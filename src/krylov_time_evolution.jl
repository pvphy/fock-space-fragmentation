module KrylovTimeEvolution
    using LinearAlgebra
    export evolve_krylov

    function evolve_krylov(kry_ham;tmin=0.0,tmax=50.0,Nt=50,prefix = "krylov",seed,L,J,delta)
        #evoluve in eigenbasis of Hk and then go back to krylov bais
        M=size(kry_ham, 1) #number of Krylov basis states


        phi0=zeros(ComplexF64, M)
        phi0[1]=1.0+0im    #At time t=0, the system is entirely at Krylov first krylov site 


        eig=eigen(kry_ham)
        U=eig.vectors
        E=eig.values


        c0=U'*phi0       #initial weight in energy space


        times=range(tmin, tmax, length=Nt)


        nvals=collect(0:M-1)


        # fname_prob = "$(prefix)_L$(L)_J$(J)_delta$(delta)_seed$(seed)_probabilities.txt"
        fname_comp="$(prefix)_L$(L)_J$(J)_delta$(delta)_seed$(seed)_complexity.txt"
        fname_phi="$(prefix)_L$(L)_J$(J)_delta$(delta)_seed$(seed)_phi_nt.txt"
        # open(fname_prob, "w") do io_p
        open(fname_comp, "w") do io_k
        open(fname_phi, "w") do io_phi    


            for t in times

                phase=exp.(-1im .* E .* t)
                phi=U*(phase .* c0)     #back to krylov basis
                pn=abs2.(phi)

                # for n in 1:M
                #     println(io_p, t, "  ", nvals[n], "  ", pn[n])
                # end
                # println(io_p) 

                for n in 1:M
                    println(io_phi,t,"  ",nvals[n],"  ",real(phi[n]),"  ",imag(phi[n]))
                end
                println(io_phi)

            
                Kt=sum(nvals .* pn)
                println(io_k,t, "  ",Kt)
            end

        end
        end

        return nothing
    end

end 
