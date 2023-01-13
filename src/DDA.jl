"""
CoupledElectricMagneticDipoles.jl : DDACore Module
This module contains the implementation of the basics functionalities of the DDA.
Note that the solve_dda
Author: Augustin Muster, November 2022, augustin@must-r.com
"""
module DDACore
###########################
# IMPORTS
###########################
using Base
using DelimitedFiles
using LinearAlgebra
#using IterativeSolvers
#using LinearSolveCUDA
#using LinearSolve
include("green_tensors_e_m.jl")
###########################
# FUNCTIONS
#######m####################
#*************************************************
#DDA solver: solve the DDA equations
#INPUTS: norm of the k, positions of the dipoles, polarisabilities tensor of each dipoles, function to generate the input field, which output to take, which solver to take, verbose
#OUTPUT: array with the polarisations vectors or inverse of the equations matrix.
#*************************************************
function solve_DDA_e(knorm,r,alpha,input_field::Function;output="polarisations",solver="LAPACK",verbose=true)
    #number of point dipoles
    n=length(r[:,1])
    #logging
    if verbose
        println("lattice informations:")
        println("number of dipoles: ",n)
        println()
    end
    #identity matrix
    id=Matrix{ComplexF64}(I,3,3)

    #**********generations of the equations*********
    if verbose
        println("formatting the equations...")
        println("coefficient matrix")
    end
    #computing the big matrix
    A=zeros(ComplexF64,3*n,3*n)
    #big green tensor
    if output=="full_matrix"
        Gbig=zeros(ComplexF64,3*n,3*n)
    end

    for j in 1:n
        for k=1:n
            if k!=j
                G=GreenTensors.G_e(r[j,:],r[k,:],knorm)
                A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(-knorm^2*G*alpha[k,:,:])
                if output=="full_matrix"
                    Gbig[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3] = copy(G)
                end
            else
                A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(id)
            end
        end
    end

    #computing incident fields
    if verbose
        println("incident fields")
    end
    E=zeros(ComplexF64,3*n)

    for j in 1:n
        pl=input_field(knorm,r[j,:])
        E[3*(j-1)+1]=pl[1]
        E[3*(j-1)+2]=pl[2]
        E[3*(j-1)+3]=pl[3]
    end
    #return inverse of the matrix if required
    if output=="matrix"
        #LAPACK.getri!(A)
        return inv(A)
    elseif output=="full_matrix"
        return Gbig,inv(A)
    end
    #
    if verbose
        println("equations formatted")
        println()
    end
    #copy of the incident fields
    E2=copy(E)

    #**********solving equations using LinearSolve********
    if solver=="LAPACK"
        if verbose
            println("solving with LAPACK solver ...")
            println()
        end
        LAPACK.gesv!(A, E)
    end
    #=
    elseif solver=="AUTO"
        if verbose
            println("solving with AUTO (LinearSolve) solver ...")
            println()
        end
        prob = LinearProblem(A, E)
        sol = solve(prob)
        E=sol.u
    elseif solver=="GPU"
        if verbose
            println("solving with GPU (LinearSolve) solver ...")
            println()
        end
        prob = LinearProblem(A, E)
        sol = solve(prob,CudaOffloadFactorization())
        E=sol.u
    elseif solver=="TEST"
        if verbose
            println("testing the solvers...")
            println()
        end
        prob = LinearProblem(A, E)
        times=zeros(Float64,8)
        times[1]=n
        println("solver1")
        times[2]= @elapsed solve(prob)
        println("solver2")
        times[3]= @elapsed solve(prob,FastLUFactorization())
        println("solver3")
        #times[4]= @elapsed solve(prob, FastQRFactorization())
        println("solver4")
        #times[5]= @elapsed solve(prob, LUFactorization())
        println("solver5")
        #times[6]= @elapsed solve(prob, QRFactorization())
        println("solver6")
        #times[7]= @elapsed solve(prob, IterativeSolversJL_CG())
        println("solver7")
        times[8]= @elapsed solve(prob, CudaOffloadFactorization())
        if verbose
            println(times)
        end
        return times
    =#
    end

    if verbose
        println("equations solved")
        println()
    end

    #**********compute polarizations+formatting incident field*******
    if verbose
        println("computing the polarisations")
        println()
    end
    p=zeros(ComplexF64,(n,3))
    e_inc=zeros(ComplexF64,(n,3))
    for i=1:n
        p[i,:]=alpha[i,:,:]*[E[3*(i-1)+1],E[3*(i-1)+2],E[3*(i-1)+3]]
        e_inc[i,:]=[E2[3*(i-1)+1],E2[3*(i-1)+2],E2[3*(i-1)+3]]
    end
    #return polarisations and incident fields
    return p, e_inc
end

#*************************************************
#DDA solver: solve the DDA equations FOR ELECTRIC AND MAGNETIC DIPOLES !!!!!!!!!!! USE RENORM FUNCTIONS!!!!!!!
#INPUTS: norm of the k, positions of the dipoles, polarisabilities tensor of each dipoles electric,polarisabilities tensor of each dipoles magnetic, function to generate the input field, which output to take, which solver to take, verbose
#OUTPUT: array with the polarisations vectors or inverse of the equations matrix.
#*************************************************
function solve_DDA_e_m(knorm,r,alpha_e,alpha_m,input_field::Function;output="polarisations",solver="AUTO",verbose=true)
    #number of point dipoles
    n=length(r[:,1])
    #logging
    if verbose
        println("lattice informations:")
        println("number of dipoles: ",n)
        println()
    end
    #identity matrix
    id=Matrix{ComplexF64}(I,3,3)

    #**********generations of the equations*********
    if verbose
        println("formatting the equations...")
        println("coefficient matrix")
    end

    A=zeros(ComplexF64,6*n,6*n)
    #big green
    if output=="full_matrix"
        Gbig=zeros(ComplexF64,6*n,6*n)
    end
    id=Matrix{ComplexF64}(I,6,6)
    a_dda=zeros(ComplexF64,6,6)
    for i=1:n
        for j=1:n
            if i==j
                A[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=id

            else
                a_dda[1:3,1:3]=-GreenTensors.G_e_renorm(knorm*r[i,:],knorm*r[j,:])*alpha_e[j,:,:]
                a_dda[4:6,4:6]=-GreenTensors.G_e_renorm(knorm*r[i,:],knorm*r[j,:])*alpha_m[j,:,:]
                a_dda[1:3,4:6]=-im*GreenTensors.G_m_renorm(knorm*r[i,:],knorm*r[j,:])*alpha_m[j,:,:]
                a_dda[4:6,1:3]=+im*GreenTensors.G_m_renorm(knorm*r[i,:],knorm*r[j,:])*alpha_e[j,:,:]

                if output=="full_matrix"
                    a_dda[1:3,1:3]=GreenTensors.G_e_renorm(knorm*r[i,:],knorm*r[j,:])
                    a_dda[4:6,4:6]=GreenTensors.G_e_renorm(knorm*r[i,:],knorm*r[j,:])
                    a_dda[1:3,4:6]=GreenTensors.G_m_renorm(knorm*r[i,:],knorm*r[j,:])
                    a_dda[4:6,1:3]=GreenTensors.G_m_renorm(knorm*r[i,:],knorm*r[j,:])
                    Gbig[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6] = copy(a_dda)
                end
                A[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=copy(a_dda)
            end
        end
    end
    #return inverse of the matrix if required
    if output=="matrix"
        #LAPACK.getri!(A)
        return inv(A)
    elseif output=="full_matrix"
        return Gbig,inv(A)
    end

    #computing incident fields
    if verbose
        println("incident fields")
    end
    phi=zeros(ComplexF64,6*n)
    for j in 1:n
        E0,H0=input_field(knorm*r[j,:])
        phi[6*(j-1)+1:6*(j-1)+3]=E0
        phi[6*(j-1)+4:6*(j-1)+6]=H0
    end

    phi2=copy(phi)

    if verbose
        println("equations formatted")
        println()
    end
    #copy of the incident fields
    #**********solving equations using LinearSolve********
    if solver=="LAPACK"
        if verbose
            println("solving with LAPACK solver ...")
            println()
        end
        LAPACK.gesv!(A, phi)
    elseif solver=="AUTO"
        if verbose
            println("solving with AUTO (LinearSolve) solver ...")
            println()
        end
        prob = LinearProblem(A, phi)
        sol = solve(prob)
        phi=sol.u
    elseif solver=="GPU"
        if verbose
            println("solving with GPU (LinearSolve) solver ...")
            println()
        end
        prob = LinearProblem(A, phi)
        sol = solve(prob,CudaOffloadFactorization())
        phi=sol.u
    elseif solver=="TEST"
        if verbose
            println("testing the solvers...")
            println()
        end
        prob = LinearProblem(A, phi)
        times=zeros(Float64,8)
        times[1]=n
        println("solver1")
        times[2]= @elapsed solve(prob)
        println("solver2")
        times[3]= @elapsed solve(prob,FastLUFactorization())
        println("solver3")
        #times[4]= @elapsed solve(prob, FastQRFactorization())
        println("solver4")
        #times[5]= @elapsed solve(prob, LUFactorization())
        println("solver5")
        #times[6]= @elapsed solve(prob, QRFactorization())
        println("solver6")
        #times[7]= @elapsed solve(prob, IterativeSolversJL_CG())
        println("solver7")
        times[8]= @elapsed solve(prob, CudaOffloadFactorization())
        if verbose
            println(times)
        end
        return times
    end

    if verbose
        println("equations solved")
        println()
    end

    #**********compute polarizations+formatting incident field*******
    if verbose
        println("computing the polarisations")
        println()
    end

    p=zeros(ComplexF64,(n,3))
    m=zeros(ComplexF64,(n,3))

    e_inc=zeros(ComplexF64,(n,3))
    h_inc=zeros(ComplexF64,(n,3))

    e_inp=zeros(ComplexF64,(n,3))
    h_inp=zeros(ComplexF64,(n,3))

    for j=1:n
        p[j,:]=alpha_e[j,:,:]*phi[6*(j-1)+1:6*(j-1)+3]
        m[j,:]=alpha_m[j,:,:]*phi[6*(j-1)+4:6*(j-1)+6]
        e_inc[j,:]=phi[6*(j-1)+1:6*(j-1)+3]
        h_inc[j,:]=phi[6*(j-1)+4:6*(j-1)+6]
        e_inp[j,:]=phi2[6*(j-1)+1:6*(j-1)+3]
        h_inp[j,:]=phi2[6*(j-1)+4:6*(j-1)+6]
    end
    #return polarisations and incident fields
    return p,m,e_inc,h_inc,e_inp,h_inp
end
end



"""
#*************************************************
#DDA solver: solve the DDA equations
#INPUTS: norm of the k vector, positions of the dipoles times k, polarisabilities tensor of each dipoles times k³, function to generate the input field (with dimensionless inputs), which output to take, which solver to take, verbose
#OUTPUT: array with the polarisations vectors or inverse of the equations matrix.
#*************************************************
function solve_DDA_renorm(knorm,kr,alpha,input_field::Function;output="polarisations",solver="LU",verbose=true)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println("lattice informations:")
        println("number of dipoles: ",n)
        println()
    end
    #identity matrix
    id=Matrix{ComplexF64}(I,3,3)

    #**********formation of the equation*********
    if verbose
        println("formatting the equations...")
        println("coefficient matrix")
    end
    #computing the big matrix
    A=zeros(ComplexF64,3*n,3*n)

    for j in 1:n
        for k=1:n
            if k!=j
                G=G_e_renorm(kr[j,:],kr[k,:])
                A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(-knorm^(-1)*G*alpha[k,:,:])
            else
                A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(id)
            end
        end
    end

    #computing incident fields
    if verbose
        println("incident fields")
    end
    E=zeros(ComplexF64,3*n)

    for j in 1:n
        pl=input_field(kr[j,:])
        E[3*(j-1)+1]=pl[1]
        E[3*(j-1)+2]=pl[2]
        E[3*(j-1)+3]=pl[3]
    end

    if verbose
        println("equations formatted")
        println()
    end
    #copy of the incident fields
    E2=copy(E)


    #**********SOLVING equation using lapack LU decomposition********
    if solver=="LU"
        if verbose
            println("solving with lapack LU ...")
            println()
        end
        LAPACK.gesv!(A, E)
    end

    if output=="matrix"
        return A
    end

    if verbose
        println("equations solved")
        println()
    end


    #**********compute polarizations+formatting incident field*******
    if verbose
        println("computing the polarisations")
        println()
    end
    p=zeros(ComplexF64,(n,3))
    e_inc=zeros(ComplexF64,(n,3))
    for i=1:n
        p[i,:]=alpha[i,:,:]*[E[3*(i-1)+1],E[3*(i-1)+2],E[3*(i-1)+3]]
        e_inc[i,:]=[E2[3*(i-1)+1],E2[3*(i-1)+2],E2[3*(i-1)+3]]
    end
    #return polarisations and incident fields
    return p, e_inc
end
"""
