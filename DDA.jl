###########################
# IMPORTS
###########################
using Base
using DelimitedFiles
using LinearAlgebra
using IterativeSolvers
###########################
# FUNCTIONS
#######m####################
#*************************************************
#green tensor: compute the green tensor
#INPUTS: two vectors, norm of the k vector
#OUTPUT: green tensor
#*************************************************
function green(r1,r2,knorm)
    R=r1-r2
    term1=exp(im*knorm*norm(R))/(4*pi*norm(R))
    term2=1+(im/(knorm*norm(R)))-(1/(knorm^2*norm(R)^2))
    term3=1+(3*im/(knorm*norm(R)))-(3/(knorm^2*norm(R)^2))
    matrix=R*transpose(R)
    id3=[1 0 0;0 1 0;0 0 1]
    return term1*(term2*id3-term3*matrix/norm(R)^2)
end
#*************************************************
#DDA solver: solve the DDA equations
#INPUTS: norm of the k, positions of the dipoles, polarisabilities tensor of each dipoles, function to generate the input field, which output to take, which solver to take, verbose
#OUTPUT: array with the polarisations vectors or inverse of the equations matrix.
#*************************************************
function solve_DDA(knorm,r,alpha,input_field::Function;output="polarisations",solver="LU",verbose=true)
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

    for j in 1:n
        for k=1:n
            if k!=j
                G=green(r[j,:],r[k,:],knorm)
                A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(-knorm^2*G*alpha[k,:,:])
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

    if verbose
        println("equations formatted")
        println()
    end
    #copy of the incident fields
    E2=copy(E)

    #**********solving equation using lapack LU decomposition********
    if solver=="LU"
        if verbose
            println("solving with lapack LU ...")
            println()
        end
        LAPACK.gesv!(A, E)
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
#green tensor: compute the green tensor with dimensionless inputs
#INPUTS: two vectors  multiplied by the norm of the k vector, norm of the k vector
#OUTPUT: green tensor
#*************************************************
function green_dl(kr1,kr2,k)
    R=kr1-kr2
    term1=exp(im*norm(R))/(4*pi*norm(R))
    term2=1+(im/(norm(R)))-(1/(norm(R)^2))
    term3=1+(3*im/(norm(R)))-(3/(norm(R)^2))
    matrix=R*transpose(R)
    id3=[1 0 0;0 1 0;0 0 1]
    return k*term1*(term2*id3-term3*matrix/norm(R)^2)
end

#*************************************************
#DDA solver: solve the DDA equations
#INPUTS: norm of the k vector, positions of the dipoles times k, polarisabilities tensor of each dipoles times k³, function to generate the input field (with dimensionless inputs), which output to take, which solver to take, verbose
#OUTPUT: array with the polarisations vectors or inverse of the equations matrix.
#*************************************************
function solve_DDA_dl(knorm,kr,k3alpha,input_field::Function;solver="LU",verbose=true)
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
                G=green_dl(kr[j,:],kr[k,:],knorm)
                A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(-knorm^(-1)*G*k3alpha[k,:,:])
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
        p[i,:]=k3alpha[i,:,:]*[E[3*(i-1)+1],E[3*(i-1)+2],E[3*(i-1)+3]]/knorm^3
        e_inc[i,:]=[E2[3*(i-1)+1],E2[3*(i-1)+2],E2[3*(i-1)+3]]
    end
    #return polarisations and incident fields
    return p, e_inc
end
