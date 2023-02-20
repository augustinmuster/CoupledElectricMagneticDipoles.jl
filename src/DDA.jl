"""
CoupledElectricMagneticDipoles.jl : DDACore Module
This module contains the implementation of the basics functionalities of the DDA.
Author: Augustin Muster, January 2022, augustin@must-r.com+
"""
module DDACore
###########################
# IMPORTS
###########################
using Base
using DelimitedFiles
using LinearAlgebra
using IterativeSolvers
using LinearSolveCUDA
using LinearSolve
using Test
include("green_tensors_e_m.jl")

########################################################################################################################################
# FUNCTIONS
########################################################################################################################################
@doc raw"""
     solve_system(matrix,vector,solver,verbose)

Solves a system of equations of the type ``Ax=b`` wher `matrix` is ``A`` and `vector` is ``b`` using the method `solver` and returns `x`.
To choose the appropriate solver, please read the informations on the home page.
"""
function solve_system(matrix,vector,solver,verbose)
    A=copy(matrix)
    phi=copy(vector)
    #lapack julia solver
    if solver=="LAPACK"
        if verbose
            println("solving with LAPACK solver ...")
        end
        LAPACK.gesv!(A, phi)
    #Linearsolve auto solver
    elseif solver=="AUTO"
        if verbose
            println("solving with AUTO (LinearSolve) solver ...")
        end
        prob = LinearProblem(A, phi)
        sol = solve(prob)
        phi=sol.u
    #Linearsolve auto GPU solver
    elseif solver=="GPU"
        if verbose
            println("solving with GPU (LinearSolve) solver ...")
        end
        prob = LinearProblem(A, phi)
        sol = solve(prob,CudaOffloadFactorization())
        phi=sol.u
    end
    if verbose
        println("equations solved")
    end
    return phi
end

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
@doc raw"""
     invert_system(matrix,solver,verbose)

Inverts the square matrix `matrix` using the `solver` method
To choose the appropriate solver, please read the informations on the home page.
"""
function invert_system(matrix,solver,verbose)
    A=copy(matrix)
    #julia inverter
    if solver=="JULIA"
        if verbose
            println("inverting with julia inv() ...")
        end
        A_inv=inv(A)
    end
    if verbose
        println("matrix inverted")
    end
    return A_inv
end

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
@doc raw"""
     load_dda_matrix_e(kr,alpha_dl,verbose)

Builds the electric only DDA matrix ``A=[I-G\alpha]`` with dimensionless postitions `kr` (two dimensional arrays of size ``Nx3``) and dimensionless polarisabilities `alpha_dl` (a three-dimensional complex array of size ``N\times 3\times 3``containing the polarisability ``3\times 3`` tensor of each dipole, or one dimensional array of size ``N`` containing the scalar polarizability of each dipole.)
Returns ``3N\times 3N`` complex DDA matrix.
"""
function load_dda_matrix_e(kr,alpha_dl,verbose)
    #id matrix
    id=Matrix{ComplexF64}(I,3,3)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println("formatting the equations...")
        println("coefficient matrix")
    end
    #computing the big matrix
    A=zeros(ComplexF64,3*n,3*n)
    #if scalar polarisability
    if length(alpha_dl)==n
        for j in 1:n
            for k=1:n
                if k!=j
                    G=GreenTensors.G_e_renorm(kr[j,:],kr[k,:])
                    A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(-G*alpha_dl[k])
                else
                    A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(id)
                end
            end
        end
    else #if tensor polarisabilty
        for j in 1:n
            for k=1:n
                if k!=j
                    G=GreenTensors.G_e_renorm(kr[j,:],kr[k,:])
                    A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(-G*alpha_dl[k,:,:])
                else
                    A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(id)
                end
            end
        end
    end
    #logging
    if verbose
        println("equations formatted")
    end
    #return DDA matrix
    return A
end
@doc raw"""
     load_dda_matrix_e_m(kr,alpha_e_dl,alpha_m_dl,verbose)

Builds the electric and magnetic DDA matrix ``A=[I-G\alpha]`` with dimensionless postitions `kr` (two dimensional arrays of size ``N\times 3``) and dimensionless electric and magnetic polarisabilities `alpha_e_dl` and  `alpha_m_dl` (a three-dimensional complex array of size ``N\times 3\times 3``containing the polarisability ``3\times 3`` tensor of each dipole, or one dimensional array of size ``N`` containing the scalar polarizability of each dipole.)
Return ``6N\times 6N`` complex DDA matrix
"""
function load_dda_matrix_e_m(kr,alpha_e_dl,alpha_m_dl,verbose)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println("formatting the equations...")
        println("coefficient matrix")
    end
    #
    A=zeros(ComplexF64,6*n,6*n)
    #
    id=Matrix{ComplexF64}(I,6,6)
    #
    a_dda=zeros(ComplexF64,6,6)
    #
    if length(alpha_e_dl)==n
        for i=1:n
            for j=1:n
                if i==j
                    A[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=id
                else
                    a_dda[1:3,1:3]=-GreenTensors.G_e_renorm(kr[i,:],kr[j,:])*alpha_e_dl[j]
                    a_dda[4:6,4:6]=-GreenTensors.G_e_renorm(kr[i,:],kr[j,:])*alpha_m_dl[j]
                    a_dda[1:3,4:6]=-im*GreenTensors.G_m_renorm(kr[i,:],kr[j,:])*alpha_m_dl[j]
                    a_dda[4:6,1:3]=+im*GreenTensors.G_m_renorm(kr[i,:],kr[j,:])*alpha_e_dl[j]
                    A[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=copy(a_dda)
                end
            end
        end
    else
        for i=1:n
            for j=1:n
                if i==j
                    A[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=id
                else
                    a_dda[1:3,1:3]=-GreenTensors.G_e_renorm(kr[i,:],kr[j,:])*alpha_e_dl[j,:,:]
                    a_dda[4:6,4:6]=-GreenTensors.G_e_renorm(kr[i,:],kr[j,:])*alpha_m_dl[j,:,:]
                    a_dda[1:3,4:6]=-im*GreenTensors.G_m_renorm(kr[i,:],kr[j,:])*alpha_m_dl[j,:,:]
                    a_dda[4:6,1:3]=+im*GreenTensors.G_m_renorm(kr[i,:],kr[j,:])*alpha_e_dl[j,:,:]
                    A[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=copy(a_dda)
                end
            end
        end
    end
    #logging
    if verbose
        println("equations formatted")
    end
    return A
end
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
@doc raw"""
     solve_DDA_e(knorm,r,alpha,input_field::Function;solver="LAPACK",verbose=true)

Builds and solves the DDA equations under a given input field, i.e.
```math
\vec{E}_{i}=\vec{E}_{0}(\vec{r}_{i})+k^2\sum_{i\neq j}^{N}\tilde{G}_e(\vec{r}_{i},\vec{r}_{j})\alpha_{j}\vec{E}_{j},
```
for a group of ``N`` only electric dipoles and returns the polarizations and incident fields of every dipole.

#Arguments
- `knorm`: the wavenumber of the input field.
- `r`: a two-dimensional float array of size ``N\times 3`` containing the positions ``\vec{r}`` of all the dipoles.
- `alpha`: a three-dimensional complex array of size ``N\times 3\times 3``containing the polarisability ``\times 3`` tensor of each dipole, or one dimensional array of size ``N`` containing the scalar polarizability of each dipole.
- `input_field`: a function taking the wavenumber and one position vector of length 3, i.e. of the form `field(knorm,r)`, and that output a complex array of length 3 which computes the input field ``E_0`` evaluated at this position. Can be for example a plane wave or a point source.
- `solver`: a string that contains the name of the solver that need to be used. For this, check the correponding section on the home page. By default set to `"LAPACK"`.
- `verbose`: whether to output pieces of information to the standard output during running or not. By default set to `true`.
#Outputs
- `p`: a two-dimensional complex array of size ``N\times 3`` containing the polarization ``p`` of each dipole.
- `e_inc`: a two-dimensional complex array of size ``N\times 3`` containing the incident fields ``E_inc`` on each dipole.
"""
function solve_DDA_e(knorm,r,alpha,input_field::Function;solver="LAPACK",verbose=true)
    #number of point dipoles
    n=length(r[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #generate the matrix
    A=load_dda_matrix_e(knorm*r,alpha*knorm^3/4/pi,verbose)
    #computing input fields
    if verbose
        println("loading input fields")
    end
    #
    E=zeros(ComplexF64,3*n)
    #
    for j in 1:n
        pl=input_field(knorm,r[j,:])
        E[3*(j-1)+1]=pl[1]
        E[3*(j-1)+2]=pl[2]
        E[3*(j-1)+3]=pl[3]
    end
    #copy of the input fields
    E2=copy(E)
    #
    E=solve_system(A,E,solver,verbose)
    #Computing dipoles moments
    if verbose
        println("computing the polarisations")
    end
    p=zeros(ComplexF64,(n,3))
    e_inc=zeros(ComplexF64,(n,3))
    if length(alpha)==n
        for i=1:n
            p[i,:]=alpha[i]*[E[3*(i-1)+1],E[3*(i-1)+2],E[3*(i-1)+3]]
            e_inc[i,:]=[E2[3*(i-1)+1],E2[3*(i-1)+2],E2[3*(i-1)+3]]
        end
    else
        for i=1:n
            p[i,:]=alpha[i,:,:]*[E[3*(i-1)+1],E[3*(i-1)+2],E[3*(i-1)+3]]
            e_inc[i,:]=[E2[3*(i-1)+1],E2[3*(i-1)+2],E2[3*(i-1)+3]]
        end
    end
    #return polarisations and incident fields
    return p, e_inc
end

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

@doc raw"""
     solve_DDA_e(knorm,r,alpha;solver="JULIA",verbose=true)

Similar to `solve_DDA_e(knorm,r,alpha,input_field::Function;solver="LAPACK",verbose=true)`, but without input field. Returns then the inverse of the DDA Matrix.
#Arguments
- `knorm`: the wavenumber of the input field.
- `r`: a two-dimensional float array of size ``N\times 3`` containing the positions ``\vec{r}`` of all the dipoles.
- `alpha`: a three-dimensional complex array of size ``N\times 3\times 3``containing the polarisability ``3x3`` tensor of each dipole, or one dimensional array of size ``N`` containing the scalar polarizability of each dipole.
- `solver`: string that contains the name of the invertion method that need to be used. For this, check the correponding section on the home page. By default set to `"JULIA"`.
- `verbose`: whether to output informations to the standard output during running or not. By default set to `true`.
#Output
- ``3N\times 3N`` inverse of the DDA matrix.
"""
function solve_DDA_e(knorm,r,alpha;solver="JULIA",verbose=true)
    #number of point dipoles
    n=length(r[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #generate the matrix
    A=load_dda_matrix_e(knorm*r,alpha*knorm^3/4/pi,verbose)
    #retrun inverse of DDA matrix
    return invert_system(A,solver,verbose)
end

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
@doc raw"""
     solve_DDA_e(kr,alpha_dl,input_field::Function;solver="LAPACK",verbose=true)

Builds and solves the DDA equations with dimensionless input under a given input field, i.e.
```math
\vec{E}_{i}=\vec{E}_{0}(\vec{r}_{i})+,\sum^N_{j\neqj}G_e(\vec{r}_i,\vec{r}_j)\alpha_j\vec{E}_j
\end{equation}
```
for a group of ``N`` only electric dipoles and returns the polarizations and incident fields.

#Arguments
- `kr`: two dimensional float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_dl`: three dimensional complex array of size ``N\times 3\times 3``containing the dimensionless polarisability ``3\times 3`` tensor of each dipole, or one dimenstional array of size ``N`` containing the scalar polarizability of each dipole.
- `input_field`: function taking one dimensionless position vector of length 3, i.e. of the form `field(kr)`, and that output a complex array of length 3 which compute the input field ``E_0`` elvaluated at this position. Can be for example a plane wave or a point source.
- `solver`: string that contains the name of the solver that need to be used. For this, check the correponding section on the home page. By default set to `"LAPACK"`.
- `verbose`: whether to output informations to the standard output during running or not. By default set to `true`.
#Outputs
- `p`: two dimensional complex array of size ``N\times 3`` containing the polarizations ``p`` of each dipoles (in units of the electric field!).
- `e_inc`: two dimensional complex array of size ``N\times 3`` containing the incident fields ``E_inc`` on every dipole.
"""
function solve_DDA_e(kr,alpha_dl,input_field::Function;solver="LAPACK",verbose=true)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #generate the matrix
    A=load_dda_matrix_e(kr,alpha_dl,verbose)
    #computing input fields
    if verbose
        println("loading input fields")
    end
    #
    E=zeros(ComplexF64,3*n)
    #
    for j in 1:n
        pl=input_field(kr[j,:])
        E[3*(j-1)+1]=pl[1]
        E[3*(j-1)+2]=pl[2]
        E[3*(j-1)+3]=pl[3]
    end
    #copy of the input fields
    E2=copy(E)
    #
    E=solve_system(A,E,solver,verbose)
    #Computing dipoles moments
    if verbose
        println("computing the polarisations")
    end
    p=zeros(ComplexF64,(n,3))
    e_inc=zeros(ComplexF64,(n,3))
    if length(alpha_dl)==n
        for i=1:n
            p[i,:]=alpha_dl[i]*[E[3*(i-1)+1],E[3*(i-1)+2],E[3*(i-1)+3]]
            e_inc[i,:]=[E2[3*(i-1)+1],E2[3*(i-1)+2],E2[3*(i-1)+3]]
        end
    else
        for i=1:n
            p[i,:]=alpha_dl[i,:,:]*[E[3*(i-1)+1],E[3*(i-1)+2],E[3*(i-1)+3]]
            e_inc[i,:]=[E2[3*(i-1)+1],E2[3*(i-1)+2],E2[3*(i-1)+3]]
        end
    end
    #return polarisations and incident fields
    return p, e_inc
end

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
@doc raw"""
     solve_DDA_e(kr,alpha_dl;solver="JULIA",verbose=true)

Similar to `solve_DDA_e(kr,alpha_dl,input_field::Function;solver="LAPACK",verbose=true)`, but without input field. Return then the inverse of the DDA Matrix.

#Arguments
- `kr`: two dimensional float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_dl`: three dimensional complex array of size ``N\times 3\times 3``containing the dimensionless polarisability ``3\times 3`` tensor of each dipole, or one dimenstional array of size ``N`` containing the scalar polarizability of each dipole.
- `solver`: string that contains the name of the solver that need to be used. For this, check the correponding section on the home page. By default set to `"LAPACK"`.
- `verbose`: whether to output informations to the standard output during running or not. By default set to `true`.
#Output
- ``3N\times 3N`` inverse of the DDA matrix.
"""
function solve_DDA_e(kr,alpha_dl;solver="JULIA",verbose=true)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #generate the matrix
    A=load_dda_matrix_e(kr,alpha_dl,verbose)
    #retrun inverse of DDA matrix
    return invert_system(A,solver,verbose)
end

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
@doc raw"""
     solve_DDA_e_m(knorm,r,alpha_e,alpha_m,input_field::Function;solver="LAPACK",verbose=true)

Builds and solves the DDA equations under a given input field, i.e.
```math
\begin{align}
\vec{E}_{i} & =\vec{E}_{0}\left(\vec{r}_{i}\right)+\sum_{j\ne i}\mathbb{G}_{E}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{E}^{(j)}\vec{E}_{j}+i\mathbb{G}_{M}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{M}^{(j)}\vec{H}_{j}\label{eq:100a}\\
\vec{H}_{i} & =\vec{H}_{0}\left(\vec{r}_{i}\right)+\sum_{j\ne i}-i\mathbb{G}_{M}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{E}^{(j)}\vec{E}_{j}+\mathbb{G}_{E}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{M}^{(j)}\vec{H}_{j}\label{eq:100b}
\end{align}
```
for a group of ``N`` electric and magnetic dipoles and return the polarisations of every particle and incident fields on every particle.

#Arguments
- `knorm`: the wavenumber of the input field.
- `r`: a two-dimensional float array of size ``N\times 3`` containing the positions ``\vec{r}`` of all the dipoles.
- `alpha_e`: a three-dimensional complex array of size ``N\times 3\times 3``containing the electric polarisability ``3\times 3`` tensor of every dipole, or one dimensional array of size ``N`` containing the scalar polarizability of every dipole.
- `alpha_m`: a three-dimensional complex array of size ``N\times 3\times 3``containing the magnetic polarisability ``3\times 3`` tensor of every dipole, or one dimensional array of size ``N`` containing the scalar polarizability of every dipole.
- `input_field`: a function taking the wavenumber and one position vector of length 3, i.e. of the form `field(knorm,r)`, and that output a complex array of length 3 which computes the input fields ``E_0`` and ``H_0`` evaluated at this position. Can be for example a plane wave or a point source.
- `solver`: a string that five the name of the solver that need to be used. For this, check the correponding section on the home page. By default set to `"LAPACK"`.
- `verbose`: whether to output pieces of information to the standard output during running or not. By default set to `true`.
#Outputs
- `p`: a two-dimensional complex array of size ``N\times 3`` containing the electric polarizations ``p`` moments of all the electric dipoles.
- `m`: a two-dimensional complex array of size ``N\times 3`` containing the magnetic polarizations ``m`` moments of all the magnetic dipoles.
- `e_inc`: a two-dimensional complex array of size ``N\times 3`` containing the incident electric fields ``E_inc`` on all the particles.
- `h_inc`: a two-dimensional complex array of size ``N\times 3`` containing the incident magnetic fields ``H_inc`` on all the particles.
- `e_inp`: a two-dimensional complex array of size ``N\times 3`` containing the input electric fields ``E_inc`` on all the particles.
- `h_inp`: a two-dimensional complex array of size ``N\times 3`` containing the input magnetic fields ``H_inc`` on all the particles.
"""
function solve_DDA_e_m(knorm,r,alpha_e,alpha_m,input_field::Function;solver="LAPACK",verbose=true)
    #number of point dipoles
    n=length(r[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #loading matrix
    A=load_dda_matrix_e_m(knorm*r,alpha_e*knorm^3/4/pi,alpha_m*knorm^3/4/pi,verbose)
    #logging
    if verbose
        println("loading input fields")
    end
    #compute incident fields
    phi=zeros(ComplexF64,6*n)
    for j in 1:n
        E0,H0=input_field(knorm,r[j,:])
        phi[6*(j-1)+1:6*(j-1)+3]=E0
        phi[6*(j-1)+4:6*(j-1)+6]=H0
    end
    #copy
    phi2=copy(phi)
    #solve
    phi=solve_system(A,phi,solver,verbose)

    if verbose
        println("computing the polarisations and incident fields")
    end

    p=zeros(ComplexF64,(n,3))
    m=zeros(ComplexF64,(n,3))

    e_inc=zeros(ComplexF64,(n,3))
    h_inc=zeros(ComplexF64,(n,3))

    e_inp=zeros(ComplexF64,(n,3))
    h_inp=zeros(ComplexF64,(n,3))

    if length(alpha_m)==n
        for j=1:n
            p[j,:]=alpha_e[j]*phi[6*(j-1)+1:6*(j-1)+3]
            m[j,:]=alpha_m[j]*phi[6*(j-1)+4:6*(j-1)+6]
            e_inc[j,:]=phi[6*(j-1)+1:6*(j-1)+3]
            h_inc[j,:]=phi[6*(j-1)+4:6*(j-1)+6]
            e_inp[j,:]=phi2[6*(j-1)+1:6*(j-1)+3]
            h_inp[j,:]=phi2[6*(j-1)+4:6*(j-1)+6]
        end
    else
        for j=1:n
            p[j,:]=alpha_e[j,:,:]*phi[6*(j-1)+1:6*(j-1)+3]
            m[j,:]=alpha_m[j,:,:]*phi[6*(j-1)+4:6*(j-1)+6]
            e_inc[j,:]=phi[6*(j-1)+1:6*(j-1)+3]
            h_inc[j,:]=phi[6*(j-1)+4:6*(j-1)+6]
            e_inp[j,:]=phi2[6*(j-1)+1:6*(j-1)+3]
            h_inp[j,:]=phi2[6*(j-1)+4:6*(j-1)+6]
        end
    end
    #return polarisations and incident fields
    return p,m,e_inc,h_inc,e_inp,h_inp
end

@doc raw"""
     solve_DDA_e_m(knorm,r,alpha_e,alpha_m;solver="LAPACK",verbose=true)

Builds and solves the DDA equations under a given input field, i.e.
```math
\begin{align}
\vec{E}_{i} & =\vec{E}_{0}\left(\vec{r}_{i}\right)+\sum_{j\ne i}\mathbb{G}_{E}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{E}^{(j)}\vec{E}_{j}+i\mathbb{G}_{M}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{M}^{(j)}\vec{H}_{j}\label{eq:100a}\\
\vec{H}_{i} & =\vec{H}_{0}\left(\vec{r}_{i}\right)+\sum_{j\ne i}-i\mathbb{G}_{M}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{E}^{(j)}\vec{E}_{j}+\mathbb{G}_{E}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{M}^{(j)}\vec{H}_{j}\label{eq:100b}
\end{align}
```
for a group of ``N`` electric and magnetic dipoles and return the polarisations of every particle and incident fields on every particle.

#Arguments
- `knorm`: the wavenumber of the input field.
- `r`: a two-dimensional float array of size ``N\times 3`` containing the positions ``\vec{r}`` of all the dipoles.
- `alpha_e`: a three-dimensional complex array of size ``N\times 3\times 3``containing the electric polarisability ``3\times 3`` tensor of every dipole, or one dimensional array of size ``N`` containing the scalar polarizability of every dipole.
- `alpha_m`: a three-dimensional complex array of size ``N\times 3\times 3``containing the magnetic polarisability ``3\times 3`` tensor of every dipole, or one dimensional array of size ``N`` containing the scalar polarizability of every dipole.
- `input_field`: a function taking the wavenumber and one position vector of length 3, i.e. of the form `field(knorm,r)`, and that output a complex array of length 3 which computes the input fields ``E_0`` and ``H_0`` evaluated at this position. Can be for example a plane wave or a point source.
- `solver`: a string that five the name of the solver that need to be used. For this, check the correponding section on the home page. By default set to `"LAPACK"`.
- `verbose`: whether to output pieces of information to the standard output during running or not. By default set to `true`.
#Outputs
- ``6N\times 6N`` inverse of the DDA matrix
"""
function solve_DDA_e_m(knorm,r,alpha_e,alpha_m;solver="JULIA",verbose=true)
    #number of point dipoles
    n=length(r[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #loading matrix
    A=load_dda_matrix_e_m(knorm*r,alpha_e*knorm^3/4/pi,alpha_m*knorm^3/4/pi,verbose)
    return invert_system(A,solver,verbose)
end

@doc raw"""
     solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl,input_field::Function;solver="AUTO",verbose=true)

Builds and solves the DDA equations with dimensionless inputs under a given input field, i.e.
```math
\begin{align}
\vec{E}_{i} & =\vec{E}_{0}\left(\vec{r}_{i}\right)+\sum_{j\ne i}\mathbb{G}_{E}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{E}^{(j)}\vec{E}_{j}+i\mathbb{G}_{M}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{M}^{(j)}\vec{H}_{j}\label{eq:100a}\\
\vec{H}_{i} & =\vec{H}_{0}\left(\vec{r}_{i}\right)+\sum_{j\ne i}-i\mathbb{G}_{M}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{E}^{(j)}\vec{E}_{j}+\mathbb{G}_{E}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{M}^{(j)}\vec{H}_{j}\label{eq:100b}
\end{align}
```
for a group of ``N`` electric and magnetic dipoles and return the polarisations of every particle and incident fields on every particle.

#Arguments
- `kr`: a two-dimensional float array of size ``N\times 3`` containing the dimensionless positions ``k\vec{r}`` of all the dipoles.
- `alpha_e_dl`: a three-dimensional complex array of size ``N\times 3\times 3``containing the dimensionless electric polarisability ``3\times 3`` tensor of every dipole, or one dimensional array of size ``N`` containing the scalar polarizability of every dipole.
- `alpha_m_dl`: a three-dimensional complex array of size ``N\times 3\times 3``containing the dimesnionless magnetic polarisability ``3\times 3`` tensor of every dipole, or one dimensional array of size ``N`` containing the scalar polarizability of every dipole.
- `input_field`: a function taking the dimensionless position vector of length 3, i.e. of the form `field(kr)`, and that output two complex array of length 3 which computes the input fields ``E_0`` and ``H_0`` evaluated at this position. Can be for example a plane wave or a point source.
- `solver`: a string that five the name of the solver that need to be used. For this, check the correponding section on the home page. By default set to `"LAPACK"`.
- `verbose`: whether to output pieces of information to the standard output during running or not. By default set to `true`.
#Outputs
- `p`: a two-dimensional complex array of size ``N\times 3`` containing the electric polarizations ``p`` moments of all the electric dipoles (in units of the electric field).
- `m`: a two-dimensional complex array of size ``N\times 3`` containing the magnetic polarizations ``m`` moments of all the magnetic dipoles (in units of the electric field).
- `e_inc`: a two-dimensional complex array of size ``N\times 3`` containing the incident electric fields ``E_inc`` on all the particles (in units of the electric field).
- `h_inc`: a two-dimensional complex array of size ``N\times 3`` containing the incident magnetic fields ``H_inc`` on all the particles (in units of the electric field).
- `e_inp`: a two-dimensional complex array of size ``N\times 3`` containing the input electric fields ``E_inc`` on all the particles (in units of the electric field).
- `h_inp`: a two-dimensional complex array of size ``N\times 3`` containing the input magnetic fields ``H_inc`` on all the particles (in units of the electric field).
"""
function solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl,input_field::Function;solver="AUTO",verbose=true)
    #number of point dipoles
    n=length(kr[:,1])

    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #loading matrix
    A=load_dda_matrix_e_m(kr,alpha_e_dl,alpha_m_dl,verbose)
    #logging
    if verbose
        println("loading input fields")
    end
    #compute incident fields
    phi=zeros(ComplexF64,6*n)
    for j in 1:n
        E0,H0=input_field(kr[j,:])
        phi[6*(j-1)+1:6*(j-1)+3]=E0
        phi[6*(j-1)+4:6*(j-1)+6]=H0
    end
    #copy
    phi2=copy(phi)
    #solve
    phi=solve_system(A,phi,solver,verbose)

    if verbose
        println("computing the polarisations and incident fields")
    end

    p=zeros(ComplexF64,(n,3))
    m=zeros(ComplexF64,(n,3))

    e_inc=zeros(ComplexF64,(n,3))
    h_inc=zeros(ComplexF64,(n,3))

    e_inp=zeros(ComplexF64,(n,3))
    h_inp=zeros(ComplexF64,(n,3))

    if length(alpha_m_dl)==n
        for j=1:n
            p[j,:]=alpha_e_dl[j]*phi[6*(j-1)+1:6*(j-1)+3]
            m[j,:]=alpha_m_dl[j]*phi[6*(j-1)+4:6*(j-1)+6]
            e_inc[j,:]=phi[6*(j-1)+1:6*(j-1)+3]
            h_inc[j,:]=phi[6*(j-1)+4:6*(j-1)+6]
            e_inp[j,:]=phi2[6*(j-1)+1:6*(j-1)+3]
            h_inp[j,:]=phi2[6*(j-1)+4:6*(j-1)+6]
        end
    else
        for j=1:n
            p[j,:]=alpha_e_dl[j,:,:]*phi[6*(j-1)+1:6*(j-1)+3]
            m[j,:]=alpha_m_dl[j,:,:]*phi[6*(j-1)+4:6*(j-1)+6]
            e_inc[j,:]=phi[6*(j-1)+1:6*(j-1)+3]
            h_inc[j,:]=phi[6*(j-1)+4:6*(j-1)+6]
            e_inp[j,:]=phi2[6*(j-1)+1:6*(j-1)+3]
            h_inp[j,:]=phi2[6*(j-1)+4:6*(j-1)+6]
        end
    end
    #return polarisations and incident fields
    return p,m,e_inc,h_inc,e_inp,h_inp
end

@doc raw"""
     solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl;solver="AUTO",verbose=true)

Similar to `solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl,input_field::Function;solver="AUTO",verbose=true)`, but without input field. Return then the inverse of the DDA Matrix.
#Arguments
- `kr`: a two-dimensional float array of size ``N\times 3`` containing the dimensionless positions ``k\vec{r}`` of all the dipoles.
- `alpha_e_dl`: a three-dimensional complex array of size ``N\times 3\times 3``containing the dimensionless electric polarisability ``3\times 3`` tensor of every dipole, or one dimensional array of size ``N`` containing the scalar polarizability of every dipole.
- `alpha_m_dl`: a three-dimensional complex array of size ``N\times 3\times 3``containing the dimesnionless magnetic polarisability ``3\times 3`` tensor of every dipole, or one dimensional array of size ``N`` containing the scalar polarizability of every dipole.
- `solver`: a string that five the name of the solver that need to be used. For this, check the correponding section on the home page. By default set to `"LAPACK"`.
- `verbose`: whether to output pieces of information to the standard output during running or not. By default set to `true`.
#Outputs
- ``6N\times 6N`` inverse of the DDA matrix
"""
function solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl;solver="AUTO",verbose=true)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #loading matrix
    A=load_dda_matrix_e_m(kr,alpha_e_dl,alpha_m_dl,verbose)
    #logging
    if verbose
        println("loading input fields")
    end
    #compute incident fields
    phi=zeros(ComplexF64,6*n)
    for j in 1:n
        E0,H0=input_field(kr[j,:])
        phi[6*(j-1)+1:6*(j-1)+3]=E0
        phi[6*(j-1)+4:6*(j-1)+6]=H0
    end
    #solve
    return invert_matrix(matrix,solver,verbose)
end
end
