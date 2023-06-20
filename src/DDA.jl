"""
CoupledElectricMagneticDipoles.jl : DDACore Module
This module contains the implementation of the basics functionalities of the DDA.
Author: Augustin Muster, augustin@must-r.comJanuary 2022,
"""
module DDACore
###########################
# IMPORTS
###########################
using Base
using DelimitedFiles
using LinearAlgebra
using CUDA
using Test
include("green_tensors_e_m.jl")
########################################################################################################################################
# FUNCTIONS
########################################################################################################################################
@doc raw"""
     solve_system(A,b,solver,verbose)

Solves a system of equations of the type ``Ax=b`` where `matrix` is ``A`` and `vector` is ``b`` using the method `solver` and returns `x`.
`x` can be a 1D column vector or a 2D matrix. In this second case, the system is going to solve each column of the matrix as a different problem (without re-inverting `A`).
The `solver` flag can be set to
- `CPU`: In this case, the system is solved using LAPACK on the CPU.
- `GPU`: In this case, the system is solved using CUSOLVE on the GPU (if available).
"""
function solve_system(A,b,solver,verbose)
    #lapack julia solver
    if solver=="CPU"
        #
        phi=copy(b)
        #
        if verbose
            println("solving with CPU LAPACK solver ...")
        end
        #
        BLAS.set_num_threads(Threads.nthreads())
        LAPACK.gesv!(A,phi)
        #
        return phi
    #CUDA solver
    elseif solver=="GPU"
        #
        if verbose
            println("solving with GPU CUSOLVER solver ...")
        end
        #
        A=CuArray(A)
        b=CuArray(b)
        b=A\b
        return Array(b)
    end
end

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
@doc raw"""
     load_dda_matrix_e(kr,alpha_dl,verbose)

Builds the electric only DDA matrix ``A=[I-G\alpha]`` with dimensionless postitions `kr` (2D array of size ``Nx3``) and dimensionless polarisabilities `alpha_dl` (3D complex array of size ``N\times 3\times 3``containing the polarisability ``3\times 3`` tensor of each dipole, or 1D array of size ``N`` containing the scalar polarizability of each dipole).
Returns ``3N\times 3N`` complex DDA matrix.
"""
function load_dda_matrix_e(kr,alpha_dl,verbose)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println("formatting the equations...")
    end
    #create DDA matrix
    A=Matrix{ComplexF64}(I,3*n,3*n)
    #if scalar polarisability
    if length(alpha_dl)==n
        Threads.@threads for j in 1:n
            for k=1:j-1
                G=GreenTensors.G_e_renorm(kr[j,:],kr[k,:])
                A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(-G*alpha_dl[k])
                A[3*(k-1)+1:3*(k-1)+3,3*(j-1)+1:3*(j-1)+3]=copy(-G*alpha_dl[j])
            end
        end
    else #if tensor polarisabilty
        Threads.@threads for j in 1:n
            for k=1:j-1
                G=GreenTensors.G_e_renorm(kr[j,:],kr[k,:])
                A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(-G*alpha_dl[k,:,:])
                A[3*(k-1)+1:3*(k-1)+3,3*(j-1)+1:3*(j-1)+3]=copy(-G*alpha_dl[j,:,:])
            end
        end
    end
    #return DDA matrix
    return A
end
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
@doc raw"""
     load_dda_matrix_e_m(kr,alpha_e_dl,alpha_m_dl,verbose)

Builds the electric and magnetic DDA matrix ``A=[I-G\alpha]`` with dimensionless postitions `kr` (2D array of size ``N\times 3``) and dimensionless electric and magnetic polarisabilities `alpha_e_dl` and  `alpha_m_dl` (3D complex array of size ``N\times 3\times 3``containing the polarisability ``3\times 3`` tensor of each dipole, or 1D  array of size ``N`` containing the scalar polarizability of each dipole).
Return ``6N\times 6N`` complex DDA matrix.
"""
function load_dda_matrix_e_m(kr,alpha_e_dl,alpha_m_dl,verbose)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println("formatting the equations...")
    end
    #create DDA Matrix
    A=Matrix{ComplexF64}(I,6*n,6*n)
    #
    a_dda=zeros(ComplexF64,6,6)
    #
    if length(alpha_e_dl)==n
        for i=1:n
            for j=1:i-1
                Ge,Gm=GreenTensors.G_em_renorm(kr[i,:],kr[j,:])
                a_dda[1:3,1:3]=-Ge*alpha_e_dl[j]
                a_dda[4:6,4:6]=-Ge*alpha_m_dl[j]
                a_dda[1:3,4:6]=-im*Gm*alpha_m_dl[j]
                a_dda[4:6,1:3]=+im*Gm*alpha_e_dl[j]
                A[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=copy(a_dda)
                a_dda[1:3,1:3]=-Ge*alpha_e_dl[i]
                a_dda[4:6,4:6]=-Ge*alpha_m_dl[i]
                a_dda[1:3,4:6]=+im*Gm*alpha_m_dl[i]
                a_dda[4:6,1:3]=-im*Gm*alpha_e_dl[i]
                A[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=copy(a_dda)
            end
        end
    else 
        for i=1:n
            for j=1:i-1
                Ge,Gm=GreenTensors.G_em_renorm(kr[i,:],kr[j,:])
                a_dda[1:3,1:3]=-Ge*alpha_e_dl[j,:,:]
                a_dda[4:6,4:6]=-Ge*alpha_m_dl[j,:,:]
                a_dda[1:3,4:6]=-im*Gm*alpha_m_dl[j,:,:]
                a_dda[4:6,1:3]=+im*Gm*alpha_e_dl[j,:,:]
                A[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=copy(a_dda)
                a_dda[1:3,1:3]=-Ge*alpha_e_dl[i,:,:]
                a_dda[4:6,4:6]=-Ge*alpha_m_dl[i,:,:]
                a_dda[1:3,4:6]=+im*Gm*alpha_m_dl[i,:,:]
                a_dda[4:6,1:3]=-im*Gm*alpha_e_dl[i,:,:]
                A[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=copy(a_dda)
            end
        end
    end
    return A
end
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
@doc raw"""
     load_dda_matrix_e_m(kr,alpha_tensor,verbose)

Builds the electric and magnetic DDA matrix ``A=[I-G\alpha]`` with dimensionless postitions `kr` (two dimensional arrays of size ``N\times 3``) and dimensionless polarisability `alpha_tensor` (a three-dimensional complex array of size ``N\times 6\times 6``containing the polarisability ``6\times 6`` tensor of each dipole).
Return ``6N\times 6N`` complex DDA matrix
"""
function load_dda_matrix_e_m(kr,alpha_tensor,verbose)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println("formatting the equations...")
    end
    #create DDA Matrix
    A=Matrix{ComplexF64}(I,6*n,6*n)
    #
    a_dda=zeros(ComplexF64,6,6)
    #
    for i=1:n
        for j=1:i-1
            Ge,Gm=GreenTensors.G_em_renorm(kr[i,:],kr[j,:])
            a_dda[1:3,1:3]=-Ge
            a_dda[4:6,4:6]=-Ge
            a_dda[1:3,4:6]=-im*Gm
            a_dda[4:6,1:3]=+im*Gm
            A[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=copy(a_dda*alpha_tensor[j,:,:])
            a_dda[1:3,4:6]=-a_dda[1:3,4:6]
            a_dda[4:6,1:3]=-a_dda[4:6,1:3]
            A[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=copy(a_dda*alpha_tensor[i,:,:])
        end
    end
    return A
end
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

@doc raw"""
    solve_DDA_e(kr,alpha_dl;input_field=nothing,solver="CPU",verbose=true)

Builds and solves the DDA equations with dimensionless input under a given input field, i.e.
```math
\vec{E}_{i}=\vec{E}_{0}(\vec{r}_{i})+\sum^N_{j\neq i}G_e(\vec{r}_i,\vec{r}_j)\alpha_j\vec{E}_j

```
for a group of ``N`` only electric dipoles and returns the incident fields on each of the dipoles.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_dl`: 3D complex array of size ``N\times 3\times 3``containing the dimensionless polarisability ``3\times 3`` tensor of each dipole, or 1D array of size ``N`` containing the scalar polarizability of each dipole.
- `input_field`: 2D complex array of size ``N\times 3`` containing the input field ``E_0`` at each of the dipoles positions.
- `solver`: string that contains the name of the solver that need to be used. For this, check the `DDACore.solve_system` function documentation. By default set to "CPU".
- `verbose`: whether to output informations to the standard output during running or not. By default set to `true`.
#Outputs
- `e_inc`: 2D complex array of size ``N\times 3`` containing the incident fields ``E_inc`` on every dipole.
"""
function solve_DDA_e(kr,alpha_dl;input_field=nothing,solver="CPU",verbose=true)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #generate the matrix
    A=load_dda_matrix_e(kr,alpha_dl,verbose)
    #solving the system
    if input_field===nothing
        E=solve_system(A,Matrix{ComplexF64}(I,3*n,3*n),solver,verbose)
        return E
    else
        if ndims(input_field)==2
            E=solve_system(A,reshape(transpose(input_field),3*n),solver,verbose)
        elseif ndims(input_field)==3
            E=zeros(ComplexF64,3*n,length(input_field[:,1,1]))
            for i=1:length(input_field[:,1,1])
                E[:,1]=reshape(input_field[i,:,:],3*n)
            end
        end
    end
    #reshaping
    if ndims(input_field)==2
        e_inc=transpose(reshape(E,3,n))
    elseif ndims(input_field)==3
        e_inc=zeros(ComplexF64,length(input_field[:,1,1]))
        for i=1:length(input_field[:,1,1])
            e_inc[i,:,:]=transpose(reshape(E[i,:],3,n))
        end
    end
    return e_inc
end
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


@doc raw"""
    solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl;input_field=nothing,solver="CPU",verbose=true)

Builds and solves the DDA equations with dimensionless inputs under a given input field, i.e.
```math
\vec{E}_{i}  =\vec{E}_{0}\left(\vec{r}_{i}\right)+\sum_{j\neq i}G_{E}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{E}^{(j)}\vec{E}_{j}+iG_{M}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{M}^{(j)}\vec{H}_{j}
```
```math
\vec{H}_{i}  =\vec{H}_{0}\left(\vec{r}_{i}\right)+\sum_{j\neq i}-iG_{M}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{E}^{(j)}\vec{E}_{j}+G_{E}\left(\vec{r}_{i},\vec{r}_{j}\right)\alpha_{M}^{(j)}\vec{H}_{j}
```
for a group of ``N`` electric and magnetic dipoles and return the polarisations of every particle and incident fields on every particle.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless positions ``k\vec{r}`` of all the dipoles.
- `alpha_e_dl`: 3D complex array of size ``N\times 3\times 3``containing the dimensionless electric polarisability ``3\times 3`` tensor of every dipole, or 1D array of size ``N`` containing the scalar polarizability of every dipole.
- `alpha_m_dl`: 3D complex array of size ``N\times 3\times 3``containing the dimesnionless magnetic polarisability ``3\times 3`` tensor of every dipole, or 1D array of size ``N`` containing the scalar polarizability of every dipole.
- `input_field`: 2D complex array of size ``N\times 6`` containing the input field ``\phi_{0}=(E_0,H_0)`` at each of the dipoles positions.
- `solver`:string that contains the name of the solver that need to be used. For this, check the `DDACore.solve_system` function documentation. By default set to "CPU".
- `verbose`: whether to output pieces of information to the standard output during running or not. By default set to `true`.
#Outputs
- `e_inc`: 2D complex array of size ``N\times 6`` containing the incident fields ``\phi_{inc}=(E_inc,H_inc)`` on every dipole.
"""
function solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl;input_field=nothing,solver="CPU",verbose=true)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #loading matrix
    A=load_dda_matrix_e_m(kr,alpha_e_dl,alpha_m_dl,verbose)
    #solving the system
    if input_field===nothing
        E=solve_system(A,Matrix{ComplexF64}(I,6*n,6*n),solver,verbose)
        return E
    else
        if ndims(input_field)==2
            E=solve_system(A,reshape(transpose(input_field),6*n),solver,verbose)
        elseif ndims(input_field)==3
            phi=zeros(ComplexF64,6*n,length(input_field[:,1,1]))
            for i=1:length(input_field[:,1,1])
                phi[:,1]=reshape(input_field[i,:,:],6*n)
            end
        end
    end
    #reshaping
    if ndims(input_field)==2
        phi_inc=transpose(reshape(E,6,n))
    elseif ndims(input_field)==3
        phi_inc=zeros(ComplexF64,length(input_field[:,1,1]))
        for i=1:length(input_field[:,1,1])
            phi_inc[i,:,:]=transpose(reshape(E[i,:],6,n))
        end
    end
    return phi_inc
end

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


@doc raw"""
    function solve_DDA_e_m(kr,alpha_dl;input_field=nothing,solver="CPU",verbose=true)

Similar to `solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl;input_field=nothing,solver="CPU",verbose=true)`, but the polarisabilities are ``6\times 6`` complex tensors taking into account both electrric and magnetic behaviour of the particles and its optical activity.
`alpha` is a 3D complex array of size ``N\times 6\times 6``containing the dimesnionless polarisability of each particle.
"""
function solve_DDA_e_m(kr,alpha_dl;input_field=nothing,solver="CPU",verbose=true)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #loading matrix
    A=load_dda_matrix_e_m(kr,alpha_dl,verbose)
    #solving the system
    if input_field===nothing
        E=solve_system(A,Matrix{ComplexF64}(I,6*n,6*n),solver,verbose)
        return E
    else
        if ndims(input_field)==2
            E=solve_system(A,reshape(transpose(input_field),6*n),solver,verbose)
        elseif ndims(input_field)==3
            phi=zeros(ComplexF64,6*n,length(input_field[:,1,1]))
            for i=1:length(input_field[:,1,1])
                phi[:,1]=reshape(input_field[i,:,:],6*n)
            end
        end
    end
    #reshaping
    if ndims(input_field)==2
        phi_inc=transpose(reshape(E,6,n))
    elseif ndims(input_field)==3
        phi_inc=zeros(ComplexF64,length(input_field[:,1,1]))
        for i=1:length(input_field[:,1,1])
            phi_inc[i,:,:]=transpose(reshape(E[i,:],6,n))
        end
    end
    return phi_inc
end


end