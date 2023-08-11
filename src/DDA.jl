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
include("alpha.jl")
########################################################################################################################################
# FUNCTIONS
########################################################################################################################################
@doc raw"""
     solve_system(A,b,solver,verbose)

Solves a system of equations of the type ``Ax=b`` using the method `solver` and returns `x`.
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
     load_dda_matrix_e(kr,alpha_e_dl,verbose)

Builds the electric only DDA matrix ``A=[I-G\alpha]`` with dimensionless postitions `kr` (2D array of size ``Nx3``) and dimensionless polarisabilities `alpha_e_dl` (see foramt rules in Alphas module).
Returns ``3N\times 3N`` complex DDA matrix.
"""
function load_dda_matrix_e(kr,alpha_e_dl,verbose)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println("formatting the equations...")
    end
    #create DDA matrix
    A=Matrix{ComplexF64}(I,3*n,3*n)
    #dispatch alphas
    alpha_e_dl=Alphas.dispatch_e(alpha_e_dl,n)
    #load matrix
    for j in 1:n
        for k=1:j-1
            G=GreenTensors.G_e_renorm(kr[j,:],kr[k,:])
            A[3*(j-1)+1:3*(j-1)+3,3*(k-1)+1:3*(k-1)+3]=copy(-G*alpha_e_dl[k])
            A[3*(k-1)+1:3*(k-1)+3,3*(j-1)+1:3*(j-1)+3]=copy(-G*alpha_e_dl[j])
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

Builds the electric and magnetic DDA matrix ``A=[I-G\alpha]`` with dimensionless postitions `kr` (2D array of size ``N\times 3``) and dimensionless electric and magnetic polarisabilities `alpha_e_dl` and  `alpha_m_dl` (see format rules in the Alphas module).
Returns the ``6N\times 6N`` complex DDA matrix.
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
    #dispactch alpha
    alpha_e_dl,alpha_m_dl=Alphas.dispatch_e_m(alpha_e_dl,alpha_m_dl,n)
    #load matrix
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
    return A
end
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
@doc raw"""
     load_dda_matrix_e_m(kr,alpha_tensor,verbose)

Builds the electric and magnetic DDA matrix ``A=[I-G\alpha]`` with dimensionless postitions `kr` (two dimensional arrays of size ``N\times 3``) and dimensionless polarisability `alpha_tensor` (see format rules in the Alphas module).
Return ``6N\times 6N`` complex DDA matrix
"""
function load_dda_matrix_e_m(kr,alpha_dl,verbose)
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
    #dispatch alpha
    alpha_tensor=Alphas.dispatch_e_m(alpha_dl,n)
    #
    for i=1:n
        for j=1:i-1
            Ge,Gm=GreenTensors.G_em_renorm(kr[i,:],kr[j,:])
            a_dda[1:3,1:3]=-Ge
            a_dda[4:6,4:6]=-Ge
            a_dda[1:3,4:6]=-im*Gm
            a_dda[4:6,1:3]=+im*Gm
            A[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=copy(a_dda*alpha_tensor[j])
            a_dda[1:3,4:6]=-a_dda[1:3,4:6]
            a_dda[4:6,1:3]=-a_dda[4:6,1:3]
            A[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=copy(a_dda*alpha_tensor[i])
        end
    end
    return A
end
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

@doc raw"""
    solve_DDA_e(kr,alpha_e_dl;input_field=nothing,solver="CPU",verbose=true)

Builds and solves the DDA equations under a given input field for a group of ``N`` only electric dipoles and returns the incident fields on each of the dipoles. 

# Arguments

- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `input_field`: 2D complex array of size ``N\times 3`` containing the electric input field ``\mathbf{E}_0(\mathbf{r}_i)`` at the position of each dipole. It can also be a 3D array of size ``N_f\times N\times 3``, allowing to solve the problem for several input fields without re-inverting the matrix. This is a keyword argument. If ```input_field=nothing``, the output of the function will be the inverse of the DDA matrix.
- `solver`: string that contains the name of the solver that need to be used. For this, check the `DDACore.solve_system` function documentation. By default set to "CPU".
- `verbose`: whether to output informations to the standard output during running or not. By default set to `true`.

# Outputs
Depending on the value of `input field`, it can be:

- `e_inc`: 2D complex array of size ``N\times 3`` containing the incident electric field ``\mathbf{E}_{i}`` on each dipole. if `input_field` is a 2D array.
- `phi_inc`: 3D complex array of size ``N_f\times N\times 3`` containing the incident electric field ``\mathbf{E}_{i}`` on each dipole for each input field, if `input_field` is a 3D array.
- `Ainv`: complex matrix of size "3N\times 3N", if `input_field=nothing`.
"""
function solve_DDA_e(kr,alpha_e_dl;input_field=nothing,solver="CPU",verbose=true)
    #number of point dipoles
    n=length(kr[:,1])
    #logging
    if verbose
        println()
        println("number of dipoles: ",n)
    end
    #generate the matrix
    A=load_dda_matrix_e(kr,alpha_e_dl,verbose)
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

Builds and solves the DDA equations with dimensionless inputs under a given input field for a group of ``N`` electric and magnetic dipoles and return the polarisations of every particle and incident fields on every particle.

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `input_field`: 2D complex array of size ``N\times 6`` containing the electric and magnetic input field ``\mathbf{\phi}=(\mathbf{E}_0(\mathbf{r}_i),\mathbf{H}_0(\mathbf{r}_i))`` at the position of each dipole. It can also be a 3D array of size ``N_f\times N\times 3``, allowing to solve the problem for several input fields without re-inverting the matrix. This is a keyword argument. If ```input_field=nothing`` (default value), the output of the function will be the inverse of the DDA matrix.
- `solver`:string that contains the name of the solver that need to be used. For this, check the `DDACore.solve_system` function documentation. By default set to "CPU".
- `verbose`: whether to output pieces of information to the standard output during running or not. By default set to `true`.

# Outputs
Depending on the value of `input field`, it can be:

- `phi_inc`: 2D complex array of size ``N\times 6`` containing the incident electric and magnetic field ``\mathbf{\phi}=(\mathbf{E}_i,\mathbf{H}_i)`` on each dipole, if `input_field` is a 2D array.
- `phi_inc`: 3D complex array of size ``N_f\times N\times 6`` containing the incident electric and magnetic field ``\mathbf{\phi}=(\mathbf{E}_i,\mathbf{H}_i)`` on each dipole for each input field, if `input_field` is a 3D array.
- `Ainv`: complex matrix of size "6N\times 6N", if `input_field=nothing`.
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

Same as `solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl;input_field=nothing,solver="CPU",verbose=true)`, but the electric and magnetic polarizabilities of each dipole are given by a single 6x6 complex matrix.  See the Alphas module documentation for accepted formats.
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