module Forces
###########################
# IMPORTS
###########################
using Base
using LinearAlgebra
include("green_tensors_e_m.jl")
include("alpha.jl")
###########################
# FUNCTIONS
###########################

@doc raw"""
    force_e_m(kr,alpha_e_dl, alpha_m_dl, Ainv, e_0, dxe_0, dye_0, dze_0)

Computes the optical forces on a system made out of electric and magnetic dipoles for deterministic input fields. The output force has units of the input
electric field squared. To get unit of forces, it is neccesary to multiply by a factor ``\epsilon_0\epsilon_r 4\pi/k^2``, taking care that the units of
the field, the vacuun permittivity and the wavevector are in accordance.

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `Ainv`: (inverse) DDA matrix.
- `e_0`: 2D complex array of size ``N\times 6`` containing the external imput field.
- `dxe_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*x`` argument of the external imput field.
- `dye_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*y`` argument of the external imput field.
- `dze_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*z`` argument of the external imput field.

# Outputs
- `real(fx)`: float array of Size ``N`` with the value of the force along the ``x``-axis at each dipole.
- `real(fy)`: float array of Size ``N`` with the value of the force along the ``y``-axis at each dipole.
- `real(fz)`: float array of Size ``N`` with the value of the force along the ``z``-axis at each dipole.
"""
function force_e_m(kr,alpha_e_dl, alpha_m_dl, Ainv, e_0, dxe_0, dye_0, dze_0)

    n_particles = length(kr[:,1])
    n_e_0 = length(e_0[:,1])
    n_dxe_0 = length(dxe_0[:,1])
    n_dye_0 = length(dye_0[:,1])
    n_dze_0 = length(dze_0[:,1])

    if n_particles != n_e_0 || n_particles != n_dxe_0 || n_e_0 != n_dxe_0 || n_e_0 != n_dye_0 || n_e_0 != n_dze_0
        println("the dimension of the filed must coincided with the number of particles")
        return 0
    end

    e_0 = reshape(transpose(e_0),n_particles*6,)
    dxe_0 = reshape(transpose(dxe_0),n_particles*6,)
    dye_0 = reshape(transpose(dye_0),n_particles*6,)
    dze_0 = reshape(transpose(dze_0),n_particles*6,)
    e_inc = Ainv*e_0
    
    dxG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    dyG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    dzG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    p = zeros(ComplexF64,n_particles*6,)

    alpha_e_dl,alpha_m_dl=Alphas.dispatch_e_m(alpha_e_dl,alpha_m_dl,n_particles)

    for i=1:n_particles
        p[6*(i-1)+1:6*(i-1)+6] = [alpha_e_dl[i]*e_inc[6*(i-1)+1:6*(i-1)+3];alpha_m_dl[i]*e_inc[6*(i-1)+4:6*(i-1)+6]]
        for j=1:i-1
            dxGe, dxGm = GreenTensors.dxG_em_renorm(kr[i,:],kr[j,:])
            dxG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dxGe*alpha_e_dl[j] im*dxGm*alpha_m_dl[j]; -im*dxGm*alpha_e_dl[j] dxGe*alpha_m_dl[j]]
            dxG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dxGe*alpha_e_dl[i] -im*dxGm*alpha_m_dl[i]; im*dxGm*alpha_e_dl[i] dxGe*alpha_m_dl[i]]
            dyGe, dyGm = GreenTensors.dyG_em_renorm(kr[i,:],kr[j,:])
            dyG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dyGe*alpha_e_dl[j] im*dyGm*alpha_m_dl[j]; -im*dyGm*alpha_e_dl[j] dyGe*alpha_m_dl[j]]
            dyG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dyGe*alpha_e_dl[i] -im*dyGm*alpha_m_dl[i]; im*dyGm*alpha_e_dl[i] dyGe*alpha_m_dl[i]]
            dzGe, dzGm = GreenTensors.dzG_em_renorm(kr[i,:],kr[j,:])
            dzG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dzGe*alpha_e_dl[j] im*dzGm*alpha_m_dl[j]; -im*dzGm*alpha_e_dl[j] dzGe*alpha_m_dl[j]]
            dzG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dzGe*alpha_e_dl[i] -im*dzGm*alpha_m_dl[i]; im*dzGm*alpha_e_dl[i] dzGe*alpha_m_dl[i]]
        end
    end

    dxe_inc = dxe_0 + dxG_alp*e_inc
    dye_inc = dye_0 + dyG_alp*e_inc
    dze_inc = dze_0 + dzG_alp*e_inc

    p = conj(transpose(reshape(p,6,n_particles)))
    dxe_inc = transpose(reshape(dxe_inc,6,n_particles))
    dye_inc = transpose(reshape(dye_inc,6,n_particles))
    dze_inc = transpose(reshape(dze_inc,6,n_particles))

    fx = sum(p.*dxe_inc,dims=2)/2
    fy = sum(p.*dye_inc,dims=2)/2
    fz = sum(p.*dze_inc,dims=2)/2
    if length(fz) == 1
        return real(fx[1]), real(fy[1]), real(fz[1])
    end
    return real(fx[:,1]), real(fy[:,1]), real(fz[:,1])
end

@doc raw"""
    force_e_m(kr,alpha_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
Same as `force_e_m(knorm,kr,alpha_e_dl, alpha_m_dl, Ainv, e_0, dxe_0, dye_0, dze_0)`, but the electric and magnetic polarizabilities of each dipole are given by a single 6x6 complex matrix.  See the Alphas module documentation for accepted formats.
The output force has units of the input electric field squared. To get unit of forces, it is neccesary to multiply by a factor ``\epsilon_0\epsilon_r 4\pi/k^2``, 
taking care that the units of the field, the vacuun permittivity and the wavevector are in accordance.
"""
function force_e_m(kr,alpha_dl, Ainv, e_0, dxe_0, dye_0, dze_0)

    n_particles = length(kr[:,1])
    n_e_0 = length(e_0[:,1])
    n_dxe_0 = length(dxe_0[:,1])
    n_dye_0 = length(dye_0[:,1])
    n_dze_0 = length(dze_0[:,1])

    if n_particles != n_e_0 || n_particles != n_dxe_0 || n_e_0 != n_dxe_0 || n_e_0 != n_dye_0 || n_e_0 != n_dze_0
        println("the dimension of the filed must coincided with the number of particles")
        return 0
    end

    e_0 = reshape(transpose(e_0),n_particles*6,)
    dxe_0 = reshape(transpose(dxe_0),n_particles*6,)
    dye_0 = reshape(transpose(dye_0),n_particles*6,)
    dze_0 = reshape(transpose(dze_0),n_particles*6,)
    e_inc = Ainv*e_0
    
    dxG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    dyG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    dzG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    p = zeros(ComplexF64,n_particles*6,)

    alpha_dl=Alphas.dispatch_e_m(alpha_dl,n_particles)

    for i=1:n_particles
        p[6*(i-1)+1:6*(i-1)+6] = alpha_dl[i]*e_inc[6*(i-1)+1:6*(i-1)+6]
        for j=1:i-1
            dxGe, dxGm = GreenTensors.dxG_em_renorm(kr[i,:],kr[j,:])
            dxG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dxGe im*dxGm; -im*dxGm dxGe]*alpha_dl[j]
            dxG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dxGe -im*dxGm; im*dxGm dxGe]*alpha_dl[i]
            dyGe, dyGm = GreenTensors.dyG_em_renorm(kr[i,:],kr[j,:])
            dyG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dyGe im*dyGm; -im*dyGm dyGe]*alpha_dl[j]
            dyG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dyGe -im*dyGm; im*dyGm dyGe]*alpha_dl[i]
            dzGe, dzGm = GreenTensors.dzG_em_renorm(kr[i,:],kr[j,:])
            dzG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dzGe im*dzGm; -im*dzGm dzGe]*alpha_dl[j]
            dzG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dzGe -im*dzGm; im*dzGm dzGe]*alpha_dl[i]
        end
    end

    dxe_inc = dxe_0 + dxG_alp*e_inc
    dye_inc = dye_0 + dyG_alp*e_inc
    dze_inc = dze_0 + dzG_alp*e_inc

    p = conj(transpose(reshape(p,6,n_particles)))
    dxe_inc = transpose(reshape(dxe_inc,6,n_particles))
    dye_inc = transpose(reshape(dye_inc,6,n_particles))
    dze_inc = transpose(reshape(dze_inc,6,n_particles))

    fx = sum(p.*dxe_inc,dims=2)/2
    fy = sum(p.*dye_inc,dims=2)/2
    fz = sum(p.*dze_inc,dims=2)/2
    if length(fz) == 1
        return real(fx[1]), real(fy[1]), real(fz[1])
    end
    return real(fx[:,1]), real(fy[:,1]), real(fz[:,1])
end

@doc raw"""
    force_e(kr,alpha_e_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
Computes the optical forces on a system made out of electric dipoles for deterministic input fields. The output force has units of the input
electric field squared. To get unit of forces, it is neccesary to multiply by a factor ``\epsilon_0\epsilon_r 4\pi/k^2``, taking care that the units of
the field, the vacuun permittivity and the wavevector are in accordance.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `Ainv`: (inverse) DDA matrix.
- `e_0`: 2D complex array of size ``N\times 3`` containing the external imput field.
- `dxe_0`: 2D complex array of size ``N\times 3`` containing the derivative respect the ``k*x`` argument of the external imput field.
- `dye_0`: 2D complex array of size ``N\times 3`` containing the derivative respect the ``k*y`` argument of the external imput field.
- `dze_0`: 2D complex array of size ``N\times 3`` containing the derivative respect the ``k*z`` argument of the external imput field.
#Outputs
- `real(fx)`: float array of Size ``N`` with the value of the force along the ``x``-axis at each dipole.
- `real(fy)`: float array of Size ``N`` with the value of the force along the ``y``-axis at each dipole.
- `real(fz)`: float array of Size ``N`` with the value of the force along the ``z``-axis at each dipole.
"""
function force_e(kr,alpha_e_dl, Ainv, e_0, dxe_0, dye_0, dze_0)

    n_particles = length(kr[:,1])
    n_e_0 = length(e_0[:,1])
    n_dxe_0 = length(dxe_0[:,1])
    n_dye_0 = length(dye_0[:,1])
    n_dze_0 = length(dze_0[:,1])

    if n_particles != n_e_0 || n_particles != n_dxe_0 || n_e_0 != n_dxe_0 || n_e_0 != n_dye_0 || n_e_0 != n_dze_0
        println("the dimension of the filed must coincided with the number of particles")
        return 0
    end

    e_0 = reshape(transpose(e_0),n_particles*3,)
    dxe_0 = reshape(transpose(dxe_0),n_particles*3,)
    dye_0 = reshape(transpose(dye_0),n_particles*3,)
    dze_0 = reshape(transpose(dze_0),n_particles*3,)
    e_inc = Ainv*e_0
    
    dxG_alp = zeros(ComplexF64,n_particles*3,n_particles*3)
    dyG_alp = zeros(ComplexF64,n_particles*3,n_particles*3)
    dzG_alp = zeros(ComplexF64,n_particles*3,n_particles*3)
    p = zeros(ComplexF64,n_particles*3,)

    alpha_e_dl = Alphas.dispatch_e(alpha_e_dl,n_particles)

    for i=1:n_particles
        p[3*(i-1)+1:3*(i-1)+3] = alpha_e_dl[i]*e_inc[3*(i-1)+1:3*(i-1)+3]
        for j=1:i-1
            dxGe = GreenTensors.dxG_e_renorm(kr[i,:],kr[j,:])
            dxG_alp[3*(i-1)+1:3*(i-1)+3,3*(j-1)+1:3*(j-1)+3]=dxGe*alpha_e_dl[j]
            dxG_alp[3*(j-1)+1:3*(j-1)+3,3*(i-1)+1:3*(i-1)+3]=-dxGe*alpha_e_dl[i]
            dyGe = GreenTensors.dyG_e_renorm(kr[i,:],kr[j,:])
            dyG_alp[3*(i-1)+1:3*(i-1)+3,3*(j-1)+1:3*(j-1)+3]=dyGe*alpha_e_dl[j]
            dyG_alp[3*(j-1)+1:3*(j-1)+3,3*(i-1)+1:3*(i-1)+3]=-dyGe*alpha_e_dl[i]
            dzGe = GreenTensors.dzG_e_renorm(kr[i,:],kr[j,:])
            dzG_alp[3*(i-1)+1:3*(i-1)+3,3*(j-1)+1:3*(j-1)+3]=dzGe*alpha_e_dl[j]
            dzG_alp[3*(j-1)+1:3*(j-1)+3,3*(i-1)+1:3*(i-1)+3]=-dzGe*alpha_e_dl[i]
        end
    end

    dxe_inc = dxe_0 + dxG_alp*e_inc
    dye_inc = dye_0 + dyG_alp*e_inc
    dze_inc = dze_0 + dzG_alp*e_inc

    p = conj(transpose(reshape(p,3,n_particles)))
    dxe_inc = transpose(reshape(dxe_inc,3,n_particles))
    dye_inc = transpose(reshape(dye_inc,3,n_particles))
    dze_inc = transpose(reshape(dze_inc,3,n_particles))

    fx = sum(p.*dxe_inc,dims=2)/2
    fy = sum(p.*dye_inc,dims=2)/2
    fz = sum(p.*dze_inc,dims=2)/2
    if length(fz) == 1
        return real(fx[1]), real(fy[1]), real(fz[1])
    end
    return real(fx[:,1]), real(fy[:,1]), real(fz[:,1])
end


end
