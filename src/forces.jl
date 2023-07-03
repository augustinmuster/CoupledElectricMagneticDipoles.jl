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
    force_e_m(k,kr,alpha_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
It computes the optical forces for deterministics imputs fields.

#Arguments
- `k`: float with the wavevector.
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_dl`: complex array containing the dimensionless polarisability.
- `Ainv`: (inverse) DDA matrix ``[I - G*alpha]^(-1)``.
- `e_0`: 2D complex array of size ``N\times 6`` containing the external imput field.
- `dxe_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*x`` argument of the external imput field.
- `dye_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*y`` argument of the external imput field.
- `dze_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*z`` argument of the external imput field.
#Outputs
- `real(fx)`: float array of Size ``N`` with the value of the force along the ``x``-axis at each dipole.
- `real(fy)`: float array of Size ``N`` with the value of the force along the ``y``-axis at each dipole.
- `real(fz)`: float array of Size ``N`` with the value of the force along the ``z``-axis at each dipole.
"""
function force_e_m(k,kr,alpha_dl, Ainv, e_0, dxe_0, dye_0, dze_0)

    eps0 = 1/2*k*8.8541878128e-12

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

    Threads.@threads for i=1:n_particles
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

    fx = eps0*sum(p.*dxe_inc,dims=2)
    fy = eps0*sum(p.*dye_inc,dims=2)
    fz = eps0*sum(p.*dze_inc,dims=2)
    if length(fz) == 1
        return real(fx[1]), real(fy[1]), real(fz[1])
    end
    return real(fx[:,1]), real(fy[:,1]), real(fz[:,1])
end

@doc raw"""
    force_e(k,kr,alpha_e_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
It computes the optical forces for deterministics imputs fields.

#Arguments
- `k`: float with the wavevector.
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex array containing the dimensionless electric polarisability.
- `Ainv`: (inverse) DDA matrix ``[I - G*alpha]^(-1)``.
- `e_0`: 2D complex array of size ``N\times 3`` containing the external imput field.
- `dxe_0`: 2D complex array of size ``N\times 3`` containing the derivative respect the ``k*x`` argument of the external imput field.
- `dye_0`: 2D complex array of size ``N\times 3`` containing the derivative respect the ``k*y`` argument of the external imput field.
- `dze_0`: 2D complex array of size ``N\times 3`` containing the derivative respect the ``k*z`` argument of the external imput field.
#Outputs
- `real(fx)`: float array of Size ``N`` with the value of the force along the ``x``-axis at each dipole.
- `real(fy)`: float array of Size ``N`` with the value of the force along the ``y``-axis at each dipole.
- `real(fz)`: float array of Size ``N`` with the value of the force along the ``z``-axis at each dipole.
"""
function force_e(k,kr,alpha_e_dl, Ainv, e_0, dxe_0, dye_0, dze_0)

    eps0 = 1/2*k*8.8541878128e-12*(4*pi)/k^3

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

    Threads.@threads for i=1:n_particles
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

    fx = eps0*sum(p.*dxe_inc,dims=2)
    fy = eps0*sum(p.*dye_inc,dims=2)
    fz = eps0*sum(p.*dze_inc,dims=2)
    if length(fz) == 1
        return real(fx[1]), real(fy[1]), real(fz[1])
    end
    return real(fx[:,1]), real(fy[:,1]), real(fz[:,1])
end

@doc raw"""
    LDOS_sc(knorm, alpha, Ainv, pos, rd, dip_o)
It Computes partial local density of states (LDOS) by the scattering cross section (sc)

Imputs
- `knorm` = wavenumber
- `alpha` = polarizability of the particles (6N x 6N matrix, where -N- is the number of dipoles)
- `Ainv` = (inverse) DDA matrix (6N x 6N matrix, [I - k^2*G*alpha]^(-1))
- `pos` = position of the dipoles (N x 3 matrix, where -N- is the number of points)
- `rd` = emitting dipole position, i. e., position where the LDOS is calculated
- `dip_o` defined the nature of the dipole (see -point_dipole- function). Therefore, it defines the component of the LDOS that is calculated 

Outputs
- `LDOS` is a scalar with the value of the partial LDOS

Equation
```math
LDOS = \sigma^{sca}/\sigma^{sca}_{0}
```
"""
function LDOS_sc(knorm, alpha, Ainv, pos, rd, dip_o)

    r = [pos; rd]
    Np = length(r[:,1])
      
    E_0i = InputFields.point_dipole(knorm, 1, pos, rd, dip_o)
    
    if length(dip_o) == 1
        dip_oi = dip_o
        dip_o = zeros(6,1)
        dip_o[dip_oi] = 1
    else
        dip_o = dip_o/norm(dip_o) # Ensure that its modulus is equal to one
    end
    
    Pd = alpha*Ainv*E_0i
    Pd = [Pd; dip_o]
    Pd = reshape(Pd,6,Np)
    
    sum_sca = 0.
    
    for i=1:Np
        sum_sca = sum_sca + 1/3*dot(Pd[1:3,i],Pd[1:3,i]) + 1/3*dot(Pd[4:6,i],Pd[4:6,i])
        for j=(i+1):Np
            sum_sca=sum_sca+real(transpose(Pd[1:3,j])*(imag(GreenTensors.G_e_renorm(knorm*r[j,:],knorm*r[i,:]))*conj(Pd[1:3,i])) + transpose(Pd[4:6,j])*(imag(GreenTensors.G_e_renorm(knorm*r[j,:],knorm*r[i,:]))*conj(Pd[4:6,i])))
            sum_sca=sum_sca+imag(-transpose(conj(Pd[1:3,i]))*imag(GreenTensors.G_m_renorm(knorm*r[i,:],knorm*r[j,:]))*Pd[4:6,j]    +   transpose(conj(Pd[1:3,j]))*imag(GreenTensors.G_m_renorm(knorm*r[i,:],knorm*r[j,:]))*Pd[4:6,i])
        end
    end
    
    
    LDOS = real(sum_sca)*3#/(1/3*sum(abs.(dip_o).^2)) # Imaginary part of the dipole component (LDOS)

    return LDOS

end


end
