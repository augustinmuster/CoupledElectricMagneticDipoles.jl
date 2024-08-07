module Alphas

export alpha0_cube,alpha0_sphere,alpha0_volume,alpha_e_m_mie,alpha_radiative,renorm_alpha,denorm_alpha,dispatch_e,dispatch_e_m
###########################
# IMPORTS
###########################
using Base
using LinearAlgebra
using ..MieCoeff
###########################
# FUNCTIONS
###########################

@doc raw"""
    alpha0_volume(V,eps,eps_h)
Computes the quasistatic polarizability of any isotropic object with volume `V` and permittivity `eps` in a host medium with permittivity `eps_h`. Permittivities can be scalars or ``3\times 3`` tensors. 
Outputs a float with units of volume.
"""
function alpha0_volume(V,eps,eps_h)
    return 3*V*(eps-eps_h)*inv(eps+2*eps_h)
end

@doc raw"""
    alpha0_cube(L,eps,eps_h)
Computes the quasistatic polarizability tensor of a cube of side `L` and permittivity `eps` in a host medium with permittivity `eps_h`. Permittivities can be scalars or ``3\times 3`` tensors. 
Outputs a ``3\times 3`` float matrix with units of volume.
"""
function alpha0_cube(L,eps,eps_h)
    return alpha0_volume(L^3,eps,eps_h)
end

@doc raw"""
    alpha0_sphere(a,eps,eps_h)
Computes the quasistatic polarizability of a sphere of radius `a` and permittivity `eps` in a host medium with permittivity `eps_h`.  Permittivities can be scalars or ``3\times 3`` tensors.
Outputs a float with units of volume.
"""
function alpha0_sphere(a,eps,eps_h)
    return alpha0_volume(4/3*pi*a^3,eps,eps_h)
end


@doc raw"""
    alpha_radiative(alpha0,knorm)
Applies the radiative correction to the polarizability matrix or scalar `alpha0` (with units of volume).
Outputs a complex dimensionless scalar or a (3x3) complex dimensionless matrix.
"""
function alpha_radiative(alpha0,knorm)
    if ndims(alpha0)==0
        return knorm^3/4/pi*inv(inv(alpha0)-im*(knorm^3)/(6*pi))
    else
        id=[1 0 0;0 1 0;0 0 1]
        return knorm^3/4/pi*inv(inv(alpha0)-im*(knorm^3)/(6*pi)*id)
    end
end


@doc raw"""
    alpha_e_m_mie(ka,eps,eps_h)
Computes the electric and magnetic polarizabilities from the mie coefficients ``a_1`` and  ``b_1`` of a particle with size parameter `ka` (wave number times radius), and permittivity `eps`, in a host medium with permittivity `eps_h`.
Outputs two dimensionless scalars that are respectively the electric and the magnetic polarizability.
"""
function alpha_e_m_mie(ka,eps,eps_h)
    a1,b1=MieCoeff.mie_ab1(ka, eps, eps_h; mu= 1.,mu_h=1.)
    alpha_e=im*1.5*a1
    alpha_m=im*1.5*b1
    return alpha_e,alpha_m
end

@doc raw"""
    dispatch_e_m(alpha_e_dl,alpha_m_dl,n_particles)
Creates an iterable with the polarizability of all particles in order to facilitate the syntax for multuplying a Green's tensor by the polarizability of a particle.

# Arguments
- `alpha_e_dl`: electric polarizability, see the Alphas module's documentation for the accepted formats.
- `alpha_m_dl`: magnetic polarizability, see the Alphas module's documentation for the accepted formats.
- `n_particles`: number of particles (integer)
# Outputs
- `alp_e`: iterable electric polarizability
- `alp_m`: iterable magnetic polarizability
"""
function dispatch_e_m(alpha_e_dl,alpha_m_dl,n_particles)
    if length(alpha_e_dl) == length(alpha_m_dl) == 1 # If alpha is the same scalar for all particles
        alp_e = fill(alpha_e_dl,n_particles)
        alp_m = fill(alpha_m_dl,n_particles)

    elseif length(alpha_e_dl) == length(alpha_m_dl) == n_particles # If alpha is the same scalar for all particles
        alp_e = alpha_e_dl
        alp_m = alpha_m_dl

    elseif length(alpha_e_dl) == length(alpha_m_dl) == 3^2 && size(alpha_e_dl,1) == 3 # If alpha is the same tensor for all particles
        alp_e = fill(alpha_e_dl,n_particles)
        alp_m = fill(alpha_m_dl,n_particles)
    
    elseif length(alpha_e_dl) == length(alpha_m_dl) == 3^2 && size(alpha_e_dl,1) == 1 && ndims(alpha_e_dl) == 3 # If alpha is the same tensor for all particles 
        alp_e = fill(alpha_e_dl[1,:,:],n_particles)
        alp_m = fill(alpha_m_dl[1,:,:],n_particles)

    elseif length(alpha_e_dl) == length(alpha_m_dl) == n_particles*3^2  # If alpha is a tensor
        alp_e = [alpha_e_dl[i,:,:] for i in 1:size(alpha_e_dl,1)]
        alp_m = [alpha_m_dl[i,:,:] for i in 1:size(alpha_m_dl,1)]

    elseif length(alpha_e_dl) != length(alpha_m_dl)   
        println("The length of the electric and magnetic polarizability must match")
    else
        println("non-implemented")  
    end

    return alp_e, alp_m
end

@doc raw"""
    dispatch_e_m(alpha_dl,n_particles)
Creates an iterable with the polarizability of all particles in order to facilitate the syntax for multuplying a Green's tensor by the polarizability of a particle.

# Arguments
- `alpha_dl`: polarizability 6x6 complex matrix, see the Alphas module's documentation for the accepted formats.
- `n_particles`: number of particles (integer)

# Outputs
- `alp`: iterable polarizability
"""
function dispatch_e_m(alpha_dl,n_particles)
    if length(alpha_dl) == 6^2 && size(alpha_dl,1) == 6  # If alpha is the same tensor for all particles 
        alp = fill(alpha_dl,n_particles)
        
    elseif length(alpha_dl) == 6^2 && size(alpha_dl,1) == 1 && ndims(alpha_dl) == 3 # If alpha is the same tensor for all particles  
        alp = fill(alpha_dl[1,:,:],n_particles)

    elseif length(alpha_dl) == n_particles*6^2  # If alpha is a tensor
        alp = [alpha_dl[i,:,:] for i in 1:size(alpha_dl,1)]

    else
        println("non-implemented")  
    end

    return alp
end

@doc raw"""
    dispatch_e(alpha_e_dl,n_particles)
Creates an iterable with the polarizability of all particles in order to facilitate the syntax for multuplying a Green's tensor by polarizability of a particle.

# Arguments
- `alpha_e_dl`: electric polarizability, see the Alphas module's documentation for the accepted formats.
- `n_particles`: number of particles (integer)

# Outputs
- `alp_e`: iterable electric polarizability
"""
function dispatch_e(alpha_e_dl,n_particles)
    if length(alpha_e_dl) == 1 # If alpha is the same scalar for all particles
        alp_e = fill(alpha_e_dl,n_particles)

    elseif length(alpha_e_dl) == n_particles # If alpha is the same scalar for all particles
        alp_e = alpha_e_dl
        
    elseif length(alpha_e_dl) == 3^2 && size(alpha_e_dl,1) == 3 # If alpha is the same tensor for all particles 
        alp_e = fill(alpha_e_dl,n_particles)

    elseif length(alpha_e_dl) == 3^2 && size(alpha_e_dl,1) == 1 && ndims(alpha_e_dl) == 3 # If alpha is the same tensor for all particles 
        alp_e = fill(alpha_e_dl[1,:,:],n_particles)

    elseif length(alpha_e_dl)  == n_particles*3^2  # If alpha is a tensor
        alp_e = [alpha_e_dl[i,:,:] for i in 1:size(alpha_e_dl,1)]

    else
        println("non-implemented")  
    end

    return alp_e
end

@doc raw"""
    renorm_alpha(knorm,alpha)
Renormalizes any polarizability `alpha` with units of volume in a dimensionless polarizability by multiplying by ``k^3/4\pi``. `knorm` is the wave number in the medium.
"""
function renorm_alpha(knorm,alpha)
    return alpha.*(knorm^3/4/pi)
end

@doc raw"""
    denorm_alpha(knorm,alpha)
Denormalizes any dimensionless polarizability `alpha` in a polarizability with units of volume by multiplying by ``4\pi /k^3``. `knorm` is the wave number in the medium.
"""
function denorm_alpha(knorm,alpha)
    return alpha.*(4*pi/knorm^3)
end


end
