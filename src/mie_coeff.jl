module MieCoeff
#code to compute the mie coefficients author Augustin Muster (translate from python code of Diego Romero Abujetas)
#imports
using Base
using LinearAlgebra
using SpecialFunctions

function spherical_jn(n,x,deriv)
    if deriv==0
        return sphericalbesselj(n, x)
    else
        return sphericalbesselj(n-1, x)-(n+1)/x*sphericalbesselj(n, x)
    end
end

function spherical_yn(n,x,deriv)
    if deriv==0
        return sphericalbessely(n, x)
    else
        return sphericalbessely(n-1, x)-(n+1)/x*sphericalbessely(n, x)
    end
end
# utlilities functions
function psi(n, x)
    return x * spherical_jn(n, x, 0)
end

function diff_psi(n, x)
    return spherical_jn(n, x, 0) + x * spherical_jn(n, x, 1)
end

function xi(n, x)
    return x * (spherical_jn(n, x, 0) + 1*im * spherical_yn(n, x, 0))
end

function diff_xi(n, x)
    return (spherical_jn(n, x, 0) + 1*im * spherical_yn(n, x, 0)) + x * (spherical_jn(n, x, 1) + 1*im * spherical_yn(n, x, 1))
end

@doc raw"""
    mie_an(knorm, a, eps, eps_h, n)

Computes the `n`-th mie coefficient ``a_n`` of a sphere of radius `a` with dielectric constant `eps`, in a medium with dielectric constant `eps_h`. `vac_knorm` is wavenumber in the medium. 
Retruns a complex scalar.
"""
function mie_an(knorm, a, eps, eps_h,n)

    mt = sqrt(eps)/sqrt(eps_h)
    alpha = knorm * a 
    beta = knorm * a *mt


    return (mt * diff_psi(n, alpha) * psi(n, beta) - psi(n, alpha) * diff_psi(n,beta)) / (mt * diff_xi(n, alpha) * psi(n, beta) - xi(n, alpha) * diff_psi(n, beta))
end

@doc raw"""
    mie_bn(knorm, a, eps, eps_h, n)

Computes the `n`-th mie coefficient ``b_n`` of a sphere of radius `a` with dielectric constant `eps`, in a medium with dielectric constant `eps_h`. `vac_knorm` is wavenumber in the medium.
Returns a complex scalar. 
"""
function mie_bn(knorm, a, eps, eps_h, n)
    mt = sqrt(eps)/sqrt(eps_h)
    alpha = knorm * a 
    beta = knorm * a * mt

    return (mt * psi(n, alpha) * diff_psi(n, beta) - diff_psi(n, alpha) * psi(n,beta)) / (mt * xi(n, alpha) * diff_psi(n, beta) - diff_xi(n, alpha) * psi(n, beta))
end

@doc raw"""
    mie_scattering_cross_section(knorm,a,eps,eps_h;cutoff=50)

Computes the scattering cross section ``C_{sca}`` of a sphere of radius `a` with dielectric constant `eps` in a medium with dielectric constant `eps_h`. `knorm` is the wavenumber in the medium. For this, we use:

```math
C_{sca} =\frac{2\pi}{k^2}\sum^{\infty}_{n=1}\left(2n+1\right)\left(|a_n|^2+|b_n|^2\right)
```
The infinite sum is computed only for terms under the `cutoff` variable (by default set to 50).

Returns a float with units of surface.

"""
function mie_scattering_cross_section(knorm,a,eps,eps_h;cutoff=50)
    sum=0
    for i=1:cutoff
        sum=sum+(2*i+1)*(abs2(mie_an(knorm, a, eps, eps_h, i))+abs2(mie_bn(knorm, a, eps, eps_h, i)))
    end
    return 2*pi/(knorm)^2*sum
end

@doc raw"""
    mie_extinction_cross_section(knorm,a,eps,eps_h;cutoff=50)

Computes the extinction cross section ``C_{sca}`` of a sphere of radius `a` with dielectric constant `eps` in a medium with dielectric constant `eps_h`. `knorm` is the wavenumber in the medium. For this, we use:

```math
C_{ext} =\frac{2\pi}{k^2}\sum^{\infty}_{n=1}\left(2n+1\right)Re\left(a_n+b_n\right)
```
The infinite sum is computed only for terms under the `cutoff` variable (by default set to 50).

Returns a float with units of surface.

"""
function mie_extinction_cross_section(knorm,a,eps,eps_h;cutoff=50)
    sum=0
    for i=1:cutoff
        sum=sum+(2*i+1)*real(mie_an(knorm, a, eps, eps_h, i)+mie_bn(knorm, a, eps, eps_h, i))
    end
    return 2*pi/(knorm)^2*sum
end


@doc raw"""
    mie_absorption_cross_section(knorm,a,eps,eps_h;cutoff=50)

Computes the extinction cross section ``C_{sca}`` of a sphere of radius `a` with dielectric constant `eps` in a medium with dielectric constant `eps_h`. `knorm` is the wavenumber in the medium. For this, we use:

```math
C_{abs} =C_{ext}-C_{sca}
```
The infinite sum is computed only for terms under the `cutoff` variable (by default set to 50).

Returns a float with units of surface.

"""
function mie_absorption_cross_section(knorm,a,eps,eps_h;cutoff=50)
    sum=0
    for i=1:cutoff
        an=mie_an(knorm, a, eps, eps_h, i)
        bn=mie_bn(knorm, a, eps, eps_h, i)
        sum=sum+(2*i+1)*(real(an+bn)-(abs2(an))+abs2(bn))
    end
    return 2*pi/(knorm)^2*sum
end

end