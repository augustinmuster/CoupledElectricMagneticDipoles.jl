"""
CoupledElectricMagneticDipoles.jl : Alphas Module
The aim of this module is to provides several functions to compute polarisabilities.
Author: Augustin Muster, November 2022, augustin@must-r.com
"""

module Alphas
###########################
# IMPORTS
###########################
using Base
using LinearAlgebra
using PyCall
include("mie_coeff.jl")
###########################
# FUNCTIONS
###########################
@doc raw"""
    depolarisation_tensor(lx,ly,lz,Vn)
Compute the (3x3) depolarisation tensor ``L`` of a rectangular box of sides `lx`, `ly` and `lz` and volume `Vn`.
"""
function depolarisation_tensor(lx,ly,lz,Vn)
    xx=2/pi*atan(1/lx^2*Vn/sqrt(lx^2+ly^2+lz^2))
    yy=2/pi*atan(1/ly^2*Vn/sqrt(lx^2+ly^2+lz^2))
    zz=2/pi*atan(1/lz^2*Vn/sqrt(lx^2+ly^2+lz^2))
    return[xx 0 0;0 yy 0;0 0 zz]
end

@doc raw"""
    alpha_0(e,e_m,Ln,Vn)
Compute the electric quasistatic polarisabilitiy ``\alpha_0`` of a particle with dielectric constant  `e`, volume `Vn` and depolarisation tensor `Ln` in a medium with dielectric constant `e_m`.
The output is a (3x3) tensor computed as follow:

```math
\alpha_{0}=(\epsilon-\epsilon_{m}I)((\epsilon-\epsilon_{m}I)+L^{-1}\epsilon_{m})^{-1}L^{-1}V
```
"""
function alpha_0(e,e_m,Ln,Vn)
    id=[1 0 0;0 1 0;0 0 1]
    Lni=inv(Ln)
    return (e*id-e_m*id)*inv((e*id-e_m*id)+Lni*e_m)*Lni*Vn
end


@doc raw"""
     alpha_radiative(a0,knorm)
Apply the radiative correction to the polarisability tensor `a0`
The output is a (3x3) tensor computed as follow:

```math
\alpha=\left(\alpha_{0}^{-1}-i\frac{k{{}^3}}{6\pi}\right)^{-1}
```
"""
function alpha_radiative(a0,knorm)
    id=[1 0 0;0 1 0;0 0 1]
    return inv(inv(a0)-im*(knorm^3)/(6*pi)*id)
end

@doc raw"""
     alpha_e_m_mie(knorm,vac_knorm,a,n,n_m)
Computes the electric and magnetic polarisabilities from the mie coefficients of a particle of refractive indes `n` and radius `a` in a medium with refractive index `n_m` with wavenumber `knorm` amd vacuum wavenumber `vac_knorm`.
It outputs two scalars which are computed as

```math
\tilde{\alpha}_{E} =i\left(\frac{k^{3}}{6\pi}\right)^{-1}a_{1},\ \tilde{\alpha}_{M} =i\left(\frac{k^{3}}{6\pi}\right)^{-1}b_{1}
```
"""
function alpha_e_m_mie(knorm,vac_knorm,a,n,n_m)
    a1=Mie_an(vac_knorm, a, n, n_m, 1)
    b1=Mie_bn(vac_knorm, a, n, n_m, 1)
    alpha_e=im*(6*pi)/(knorm^3)*a1
    alpha_m=im*(6*pi)/(knorm^3)*b1
    return alpha_e,alpha_m
end

@doc raw"""
     alpha_e_m_mie_renorm(knorm,vac_knorm,a,n,n_m)
Computes the electric and magnetic polarisabilities in the renormalized system of units from the mie coefficients of a particle of refractive indes `n` and radius `a` in a medium with refractive index `n_m` with wavenumber `knorm` amd vacuum wavenumber `vac_knorm`.
It outputs two scalars which are computed as

```math
\alpha_{E} =\frac{k^{3}\tilde{\alpha}_{E}}{4\pi}=i\frac{3}{2}a_{1},\ \alpha_{M} =\frac{k^{3}\tilde{\alpha}_{M}}{4\pi}=i\frac{3}{2}b_{1}
```
"""
function alpha_e_m_mie_renorm(vac_knorm,a,n,n_m)
    a1=Mie_an(vac_knorm, a, n, n_m, 1)
    b1=Mie_bn(vac_knorm, a, n, n_m, 1)
    alpha_e=im*1.5*a1
    alpha_m=im*1.5*b1
    return alpha_e,alpha_m
end
end