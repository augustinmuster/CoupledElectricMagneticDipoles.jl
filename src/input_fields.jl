module InputFields
###########################
# IMPORTS
#######m####################
using Base
using LinearAlgebra
###########################
# FUNCTIONS
#######m####################
@doc raw"""
    plane_wave(knorm,r,khat=[0,0,1],e0=[1,0,0])
Computes a simple plane wave of wavenumber `knorm` evaluated at position `r` function. `khat` is the direction of propagation and `e0` is the polarization.
The output is a 3d complex vector.
This plane wave is defined as:
```math
\mathbf{E}\left(\mathbf{r}\right)=\mathbf{E}_{0}e^{i\mathbf{k}\cdot\mathbf{r}}
```
"""
function plane_wave(knorm,r,khat=[0,0,1],e0=[1,0,0])
    return exp(im*dot(knorm*khat,r))*e0
end

@doc raw"""
    plane_wave(kr,khat=[0,0,1],e0=[1,0,0])
Computes a simple plane with dimensionless input evaluated at "kr". `khat` is the direction of propagation and `e0` is the polarization.
The output is a 3d complex vector.
This plane wave is defined as:
```math
\mathbf{E}\left(\mathbf{r}\right)=\mathbf{E}_{0}e^{i\mathbf{k}\cdot\mathbf{r}}
```
"""
function plane_wave_renorm(kr,khat=[0,0,1],e0=[1,0,0])
    return exp(im*dot(khat,kr))*e0
end

@doc raw"""
    plane_wave(knorm,r,khat=[0,0,1],e0=[1,0,0])
Computes a simple plane wave of wavenumber `knorm` evaluated at position `r` function. `khat` is the direction of propagation and `e0` is the polarization.
The output is a 3d complex vector.
This plane wave is defined as:
```math
\mathbf{E}\left(\mathbf{r}\right)=\mathbf{E}_{0}e^{i\mathbf{k}\cdot\mathbf{r}}
```
"""
function plane_wave_e_m(knorm,r,khat=[0,0,1],e0=[1,0,0])
    E=exp(im*dot(knorm*khat,r))*e0
    H=cross(khat,E)
    return E,H
end

#*************************************************
#PLANE WAVE ELECTRIC AND MAGNETIC, RENORM
#INPUTS:   position vector,direction of the wave vector,polaristaion,
#OUTPUT: plane wave vector
#*************************************************
function plane_wave_e_m_renorm(kr,khat=[0,0,1],e0=[1,0,0])
    E=exp(im*dot(khat,kr))*e0
    H=cross(khat,E)
    return E,H
end
end