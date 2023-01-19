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

#=
Function that calculated the electromagnetic field emitted by a point dipole with dipole moment "\dfrac{1}{\epsilon_0\epsilon} \bm \mu = "dip_o"" (see equation below)

Imputs
- "knorm0" is the medium wavevector (scalar)
- "E0_const" is the field intensity (scalar). The modulus of the dipole moment is set to "(epsilon_0*epsilon_m)", where "epsilon_0" and "epsilon_m" are the vacuum and medium permittivity, respectively.
- "position" contains the position at which the field is calculated ("N x 3" matrix, where "N" is the number of points)
- "rd" is the position of the emitting source/dipole ("1 x 3" vector)
- "dip_o" defined the "nature of the dipole". If "dip_o" is a scalar then
dip_o = 1 -> elecric dipole along "x" axis
dip_o = 2 -> elecric dipole along "y" axis
dip_o = 3 -> elecric dipole along "z" axis
dip_o = 4 -> magnetic dipole along "x" axis
dip_o = 5 -> magnetic dipole along "y" axis
dip_o = 6 -> magnetic dipole along "z" axis
if "dip_o" is "6 x 1" vector then specified the dipole moment orentation of the source. 

Outputs
- "E_0i" is the electromagnetic field vector of the field at the requiered positions ("6N x 1" vector)

Equation

\E_{\bm \mu}(\r) = \omega^2 \mu \mu_0 \G(\r, \r_0) \bm \mu = k^2 \G(\r_0, \r_0) \dfrac{1}{\epsilon_0\epsilon} \bm \mu

\dfrac{1}{\epsilon_0\epsilon} \bm \mu = "dip_o"
\G(\r_0, \r_0) = "G"

=#

function point_dipole(knorm, E0_const, positions, rd, dip_o)
    
    N_points = length(positions[:,1])
    
    G_tensor = zeros(ComplexF64,N_points*6,6) 

# "dip_o" is the dipole orientation ("1 x 6" vector. A electric dipole along the "y" axis would be "dip_o = [0,1,0,0,0,0]", while a magnetic dipole along the "z" axis would be "dip_o = [0,0,0,0,0,1]") 

    if length(dip_o) == 1
        dip_oi = dip_o
        dip_o = zeros(6,1)
        dip_o[dip_oi] = 1
    else
        dip_o = dip_o/norm(dip_o) # Ensure that its modulus is equal to one
    end

    for i = 1:N_points     
        G = knorm^2*GreenTensors.G_em_s(positions[i,:],rd[1,:],knorm)
        G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , 1:6] = copy(G)
    end
	    
    E_0i = G_tensor*dip_o*E0_const

    return E_0i
        
end

# Same as before, but the possicional arguments are given in dimmensiola units
# kpositions = knorm*positions
# krd = knorm*krd

function point_dipole_dl(knorm, E0_const, kpositions, krd, dip_o)
    
    N_points = length(kpositions[:,1])
    
    G_tensor = zeros(ComplexF64,N_points*6,6) 

# "dip_o" is the dipole orientation ("1 x 6" vector. A electric dipole along the "y" axis would be "dip_o = [0,1,0,0,0,0]", while a magnetic dipole along the "z" axis would be "dip_o = [0,0,0,0,0,1]") 

    if length(dip_o) == 1
        dip_oi = dip_o
        dip_o = zeros(6,1)
        dip_o[dip_oi] = 1
    else
        dip_o = dip_o/norm(dip_o) # Ensure that its modulus is equal to one
    end

    for i = 1:N_points     
        G = GreenTensors.G_em_s_renorm(kpositions[i,:],krd[1,:])
        G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , 1:6] = copy(G)
    end
	    
    E_0i = G_tensor*dip_o*E0_const

    return E_0i
        
end

