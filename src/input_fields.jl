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
    plane_wave_renorm(kr,khat=[0,0,1],e0=[1,0,0])
Computes a simple plane with dimensionless input evaluated at `kr`. `khat` is the direction of propagation and `e0` is the polarization.
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
    plane_wave_e_m(knorm,r,khat=[0,0,1],e0=[1,0,0])
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

@doc raw"""
    plane_wave_e_m_renorm(kr,khat=[0,0,1],e0=[1,0,0])
Computes a simple plane with dimensionless input evaluated at `kr`. `khat` is the direction of propagation and `e0` is the polarization.
The output is a 3d complex vector.
This plane wave is defined as:
```math
\mathbf{E}\left(\mathbf{r}\right)=\mathbf{E}_{0}e^{i\mathbf{k}\cdot\mathbf{r}}
```
"""
function plane_wave_e_m_renorm(kr,khat=[0,0,1],e0=[1,0,0])
    E=exp(im*dot(khat,kr))*e0
    H=cross(khat,E)
    return E,H
end

@doc raw"""
    point_dipole(knorm, E0_const, positions, rd, dip_o)
Function that calculated the electromagnetic field emitted by a point dipole with dipole moment 
```math
\mathrm{dip}_o = \dfrac{1}{\epsilon_0\epsilon} \overrightarrow{\mu}, \quad \mathrm{(see \ equation \ below)}
``` 

Imputs
- `knorm0` is the medium wavevector (scalar)
- `E0_const` is the field intensity (scalar). The modulus of the dipole moment is set to -epsilon_0*epsilon_m-, where -epsilon_0- and -epsilon_m- are the vacuum and medium permittivity, respectively.
- `position` contains the position at which the field is calculated (N x 3 matrix, where -N- is the number of points)
- `rd` is the position of the emitting source/dipole (1 x 3 vector)
- `dip_o` defined the nature of the dipole. If `dip_o` is a scalar then:
    - dip_o = 1 -> elecric dipole along x-axis
    - dip_o = 2 -> elecric dipole along y-axis
    - dip_o = 3 -> elecric dipole along z-axis
    - dip_o = 4 -> magnetic dipole along x-axis
    - dip_o = 5 -> magnetic dipole along y-axis
    - dip_o = 6 -> magnetic dipole along z-axis
if `dip_o` is a 6 x 1 vector then it specifies the dipole moment orentation of the source. 

Outputs
- `E_0i` is the electromagnetic field vector of the field at the requiered positions (6N x 1 vector)

Equation

```math
\mathbf{E}_{\mathbf{\mu}}(\mathbf{r}) = \omega^2 \mu \mu_0 G(\mathbf{r}, \mathbf{r}_0) \overrightarrow{\mu} = k^2 G(\mathbf{r}, \mathbf{r_0}) \dfrac{1}{\epsilon_0\epsilon} \overrightarrow{\mu}
```
with
```math
\mathrm{positions} = \mathbf{r}, \\
\mathrm{rd} = \mathbf{r}_0, \\
\mathrm{E}_{\mathrm{0i}} = \mathbf{E}_{\mathbf{\mu}}(\mathbf{r}).
``` 
"""
function point_dipole(knorm, E0_const, positions, rd, dip_o)
    
    N_points = length(positions[:,1])
    
    G_tensor = zeros(ComplexF64,N_points*6,6)
    G = zeros(ComplexF64,6,6) 

    if length(dip_o) == 1
        dip_oi = dip_o
        dip_o = zeros(6,1)
        dip_o[dip_oi] = 1
    else
        dip_o = dip_o/norm(dip_o) # Ensure that its modulus is equal to one
    end

    for i = 1:N_points  
        Ge, Gm = GreenTensors.G_em(positions[i,:],rd[1,:],knorm)   
        G[:,:] = [Ge im*Gm; -im*Gm Ge]
        G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , 1:6] = copy(G)
    end
	    
    E_0i = knorm^2*G_tensor*dip_o*E0_const

    return E_0i
        
end


@doc raw"""
    point_dipole_dl(knorm, E0_const, kpositions, krd, dip_o)
Function that calculated the electromagnetic field emitted by a point dipole with dipole moment 
```math
\mathrm{dip}_o = \dfrac{1}{\epsilon_0\epsilon} \overrightarrow{\mu}, \quad \mathrm{(see \ equation \ below)}
``` 

Imputs
- `knorm0` is the medium wavevector (scalar)
- `E0_const` is the field intensity (scalar). The modulus of the dipole moment is set to -epsilon_0*epsilon_m-, where -epsilon_0- and -epsilon_m- are the vacuum and medium permittivity, respectively.
- `kposition` contains the position (multiplied by the wavevector) at which the field is calculated (N x 3 matrix, where -N- is the number of points)
- `krd` is the position of the emitting source/dipole (multiplied by the wavevector) (1 x 3 vector)
- `dip_o` defined the nature of the dipole. If `dip_o` is a scalar then:
    - dip_o = 1 -> elecric dipole along x-axis
    - dip_o = 2 -> elecric dipole along y-axis
    - dip_o = 3 -> elecric dipole along z-axis
    - dip_o = 4 -> magnetic dipole along x-axis
    - dip_o = 5 -> magnetic dipole along y-axis
    - dip_o = 6 -> magnetic dipole along z-axis
if `dip_o` is a 6 x 1 vector then it specifies the dipole moment orentation of the source. 

Outputs
- `E_0i` is the electromagnetic field vector of the field at the requiered positions (6N x 1 vector)

Equation

```math
\mathbf{E}_{\mathbf{\mu}}(\mathbf{r}) =  k^2 G(k\mathbf{r}, k\mathbf{r_0}) \dfrac{1}{\epsilon_0\epsilon} \overrightarrow{\mu}
```
with
```math
\mathrm{kpositions} = \mathbf{kr}, \\
\mathrm{krd} = \mathbf{kr}_0, \\
\mathrm{E}_{\mathrm{0i}} = \mathbf{E}_{\mathbf{\mu}}(\mathbf{r}).
``` 
"""
function point_dipole_dl(knorm, E0_const, kpositions, krd, dip_o)
    
    N_points = length(kpositions[:,1])
    
    G_tensor = zeros(ComplexF64,N_points*6,6)
    G = zeros(ComplexF64,6,6)  

    if length(dip_o) == 1
        dip_oi = dip_o
        dip_o = zeros(6,1)
        dip_o[dip_oi] = 1
    else
        dip_o = dip_o/norm(dip_o) # Ensure that its modulus is equal to one
    end

    for i = 1:N_points     
        Ge, Gm = GreenTensors.G_em_renorm(kpositions[i,:],krd[1,:],knorm)   
        G[:,:] = k^3/(6*pi)*[Ge im*Gm; -im*Gm Ge]
        G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , 1:6] = copy(G)
    end
	    
    E_0i = knorm^3/(6*pi)*G_tensor*dip_o*E0_const

    return E_0i
        
end

using Cubature
using SpecialFunctions

@doc raw"""
    besselj2(z)
Bessel function order 2 

#Arguments
- `z`: = argument
"""
function besselj2(z)
    if z == 0
        return 0
    else
        return 2/z*besselj1(z) - besselj0(z)
    end
end

@doc raw"""
    gauss_beam_e_m(rf,k,bw0, maxe = Int(5e3))
Field distribution of a Gaussian beam that propagates along the z-axis and the electric field is polarized along the x-axis (polarized electric).
For another polarization just rotate the field in the .xy-plane. Also, for a polarized magnetic field, exchange E -> Hz and Hz -> -E. 

#Arguments
- `rf`: 2D float array of size ``N\times 3`` containing the positions where the field is calculated (``N`` is the number of positions).
- `k`: scalar with the modulus wavevector.
- `bw0`: float with the beam waist radius.
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_gauss`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function gauss_beam_e_m(rf,k,bw0,maxe=Int(5e3))
    n_rf = length(rf[:,1])
    eh_gauss = zeros(ComplexF64,n_rf,6)
    for i = 1:n_rf
        r = rf[i,:]
        GB(k_i, GB_Q) = begin
            Q = k_i[1]
            kp = sqrt(k^2 - Q^2)
            rp = sqrt(r[1]^2 + r[2]^2)
            phi = atan(r[2],r[1])

            bj0 = besselj0(kp*rp)
            bj1 = besselj1(kp*rp)
            bj2 = besselj2(kp*rp)

            factor = exp(im*Q*r[3])*bw0^2*exp(-bw0^2*kp^2/4)/(2)
            Er_x = Q*factor*bj0
            Er_z = - im*kp*factor*bj1*cos(phi)
            Hr_x = factor/k*1/2*kp^2*bj2*sin(2*phi)
            Hr_y = factor/k*(-1/2*kp^2*bj2*cos(2*phi) + (1/2*kp^2 + Q^2)*bj0  )
            Hr_z = - kp*Q*factor/k*im*bj1*sin(phi)

            GB_Q[1] = real(Er_x)
            GB_Q[2] = imag(Er_x)
            GB_Q[3] = real(Er_z)
            GB_Q[4] = imag(Er_z)
            GB_Q[5] = real(Hr_x)
            GB_Q[6] = imag(Hr_x)
            GB_Q[7] = real(Hr_y)
            GB_Q[8] = imag(Hr_y)
            GB_Q[9] = real(Hr_z)
            GB_Q[10] = imag(Hr_z)
        end
        Q0 = [0]
        Q1 = [k]
        (EHg, erEg) = hcubature(10, GB, Q0, Q1, maxevals = maxe)
        E = [EHg[1] + im*EHg[2], 0, EHg[3] + im*EHg[4]]
        ZH = [EHg[5] + im*EHg[6], EHg[7] + im*EHg[8], EHg[9] + im*EHg[10]]
        eh_gauss[i,:] = [E; ZH]
    end
    return eh_gauss
end

@doc raw"""
    ghermite_beam_e_m(rf,k,bw0,n,m,maxe=Int(5e3))
Field distribution of a Hermite-Gaussian beam that propagates along the z-axis and the electric field is polarized along the x-axis (polarized electric).
For another polarization just rotate the field in the .xy-plane. Also, for a polarized magnetic field, exchange E -> Hz and Hz -> -E. 

#Arguments
- `rf`: 2D float array of size ``N\times 3`` containing the positions where the field is calculated (``N`` is the number of positions).
- `k`: scalar with the modulus wavevector.
- `bw0`: float with the beam waist radius.
- `n`: int with the order of the beam.
- `m`: int with the degree of the beam.
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_hermite`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function ghermite_beam_e_m(rf,k,bw0,n,m,maxe=Int(5e3))
    n_rf = length(rf[:,1])
    eh_hermite = zeros(ComplexF64,n_rf,6)
    for i = 1:n_rf
        r = rf[i,:]
        GB(QT, GB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(k^2 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_kp = bw0^(n+m)*bw0^2*exp(-bw0^2*kp^2/4)/(4*pi)*(im*kx)^n*(im*ky)^m
            factor = exp(im*(kx*r[1] + ky*r[2] + Q*r[3] ))*f_kp
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor/k
            Hr_y = (kx^2 + Q^2)*factor/k
            Hr_z = - ky*Q*factor/k

            GB_QT[1] = real(Er_x)
            GB_QT[2] = imag(Er_x)
            GB_QT[3] = real(Er_z)
            GB_QT[4] = imag(Er_z)
            GB_QT[5] = real(Hr_x)
            GB_QT[6] = imag(Hr_x)
            GB_QT[7] = real(Hr_y)
            GB_QT[8] = imag(Hr_y)
            GB_QT[9] = real(Hr_z)
            GB_QT[10] = imag(Hr_z)
        end
        QT0 = [0, 0]
        QT1 = [k, 2*pi]
        (EHg, erEg) = hcubature(10, GB, QT0, QT1, maxevals = maxe)
        E = [EHg[1] + im*EHg[2], 0, EHg[3] + im*EHg[4]]
        ZH = [EHg[5] + im*EHg[6], EHg[7] + im*EHg[8], EHg[9] + im*EHg[10]]
        eh_hermite[i,:] = [E; ZH]
    end
    return eh_hermite
end

@doc raw"""
    glaguerre_beam_e_m(rf,k,bw0,n,m,maxe=Int(5e3))
Field distribution of a Laguerre-Gaussian beam that propagates along the z-axis and the electric field is polarized along the x-axis (polarized electric).
For another polarization just rotate the field in the .xy-plane. Also, for a polarized magnetic field, exchange E -> Hz and Hz -> -E. 

#Arguments
- `rf`: 2D float array of size ``N\times 3`` containing the positions where the field is calculated (``N`` is the number of positions).
- `k`: scalar with the modulus wavevector.
- `bw0`: float with the beam waist radius.
- `n`: non-negative int with the radial order of the beam.
- `m`: int with the azimuthal order of the beam.
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_hermite`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function glaguerre_beam_e_m(rf,k,bw0,n,m, maxe = Int(5e3))
    n_rf = length(rf[:,1])
    eh_laguerre = zeros(ComplexF64,n_rf,6)
    for i = 1:n_rf
        r = rf[i,:]
        GB(QT, GB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(k^2 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_xy = 0
            for p = 0:m
                f_xy = f_xy + binomial(m,p)*(im*kx)^(m-p)*(-1*ky)^p 
            end
            f_kp = k^n*bw0^(2*n+m)*exp(im*k*r[3])*bw0^2*exp(-bw0^2*kp^2/4)/(4*pi)*(im*(Q-k))^n*f_xy
            factor = exp(im*(kx*r[1] + ky*r[2] + Q*r[3] ))*f_kp
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor/k
            Hr_y = (kx^2 + Q^2)*factor/k
            Hr_z = - ky*Q*factor/k

            GB_QT[1] = real(Er_x)
            GB_QT[2] = imag(Er_x)
            GB_QT[3] = real(Er_z)
            GB_QT[4] = imag(Er_z)
            GB_QT[5] = real(Hr_x)
            GB_QT[6] = imag(Hr_x)
            GB_QT[7] = real(Hr_y)
            GB_QT[8] = imag(Hr_y)
            GB_QT[9] = real(Hr_z)
            GB_QT[10] = imag(Hr_z)
        end
        QT0 = [0, 0]
        QT1 = [k, 2*pi]
        (EHg, erEg) = hcubature(10, GB, QT0, QT1, maxevals = maxe)
        E = [EHg[1] + im*EHg[2], 0, EHg[3] + im*EHg[4]]
        ZH = [EHg[5] + im*EHg[6], EHg[7] + im*EHg[8], EHg[9] + im*EHg[10]]
        eh_laguerre[i,:] = [E; ZH]
    end
    return eh_laguerre
end

@doc raw"""
    gauss_beam_e(rf,k,bw0, maxe = Int(5e3))
Field distribution of a Gaussian beam that propagates along the z-axis and the electric field is polarized along the x-axis (polarized electric).
For another polarization just rotate the field in the .xy-plane. Also, for a polarized magnetic field, exchange E -> Hz and Hz -> -E. 

#Arguments
- `rf`: 2D float array of size ``N\times 3`` containing the positions where the field is calculated (``N`` is the number of positions).
- `k`: scalar with the modulus wavevector.
- `bw0`: float with the beam waist radius.
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_gauss`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function gauss_beam_e(rf,k,bw0,maxe=Int(5e3))
    n_rf = length(rf[:,1])
    e_gauss = zeros(ComplexF64,n_rf,3)
    for i = 1:n_rf
        r = rf[i,:]
        GB(k_i, GB_Q) = begin
            Q = k_i[1]
            kp = sqrt(k^2 - Q^2)
            rp = sqrt(r[1]^2 + r[2]^2)
            phi = atan(r[2],r[1])

            bj0 = besselj0(kp*rp)
            bj1 = besselj1(kp*rp)
            bj2 = besselj2(kp*rp)

            factor = exp(im*Q*r[3])*bw0^2*exp(-bw0^2*kp^2/4)/(2)
            Er_x = Q*factor*bj0
            Er_z = - im*kp*factor*bj1*cos(phi)

            GB_Q[1] = real(Er_x)
            GB_Q[2] = imag(Er_x)
            GB_Q[3] = real(Er_z)
            GB_Q[4] = imag(Er_z)
        end
        Q0 = [0]
        Q1 = [k]
        (Eg, erEg) = hcubature(4, GB, Q0, Q1, maxevals = maxe)
        E = [Eg[1] + im*Eg[2], 0, Eg[3] + im*Eg[4]]
        e_gauss[i,:] = E
    end
    return e_gauss
end

@doc raw"""
    ghermite_beam_e(rf,k,bw0,n,m,maxe=Int(5e3))
Field distribution of a Hermite-Gaussian beam that propagates along the z-axis and the electric field is polarized along the x-axis (polarized electric).
For another polarization just rotate the field in the .xy-plane. Also, for a polarized magnetic field, exchange E -> Hz and Hz -> -E. 

#Arguments
- `rf`: 2D float array of size ``N\times 3`` containing the positions where the field is calculated (``N`` is the number of positions).
- `k`: scalar with the modulus wavevector.
- `bw0`: float with the beam waist radius.
- `n`: int with the order of the beam.
- `m`: int with the degree of the beam.
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_hermite`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function ghermite_beam_e(rf,k,bw0,n,m,maxe=Int(5e3))
    n_rf = length(rf[:,1])
    e_hermite = zeros(ComplexF64,n_rf,3)
    for i = 1:n_rf
        r = rf[i,:]
        GB(QT, GB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(k^2 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_kp = bw0^(n+m)*bw0^2*exp(-bw0^2*kp^2/4)/(4*pi)*(im*kx)^n*(im*ky)^m
            factor = exp(im*(kx*r[1] + ky*r[2] + Q*r[3] ))*f_kp
            Er_x = Q*factor
            Er_z = - kx*factor

            GB_QT[1] = real(Er_x)
            GB_QT[2] = imag(Er_x)
            GB_QT[3] = real(Er_z)
            GB_QT[4] = imag(Er_z)
        end
        QT0 = [0, 0]
        QT1 = [k, 2*pi]
        (Eg, erEg) = hcubature(4, GB, QT0, QT1, maxevals = maxe)
        E = [Eg[1] + im*Eg[2], 0, Eg[3] + im*Eg[4]]
        e_hermite[i,:] = E
    end
    return e_hermite
end

@doc raw"""
    glaguerre_beam_e(rf,k,bw0,n,m,maxe=Int(5e3))
Field distribution of a Laguerre-Gaussian beam that propagates along the z-axis and the electric field is polarized along the x-axis (polarized electric).
For another polarization just rotate the field in the .xy-plane. Also, for a polarized magnetic field, exchange E -> Hz and Hz -> -E. 

#Arguments
- `rf`: 2D float array of size ``N\times 3`` containing the positions where the field is calculated (``N`` is the number of positions).
- `k`: scalar with the modulus wavevector.
- `bw0`: float with the beam waist radius.
- `n`: non-negative int with the radial order of the beam.
- `m`: int with the azimuthal order of the beam.
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_hermite`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function glaguerre_beam_e(rf,k,bw0,n,m, maxe = Int(5e3))
    n_rf = length(rf[:,1])
    e_laguerre = zeros(ComplexF64,n_rf,3)
    for i = 1:n_rf
        r = rf[i,:]
        GB(QT, GB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(k^2 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_xy = 0
            for p = 0:m
                f_xy = f_xy + binomial(m,p)*(im*kx)^(m-p)*(-1*ky)^p 
            end
            f_kp = k^n*bw0^(2*n+m)*exp(im*k*r[3])*bw0^2*exp(-bw0^2*kp^2/4)/(4*pi)*(im*(Q-k))^n*f_xy
            factor = exp(im*(kx*r[1] + ky*r[2] + Q*r[3] ))*f_kp
            Er_x = Q*factor
            Er_z = - kx*factor

            GB_QT[1] = real(Er_x)
            GB_QT[2] = imag(Er_x)
            GB_QT[3] = real(Er_z)
            GB_QT[4] = imag(Er_z)
        end
        QT0 = [0, 0]
        QT1 = [k, 2*pi]
        (Eg, erEg) = hcubature(4, GB, QT0, QT1, maxevals = maxe)
        E = [Eg[1] + im*Eg[2], 0, Eg[3] + im*Eg[4]]
        e_laguerre[i,:] = E
    end
    return e_laguerre
end

end
