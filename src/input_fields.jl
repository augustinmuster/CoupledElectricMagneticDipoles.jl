module InputFields
###########################
# IMPORTS
#######m####################
using Base
using LinearAlgebra
include("green_tensors_e_m.jl")
###########################
# FUNCTIONS
#######m####################

@doc raw"""
    plane_wave_e(kr;khat=[0,0,1],e0=[1,0,0])
Computes a simple plane with dimensionless input evaluated at `kr`. `khat` is the direction of propagation and `e0` is the polarization.
`kr` is a ``N\times 3`` float array. The output is a ``N\times 3`` complex array represeinting the electric field.
This plane wave is defined as:
```math
\mathbf{E}\left(\mathbf{r}\right)=\mathbf{E}_{0}e^{i\mathbf{k}\cdot\mathbf{r}}
```
"""
function plane_wave_e(kr;khat=[0,0,1],e0=[1,0,0])
    n=length(kr[:,1])
    E0=zeros(ComplexF64,n,3)
    for i=1:n
        E0[i,:]=exp(im*dot(khat,kr[i,:]))*e0
    end
    return E0
end


@doc raw"""
    plane_wave_e_m(kr;khat=[0,0,1],e0=[1,0,0])
Computes a simple plane with dimensionless input evaluated at `kr`. `khat` is the direction of propagation and `e0` is the polarization.
`kr` is a ``N\times 3`` float array. The output is a ``N\times 6`` complex array representing the electric and magnetic field.
This plane wave is defined as:
```math
\mathbf{E}\left(\mathbf{r}\right)=\mathbf{E}_{0}e^{i\mathbf{k}\cdot\mathbf{r}}
```
"""
function plane_wave_e_m(kr;khat=[0,0,1],e0=[1,0,0])
    n=length(kr[:,1])
    phi=zeros(ComplexF64,n,6)
    for i=1:n
        phi[i,1:3]=exp(im*dot(khat,kr[i,:]))*e0
        phi[i,4:6]=cross(khat,phi[i,1:3])
    end
    return phi
end

@doc raw"""
    point_dipole_e_m(krf, krd, dip, e0_const=1)
Function that calculated the electromagnetic field emitted by a point dipole.

#Arguments
- `krf`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the field is calculated.
- `krd`: 2D float array of size ``1\times 3`` containing the dimentionless positions ``k\vec{r_d}`` where source is located.
- `dip`: integer defining the dipole moment (``dip = 1`` is an electric x-dipole, ``dip = 2`` an elctric y-dipole...) or float array of size 6 with the desired dipole moment of the dipole.  
#Outputs
- `e_dipole`: complex array with the electromagnetic field.
"""
function point_dipole_e_m(krf, krd, dip, e0_const=1)
    n_r0 = length(krf[:,1])
    G_tensor = zeros(ComplexF64,n_r0*6,6)
    if length(dip) == 1  && dip < 7 && dip > 0
        dip_o = dip
        dip = zeros(6)
        dip[dip_o] = 1
    elseif length(dip) == 6
        dip = dip/norm(dip) # Ensure that its modulus is equal to one
    else
        dip = zeros(6)
        println("dip should be an integer (between 1 and 6) or a vector of length 6")
    end
    for i = 1:n_r0  
        Ge, Gm = GreenTensors.G_em_renorm(krf[i,:],krd[1,:])   
        G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [Ge im*Gm; -im*Gm Ge]
    end
    e_dipole = G_tensor*dip*e0_const
    e_dipole = transpose(reshape(e_dipole,6,n_r0))
    return e_dipole      
end

@doc raw"""
    point_dipole_e(krf, krd, dip, e0_const=1)
Function that calculated the electromagnetic field emitted by a point dipole.

#Arguments
- `krf`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the field is calculated.
- `krd`: 2D float array of size ``1\times 3`` containing the dimentionless positions ``k\vec{r_d}`` where source is located.
- `dip`: integer defining the dipole moment (``dip = 1`` is an electric x-dipole, ``dip = 2`` an elctric y-dipole...) or float array of size 3 with the desired dipole moment of the dipole.  
#Outputs
- `e_dipole`: complex array with the electromagnetic field.
"""
function point_dipole_e(krf, krd, dip, e0_const=1)
    n_r0 = length(krf[:,1])
    G_tensor = zeros(ComplexF64,n_r0*3,3)
    if length(dip) == 1  && dip < 4 && dip > 0
        dip_o = dip
        dip = zeros(3)
        dip[dip_o] = 1
    elseif length(dip) == 3
        dip = dip/norm(dip) # Ensure that its modulus is equal to one
    else
        dip = zeros(6)
        println("dip should be an integer (between 1 and 6) or a vector of length 6")
    end
    for i = 1:n_r0  
        Ge, Gm = GreenTensors.G_em_renorm(krf[i,:],krd[1,:])   
        G_tensor[3 * (i-1) + 1:3 * (i-1) + 3, :] = Ge
    end
    e_dipole = G_tensor*dip*e0_const
    e_dipole = transpose(reshape(e_dipole,3,n_r0))
    return e_dipole      
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
