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
function point_dipole_e_m(krf, krd, dip; e0_const=1)
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
function point_dipole_e(krf, krd, dip; e0_const=1)
    n_r0 = length(krf[:,1])
    G_tensor = zeros(ComplexF64,n_r0*3,3)
    if length(dip) == 1  && dip < 4 && dip > 0
        dip_o = dip
        dip = zeros(3)
        dip[dip_o] = 1
    elseif length(dip) == 3
        dip = dip/norm(dip) # Ensure that its modulus is equal to one
    else
        dip = zeros(3)
        println("dip should be an integer (between 1 and 3) or a vector of length 3")
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
For another polarization just rotate the field in the xy-plane. Also, for a polarized magnetic field, exchange E -> ZH and H -> -E. 

#Arguments
- `rf`: 2D float array of size ``N\times 3`` containing the positions where the field is calculated (``N`` is the number of positions).
- `k`: scalar with the modulus wavevector.
- `bw0`: float with the beam waist radius.
- `kmax`: float setting the limit of the radial integration (it shoud be ``kmax < k``).
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_gauss`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function gauss_beam_e_m(rf,k,bw0; kmax = nothing, maxe=Int(5e3))
    if kmax===nothing
        kmax = k
    elseif kmax>k
        kmax = k
    end
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
        Q1 = [kmax]
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
- `kmax`: float setting the limit of the radial integration (it shoud be ``kmax < k``).
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_hermite`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function ghermite_beam_e_m(rf,k,bw0,n,m; kmax = nothing, maxe=Int(5e3))
    if kmax===nothing
        kmax = k
    elseif kmax>k
        kmax = k
    end
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
        QT1 = [kmax, 2*pi]
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
- `kmax`: float setting the limit of the radial integration (it shoud be ``kmax < k``).
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_hermite`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function glaguerre_beam_e_m(rf,k,bw0,n,m; kmax = nothing, maxe = Int(5e3))
    if kmax===nothing
        kmax = k
    elseif kmax>k
        kmax = k
    end
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
        QT1 = [kmax, 2*pi]
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
- `kmax`: float setting the limit of the radial integration (it shoud be ``kmax < k``).
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_gauss`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function gauss_beam_e(rf,k,bw0; kmax = nothing, maxe=Int(5e3))
    if kmax===nothing
        kmax = k
    elseif kmax>k
        kmax = k
    end
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
        Q1 = [kmax]
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
- `kmax`: float setting the limit of the radial integration (it shoud be ``kmax < k``).
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_hermite`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function ghermite_beam_e(rf,k,bw0,n,m; kmax = nothing, maxe=Int(5e3))
    if kmax===nothing
        kmax = k
    elseif kmax>k
        kmax = k
    end
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
        QT1 = [kmax, 2*pi]
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
- `kmax`: float setting the limit of the radial integration (it shoud be ``kmax < k``).
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `eh_hermite`: 2D complex array of size ``N\times 6`` with the value of the field at every position.
"""
function glaguerre_beam_e(rf,k,bw0,n,m; kmax = nothing, maxe = Int(5e3))
    if kmax===nothing
        kmax = k
    elseif kmax>k
        kmax = k
    end
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
        QT1 = [kmax, 2*pi]
        (Eg, erEg) = hcubature(4, GB, QT0, QT1, maxevals = maxe)
        E = [Eg[1] + im*Eg[2], 0, Eg[3] + im*Eg[4]]
        e_laguerre[i,:] = E
    end
    return e_laguerre
end

# Derivative of the fields

@doc raw"""
    d_plane_wave_e(kr;khat=[0,0,1],e0=[1,0,0])
Computes the derivaties of a plane with dimensionless input evaluated at `kr`. `khat` is the direction of propagation and `e0` is the polarization.
`kr` is a ``N\times 3`` float array. The output is a ``N\times 3`` complex array represeinting the electric field.
This plane wave is defined as:
```math
\frac{\partial}{\partial k x}\mathbf{E}\left(\mathbf{r}\right)= i \hat{k}_x \mathbf{E}_{0}e^{i\mathbf{k}\cdot\mathbf{r}}
```
"""
function d_plane_wave_e(kr;khat=[0,0,1],e0=[1,0,0])
    n=length(kr[:,1])
    dxE0=zeros(ComplexF64,n,3)
    dyE0=zeros(ComplexF64,n,3)
    dzE0=zeros(ComplexF64,n,3)
    khat = khat/norm(khat)
    for i=1:n
        E0fac = exp(im*dot(khat,kr[i,:]))*e0*im
        dxE0[i,:]=E0fac*khat[1]
        dyE0[i,:]=E0fac*khat[2]
        dzE0[i,:]=E0fac*khat[3]
    end
    return dxE0, dyE0, dzE0
end


@doc raw"""
    d_plane_wave_e_m(kr;khat=[0,0,1],e0=[1,0,0])
Computes the derivatie of a plane with dimensionless input evaluated at `kr`. `khat` is the direction of propagation and `e0` is the polarization.
`kr` is a ``N\times 3`` float array. The output is a ``N\times 6`` complex array representing the electric and magnetic field.
"""
function d_plane_wave_e_m(kr;khat=[0,0,1],e0=[1,0,0])
    n=length(kr[:,1])
    dxphi=zeros(ComplexF64,n,6)
    dyphi=zeros(ComplexF64,n,6)
    dzphi=zeros(ComplexF64,n,6)
    khat = khat/norm(khat)
    for i=1:n
        phifac = exp(im*dot(khat,kr[i,:]))*e0*im
        dxphi[i,1:3]=phifac*khat[1]
        dxphi[i,4:6]=cross(khat,dxphi[i,1:3])
        dyphi[i,1:3]=phifac*khat[2]
        dyphi[i,4:6]=cross(khat,dyphi[i,1:3])
        dzphi[i,1:3]=phifac*khat[3]
        dzphi[i,4:6]=cross(khat,dzphi[i,1:3])
    end
    return dxphi, dyphi, dzphi
end

@doc raw"""
    d_point_dipole_e_m(krf, krd, dip, e0_const=1)
Function that calculated the (adimensional) derivative of the electromagnetic field emitted by a point dipole.

#Arguments
- `krf`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the field is calculated.
- `krd`: 2D float array of size ``1\times 3`` containing the dimentionless positions ``k\vec{r_d}`` where source is located.
- `dip`: integer defining the dipole moment (``dip = 1`` is an electric x-dipole, ``dip = 2`` an elctric y-dipole...) or float array of size 6 with the desired dipole moment of the dipole.  
#Outputs
- `dxe_dipole`: complex array with the derivative of the electromagnetic field with respect to `k*x`.
- `dye_dipole`: complex array with the derivative of the electromagnetic field with respect to `k*y`.
- `dze_dipole`: complex array with the derivative of the electromagnetic field with respect to `k*z`.
"""
function d_point_dipole_e_m(krf, krd, dip; e0_const=1)
    n_r0 = length(krf[:,1])
    dxG_tensor = zeros(ComplexF64,n_r0*6,6)
    dyG_tensor = zeros(ComplexF64,n_r0*6,6)
    dzG_tensor = zeros(ComplexF64,n_r0*6,6)
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
        dxGe, dxGm = GreenTensors.dxG_em_renorm(krf[i,:],krd[1,:])   
        dxG_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [dxGe im*dxGm; -im*dxGm dxGe]
        dyGe, dyGm = GreenTensors.dyG_em_renorm(krf[i,:],krd[1,:])   
        dyG_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [dyGe im*dyGm; -im*dyGm dyGe]
        dzGe, dzGm = GreenTensors.dzG_em_renorm(krf[i,:],krd[1,:])   
        dzG_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [dzGe im*dzGm; -im*dzGm dzGe]
    end
    dxe_dipole = dxG_tensor*dip*e0_const
    dxe_dipole = transpose(reshape(dxe_dipole,6,n_r0))
    dye_dipole = dyG_tensor*dip*e0_const
    dye_dipole = transpose(reshape(dye_dipole,6,n_r0))
    dze_dipole = dzG_tensor*dip*e0_const
    dze_dipole = transpose(reshape(dze_dipole,6,n_r0))

    return dxe_dipole, dye_dipole, dze_dipole        
end

@doc raw"""
    d_point_dipole_e(krf, krd, dip, e0_const=1)
Function that calculated the (adimensional) derivative of the electromagnetic field emitted by a point dipole.

#Arguments
- `krf`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the field is calculated.
- `krd`: 2D float array of size ``1\times 3`` containing the dimentionless positions ``k\vec{r_d}`` where source is located.
- `dip`: integer defining the dipole moment (``dip = 1`` is an electric x-dipole, ``dip = 2`` an elctric y-dipole...) or float array of size 3 with the desired dipole moment of the dipole.  
#Outputs
- `dxe_dipole`: complex array with the derivative of the electromagnetic field with respect to `k*x`.
- `dye_dipole`: complex array with the derivative of the electromagnetic field with respect to `k*y`.
- `dze_dipole`: complex array with the derivative of the electromagnetic field with respect to `k*z`.
"""
function d_point_dipole_e(krf, krd, dip; e0_const=1)
    n_r0 = length(krf[:,1])
    dxG_tensor = zeros(ComplexF64,n_r0*3,3)
    dyG_tensor = zeros(ComplexF64,n_r0*3,3)
    dzG_tensor = zeros(ComplexF64,n_r0*3,3)
    if length(dip) == 1  && dip < 4 && dip > 0
        dip_o = dip
        dip = zeros(3)
        dip[dip_o] = 1
    elseif length(dip) == 3
        dip = dip/norm(dip) # Ensure that its modulus is equal to one
    else
        dip = zeros(3)
        println("dip should be an integer (between 1 and 3) or a vector of length 3")
    end
    for i = 1:n_r0  
        dxGe = GreenTensors.dxG_e_renorm(krf[i,:],krd[1,:])   
        dxG_tensor[3 * (i-1) + 1:3 * (i-1) + 3 , :] = dxGe
        dyGe = GreenTensors.dyG_e_renorm(krf[i,:],krd[1,:])   
        dyG_tensor[3 * (i-1) + 1:3 * (i-1) + 3 , :] = dyGe
        dzGe = GreenTensors.dzG_e_renorm(krf[i,:],krd[1,:])   
        dzG_tensor[3 * (i-1) + 1:3 * (i-1) + 3 , :] = dzGe
    end
    dxe_dipole = dxG_tensor*dip*e0_const
    dxe_dipole = transpose(reshape(dxe_dipole,3,n_r0))
    dye_dipole = dyG_tensor*dip*e0_const
    dye_dipole = transpose(reshape(dye_dipole,3,n_r0))
    dze_dipole = dzG_tensor*dip*e0_const
    dze_dipole = transpose(reshape(dze_dipole,3,n_r0))

    return dxe_dipole, dye_dipole, dze_dipole      
end

@doc raw"""
    d_gauss_beam_e_m(rf,k,bw0, maxe = Int(5e3))
Adimensional derivative of the field distribution of a Gaussian beam that propagates along the z-axis and the electric field is polarized along the x-axis (polarized electric).
For another polarization just rotate the field in the xy-plane. Also, for a polarized magnetic field, exchange E -> Hz and Hz -> -E. 

#Arguments
- `rf`: 2D float array of size ``N\times 3`` containing the positions where the field is calculated (``N`` is the number of positions).
- `k`: scalar with the modulus wavevector.
- `bw0`: float with the beam waist radius.
- `kmax`: float setting the limit of the radial integration (it shoud be ``kmax < k``).
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `dxeh_gauss`: complex array with the derivative of the electromagnetic field with respect to `k*x`.
- `dyeh_gauss`: complex array with the derivative of the electromagnetic field with respect to `k*y`.
- `dzeh_gauss`: complex array with the derivative of the electromagnetic field with respect to `k*z`.
"""
function d_gauss_beam_e_m(rf,k,bw0; kmax = nothing, maxe=Int(5e3))
    if kmax===nothing
        kmax = k
    elseif kmax>k
        kmax = k
    end
    n_rf = length(rf[:,1])
    dxeh_gauss = zeros(ComplexF64,n_rf,6)
    dyeh_gauss = zeros(ComplexF64,n_rf,6)
    dzeh_gauss = zeros(ComplexF64,n_rf,6)
    for i = 1:n_rf
        r = rf[i,:]
        dxGB(QT, dxGB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(k^2 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_kp = bw0^2*exp(-bw0^2*kp^2/4)/(4*pi)*im*kx
            factor = exp(im*(kx*r[1] + ky*r[2] + Q*r[3] ))*f_kp
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor/k
            Hr_y = (kx^2 + Q^2)*factor/k
            Hr_z = - ky*Q*factor/k

            dxGB_QT[1] = real(Er_x)
            dxGB_QT[2] = imag(Er_x)
            dxGB_QT[3] = real(Er_z)
            dxGB_QT[4] = imag(Er_z)
            dxGB_QT[5] = real(Hr_x)
            dxGB_QT[6] = imag(Hr_x)
            dxGB_QT[7] = real(Hr_y)
            dxGB_QT[8] = imag(Hr_y)
            dxGB_QT[9] = real(Hr_z)
            dxGB_QT[10] = imag(Hr_z)
        end
        dyGB(QT, dyGB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(k^2 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_kp = bw0^2*exp(-bw0^2*kp^2/4)/(4*pi)*im*ky
            factor = exp(im*(kx*r[1] + ky*r[2] + Q*r[3] ))*f_kp
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor/k
            Hr_y = (kx^2 + Q^2)*factor/k
            Hr_z = - ky*Q*factor/k

            dyGB_QT[1] = real(Er_x)
            dyGB_QT[2] = imag(Er_x)
            dyGB_QT[3] = real(Er_z)
            dyGB_QT[4] = imag(Er_z)
            dyGB_QT[5] = real(Hr_x)
            dyGB_QT[6] = imag(Hr_x)
            dyGB_QT[7] = real(Hr_y)
            dyGB_QT[8] = imag(Hr_y)
            dyGB_QT[9] = real(Hr_z)
            dyGB_QT[10] = imag(Hr_z)
        end
        dzGB(QT, dzGB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(k^2 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_kp = bw0^2*exp(-bw0^2*kp^2/4)/(4*pi)*im*Q
            factor = exp(im*(kx*r[1] + ky*r[2] + Q*r[3] ))*f_kp
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor/k
            Hr_y = (kx^2 + Q^2)*factor/k
            Hr_z = - ky*Q*factor/k

            dzGB_QT[1] = real(Er_x)
            dzGB_QT[2] = imag(Er_x)
            dzGB_QT[3] = real(Er_z)
            dzGB_QT[4] = imag(Er_z)
            dzGB_QT[5] = real(Hr_x)
            dzGB_QT[6] = imag(Hr_x)
            dzGB_QT[7] = real(Hr_y)
            dzGB_QT[8] = imag(Hr_y)
            dzGB_QT[9] = real(Hr_z)
            dzGB_QT[10] = imag(Hr_z)
        end
        QT0 = [0, 0]
        QT1 = [kmax, 2*pi]
        (dxEHg, dxerEg) = hcubature(10, dxGB, QT0, QT1, maxevals = maxe)
        (dyEHg, dyerEg) = hcubature(10, dyGB, QT0, QT1, maxevals = maxe)
        (dzEHg, dzerEg) = hcubature(10, dzGB, QT0, QT1, maxevals = maxe)
        dxE = [dxEHg[1] + im*dxEHg[2], 0, dxEHg[3] + im*dxEHg[4]]
        dxZH = [dxEHg[5] + im*dxEHg[6], dxEHg[7] + im*dxEHg[8], dxEHg[9] + im*dxEHg[10]]
        dxeh_gauss[i,:] = [dxE; dxZH]
        dyE = [dyEHg[1] + im*dyEHg[2], 0, dyEHg[3] + im*dyEHg[4]]
        dyZH = [dyEHg[5] + im*dyEHg[6], dyEHg[7] + im*dyEHg[8], dyEHg[9] + im*dyEHg[10]]
        dyeh_gauss[i,:] = [dyE; dyZH]
        dzE = [dzEHg[1] + im*dzEHg[2], 0, dzEHg[3] + im*dzEHg[4]]
        dzZH = [dzEHg[5] + im*dzEHg[6], dzEHg[7] + im*dzEHg[8], dzEHg[9] + im*dzEHg[10]]
        dzeh_gauss[i,:] = [dzE; dzZH]
    end
    return 1/k*dxeh_gauss, 1/k*dyeh_gauss, 1/k*dzeh_gauss
end

@doc raw"""
    d_gauss_beam_e(rf,k,bw0, maxe = Int(5e3))
Adimensional derivative of the field distribution of a Gaussian beam that propagates along the z-axis and the electric field is polarized along the x-axis (polarized electric).
For another polarization just rotate the field in the xy-plane. Also, for a polarized magnetic field, exchange E -> Hz and Hz -> -E. 

#Arguments
- `rf`: 2D float array of size ``N\times 3`` containing the positions where the field is calculated (``N`` is the number of positions).
- `k`: scalar with the modulus wavevector.
- `bw0`: float with the beam waist radius.
- `kmax`: float setting the limit of the radial integration (it shoud be ``kmax < k``).
- `maxe`: maximum number of evaluations in the adapative integral (see Cubature for more details).
#Outputs
- `dxeh_gauss`: complex array with the derivative of the electromagnetic field with respect to `k*x`.
- `dyeh_gauss`: complex array with the derivative of the electromagnetic field with respect to `k*y`.
- `dzeh_gauss`: complex array with the derivative of the electromagnetic field with respect to `k*z`.
"""
function d_gauss_beam_e(rf,k,bw0; kmax = nothing, maxe=Int(5e3))
    if kmax===nothing
        kmax = k
    elseif kmax>k
        kmax = k
    end
    n_rf = length(rf[:,1])
    dxeh_gauss = zeros(ComplexF64,n_rf,3)
    dyeh_gauss = zeros(ComplexF64,n_rf,3)
    dzeh_gauss = zeros(ComplexF64,n_rf,3)
    for i = 1:n_rf
        r = rf[i,:]
        dxGB(QT, dxGB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(k^2 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_kp = bw0^2*exp(-bw0^2*kp^2/4)/(4*pi)*im*kx
            factor = exp(im*(kx*r[1] + ky*r[2] + Q*r[3] ))*f_kp
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor/k
            Hr_y = (kx^2 + Q^2)*factor/k
            Hr_z = - ky*Q*factor/k

            dxGB_QT[1] = real(Er_x)
            dxGB_QT[2] = imag(Er_x)
            dxGB_QT[3] = real(Er_z)
            dxGB_QT[4] = imag(Er_z)
        end
        dyGB(QT, dyGB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(k^2 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_kp = bw0^2*exp(-bw0^2*kp^2/4)/(4*pi)*im*ky
            factor = exp(im*(kx*r[1] + ky*r[2] + Q*r[3] ))*f_kp
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor/k
            Hr_y = (kx^2 + Q^2)*factor/k
            Hr_z = - ky*Q*factor/k

            dyGB_QT[1] = real(Er_x)
            dyGB_QT[2] = imag(Er_x)
            dyGB_QT[3] = real(Er_z)
            dyGB_QT[4] = imag(Er_z)
        end
        dzGB(QT, dzGB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(k^2 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_kp = bw0^2*exp(-bw0^2*kp^2/4)/(4*pi)*im*Q
            factor = exp(im*(kx*r[1] + ky*r[2] + Q*r[3] ))*f_kp
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor/k
            Hr_y = (kx^2 + Q^2)*factor/k
            Hr_z = - ky*Q*factor/k

            dzGB_QT[1] = real(Er_x)
            dzGB_QT[2] = imag(Er_x)
            dzGB_QT[3] = real(Er_z)
            dzGB_QT[4] = imag(Er_z)
        end
        QT0 = [0, 0]
        QT1 = [kmax, 2*pi]
        (dxEHg, dxerEg) = hcubature(4, dxGB, QT0, QT1, maxevals = maxe)
        (dyEHg, dyerEg) = hcubature(4, dyGB, QT0, QT1, maxevals = maxe)
        (dzEHg, dzerEg) = hcubature(4, dzGB, QT0, QT1, maxevals = maxe)
        dxE = [dxEHg[1] + im*dxEHg[2], 0, dxEHg[3] + im*dxEHg[4]]
        dxeh_gauss[i,:] = dxE
        dyE = [dyEHg[1] + im*dyEHg[2], 0, dyEHg[3] + im*dyEHg[4]]
        dyeh_gauss[i,:] = dyE
        dzE = [dzEHg[1] + im*dzEHg[2], 0, dzEHg[3] + im*dzEHg[4]]
        dzeh_gauss[i,:] = dzE
    end
    return 1/k*dxeh_gauss, 1/k*dyeh_gauss, 1/k*dzeh_gauss
end

end