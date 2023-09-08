module InputFields

export plane_wave_e, plane_wave_e_m, point_dipole_e, point_dipole_e_m, gauss_beam_e, ghermite_beam_e, glaguerre_beam_e, gauss_beam_e_m, ghermite_beam_e_m, glaguerre_beam_e_m, d_plane_wave_e, d_plane_wave_e_m, d_point_dipole_e, d_point_dipole_e_m, d_gauss_beam_e, d_gauss_beam_e_m

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
    plane_wave_e(krf;khat=[0,0,1],e0=[1,0,0])
Computes a simple plane with dimensionless input evaluated at `krf`. `khat` is the direction of propagation and `e0` is the polarization.
`krf` is a ``N\times 3`` float array. 
The output is a ``N\times 3`` complex array representing the electric field.
"""
function plane_wave_e(krf;khat=[0,0,1],e0=[1,0,0])
    n=length(krf[:,1])
    E0=zeros(ComplexF64,n,3)
    khat = khat/norm(khat)
    for i=1:n
        E0[i,:]=exp(im*dot(khat,krf[i,:]))*e0
    end
    return E0
end


@doc raw"""
    plane_wave_e_m(krf;khat=[0,0,1],e0=[1,0,0])
Computes a simple plane with dimensionless input evaluated at `krf`. `khat` is the direction of propagation and `e0` is the polarization.
`krf` is a ``N\times 3`` float array. 
The output is a ``N\times 6`` complex array representing the electric and magnetic fields.
"""
function plane_wave_e_m(krf;khat=[0,0,1],e0=[1,0,0])
    n=length(krf[:,1])
    phi=zeros(ComplexF64,n,6)
    khat = khat/norm(khat)
    for i=1:n
        phi[i,1:3]=exp(im*dot(khat,krf[i,:]))*e0
        phi[i,4:6]=cross(khat,phi[i,1:3])
    end
    return phi
end

@doc raw"""
    point_dipole_e_m(krf, krd, dip, e0=1)

Computes the electric and magnetic fields emitted by a point dipole.

# Arguments

- `krf`: 2D float array of size ``N\times 3`` containing the dimensionless positions ``k\mathbf{r}`` where the field is computed.
- `krd`: 1D float array of size 3 containing the dimensionless position ``k\mathbf{r_d}`` where source is located.
- `dip`: integer defining the dipole moment (`dip = 1` is an electric x-dipole, `dip = 2` an elctric y-dipole...) or complex array of size 6 with the desired dipole moment of the dipole.  
- `e0`: float with the modulus of the dipole moment. 

# Outputs
- `phi_dipole`: 2D ``N\times 6``complex array with the electromagnetic fields at each position.
"""
function point_dipole_e_m(krf, krd, dip; e0=1)
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
        Ge, Gm = GreenTensors.G_em_renorm(krf[i,:],krd)   
        G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [Ge im*Gm; -im*Gm Ge]
    end
    phi_dipole = G_tensor*dip*e0
    phi_dipole = transpose(reshape(phi_dipole,6,n_r0))
    return phi_dipole      
end

@doc raw"""
    point_dipole_e(krf, krd, dip, e0_const=1)

Computes the electric field emitted by a point dipole.

# Arguments
- `krf`: 2D float array of size ``N\times 3`` containing the dimensionless positions ``k\mathbf{r}`` where the field is computed.
- `krd`: 1D float array of size 3 containing the dimensionless position ``k\mathbf{r_d}`` where source is located.
- `dip`: integer defining the dipole moment (`dip = 1` is an electric x-dipole, `dip = 2` an elctric y-dipole...) or complex array of size 6 with the desired dipole moment of the dipole.  
- `e0`: scalar with the modulus of the dipole moment. 

# Outputs
- `e_dipole`:2D ``N\times 3`` complex array with the electromagnetic field.
"""
function point_dipole_e(krf, krd, dip; e0=1)
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
        Ge, Gm = GreenTensors.G_em_renorm(krf[i,:],krd)   
        G_tensor[3 * (i-1) + 1:3 * (i-1) + 3, :] = Ge
    end
    e_dipole = G_tensor*dip*e0
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
    hermite_e_pol(x,n)

Computes probabilist's Hermite polynomials.

# Arguments
- `x`: float with the argument.
- `n`: int with the order.

# Outputs
- `hp`: probabilist's Hermite polynomials of argument ``x`` and order ``n``.
"""
function hermite_e_pol(x,n)
    if n == 0
        hp = 1
    elseif n ==1
        hp = x
    else
        hp = x*hermite_e_pol(x,n-1) - (n-1)*hermite_e_pol(x,n-2)
    end
    return hp
end


@doc raw"""
    laguerre_pol(x,n,a)

Computes generalized Laguerre polynomials.

# Arguments
- `x`: float with the argument.
- `n`: int with the order.
- `a`: int with the second order.

# Outputs
- `lp`: generalized Laguerre polynomials of argument ``x`` and order ``n`` and ``a``.
"""
function laguerre_pol(x,n,a)
    if n == 0
        lp = 1
    elseif n ==1
        lp = 1 + a - x
    else
        lp = ((2*n - 1 + a - x)*laguerre_pol(x,n-1,a) - (n - 1 + a)*laguerre_pol(x,n-2,a))/n
    end
    return lp
end

@doc raw"""
    create_f_gauss_kind(kbw0,n,m,kind)

Create the function to calculate the fourier transform components for the spefic beam (gaussian, hermite-gaussian or legendre-gaussian) used in the functions that calculates the field and the derivaties.

# Arguments
- `kbw0`: float with the dimensionless beam waist radius (``k\omega_0``, where ``\omega_0`` is the beam waist radius).
- `n`: int with the order of the beam.
- `m`: int with the degree of the beam.
- `kind`: string with the kind of beam ("hermite" or "laguerre").

# Outputs
- `lp`: generalized Laguerre polynomials of argument ``x`` and order ``n`` and ``a``.
"""
function create_f_gauss_kind(kbw0,n,m,kind)
    return function(kx,ky,kp,theta)
        if n == 0 && m == 0
            return 1
        elseif kind == "hermite"
            return (-sqrt(2)*im)^(n+m)*hermite_e_pol(kx*kbw0,n)*hermite_e_pol(ky*kbw0,m)
        elseif kind == "laguerre"
            return laguerre_pol(kp^2*kbw0^2/2,n,m)*exp(im*m*theta)*(-1)^(m+n)*sqrt(2)^(-m)*kbw0^m*kp^m*(im)^(m)
        end
    end
end

@doc raw"""
    gaussian_beam_e_m(krf, kbw0,n,m; n = 0, m = 0, kind = "hermite", e0 = 1, kmax = nothing, maxe=Int(5e3))

Computes the electromagnetic field distribution of a Gaussian, Hermite-Gaussian and Laguerre-Gaussian beam that propagates along the z-axis and where the electric field is polarized along the x-axis (polarized electric).
By default, the Gaussian beam profile is calculated.
For another polarization just rotate the field in the xy-plane. Also, for a polarized magnetic field, exchange E with ZH and H with -E. 

# Arguments
- `krf`: 2D float array of size ``N\times 3`` containing the dimensionless positions ``k\mathbf{r}`` where the field is computed.
- `kbw0`: float with the dimensionless beam waist radius (``k\omega_0``, where ``\omega_0`` is the beam waist radius).
- `n`: int with the order of the beam.
- `m`: int with the degree of the beam.
- `kind`: string with the kind of beam ("hermite" or "laguerre"). 
- `e0`: float with the modulus of the electric field at the origin of coordinates of the theoretical field (including evanescent waves). 
- `kmax`: float setting the limit of the radial integration (it shoud be `kmax < 1`).
- `maxe`: maximum number of evaluations in the adapative integral (see [Cubature.jl](https://github.com/JuliaMath/Cubature.jl) for more details).

# Outputs
- `phi_gauss`: 2D complex array of size ``N\times 6`` with the value of the field at each position.
"""
function gaussian_beam_e_m(krf, kbw0; n = 0, m = 0, kind = "hermite", e0 = 1, kmax = nothing, maxe=Int(5e3))
    if kmax===nothing
        kmax = 1
    elseif kmax>1
        kmax = 1
    end
    n_rf = length(krf[:,1])
    phi_gauss = zeros(ComplexF64,n_rf,6)
    if n == 0 && m == 0
        for i = 1:n_rf
            kr = krf[i,:]
            GB(k_i, GB_Q) = begin
                Q = k_i[1]
                kp = sqrt(1 - Q^2)
                krp = sqrt(kr[1]^2 + kr[2]^2)
                phi = atan(kr[2],kr[1])
    
                bj0 = besselj0(kp*krp)
                bj1 = besselj1(kp*krp)
                bj2 = besselj2(kp*krp)
    
                factor = exp(im*Q*kr[3])*kbw0^2*exp(-kbw0^2*kp^2/4)/(2)
                Er_x = Q*factor*bj0
                Er_z = - im*kp*factor*bj1*cos(phi)
                Hr_x = factor*1/2*kp^2*bj2*sin(2*phi)
                Hr_y = factor*(-1/2*kp^2*bj2*cos(2*phi) + (1/2*kp^2 + Q^2)*bj0  )
                Hr_z = - kp*Q*factor*im*bj1*sin(phi)
    
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
            phi_gauss[i,:] = [E; ZH]
        end
    else
        f_gauss_kind = create_f_gauss_kind(kbw0,n,m,kind)
        for i = 1:n_rf
            kr = krf[i,:]
            GB(QT, GB_QT) = begin
                Q = QT[1]
                theta = QT[2]
                kp = sqrt(1 - Q^2)
                kx = kp*cos(theta)
                ky = kp*sin(theta)

                f_beam = kbw0^2*exp(-kbw0^2*kp^2/4)/(4*pi)*f_gauss_kind(kx,ky,kp,theta)
                factor = exp(im*(kx*kr[1] + ky*kr[2] + Q*kr[3]))*f_beam
                Er_x = Q*factor
                Er_z = - kx*factor
                Hr_x = -kx*ky*factor
                Hr_y = (kx^2 + Q^2)*factor
                Hr_z = - ky*Q*factor

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
            phi_gauss[i,:] = [E; ZH]
        end
    end
    return phi_gauss*e0
end

@doc raw"""
    gaussian_beam_e(krf, kbw0,n,m; n = 0, m = 0, kind = "hermite", e0 = 1, kmax = nothing, maxe=Int(5e3))

Computes the electric field distribution of a Gaussian, Hermite-Gaussian and Laguerre-Gaussian beam that propagates along the z-axis and where the electric field is polarized along the x-axis (polarized electric).
By default, the Gaussian beam profile is calculated.
For another polarization just rotate the field in the xy-plane. Also, for a polarized magnetic field, exchange E with ZH and H with -E. 

# Arguments
- `krf`: 2D float array of size ``N\times 3`` containing the dimensionless positions ``k\mathbf{r}`` where the field is computed.
- `kbw0`: float with the dimensionless beam waist radius (``k\omega_0``, where ``\omega_0`` is the beam waist radius).
- `n`: int with the order of the beam.
- `m`: int with the degree of the beam.
- `kind`: string with the kind of beam ("hermite" or "laguerre"). 
- `e0`: float with the modulus of the electric field at the origin of coordinates of the theoretical field (including evanescent waves). 
- `kmax`: float setting the limit of the radial integration (it shoud be `kmax < 1`).
- `maxe`: maximum number of evaluations in the adapative integral (see [Cubature.jl](https://github.com/JuliaMath/Cubature.jl) for more details).

# Outputs
- `eh_gauss`: 2D complex array of size ``N\times 6`` with the value of the field at each position.
"""
function gaussian_beam_e(krf, kbw0; n = 0, m = 0, kind = "hermite", e0 = 1, kmax = nothing, maxe=Int(5e3))
    if kmax===nothing
        kmax = 1
    elseif kmax>1
        kmax = 1
    end
    n_rf = length(krf[:,1])
    e_gauss = zeros(ComplexF64,n_rf,3)
    if n == 0 && m == 0
        for i = 1:n_rf
            kr = krf[i,:]
            GB(k_i, GB_Q) = begin
                Q = k_i[1]
                kp = sqrt(1 - Q^2)
                krp = sqrt(kr[1]^2 + kr[2]^2)
                phi = atan(kr[2],kr[1])
    
                bj0 = besselj0(kp*krp)
                bj1 = besselj1(kp*krp)
                bj2 = besselj2(kp*krp)
    
                factor = exp(im*Q*kr[3])*kbw0^2*exp(-kbw0^2*kp^2/4)/(2)
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
    else
        f_gauss_kind = create_f_gauss_kind(kbw0,n,m,kind)
        for i = 1:n_rf
            kr = krf[i,:]
            GB(QT, GB_QT) = begin
                Q = QT[1]
                theta = QT[2]
                kp = sqrt(1 - Q^2)
                kx = kp*cos(theta)
                ky = kp*sin(theta)

                f_beam = kbw0^2*exp(-kbw0^2*kp^2/4)/(4*pi)*f_gauss_kind(kx,ky,kp,theta)
                factor = exp(im*(kx*kr[1] + ky*kr[2] + Q*kr[3]))*f_beam
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
            e_gauss[i,:] = E
        end
    end
    return e_gauss*e0
end

# Derivative of the fields

@doc raw"""
    d_plane_wave_e(kr;khat=[0,0,1],e0=[1,0,0])
Computes the derivatives of an electric field generated with `plane_wave_e` (the arguments are the same).
Outputs three 2D arrays of size ``N\times 3`` containing the field derivatives with respect `k*x`, `k*y` and `k*z`.
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
Computes the derivatives of an electromagnetic field generated with `plane_wave_e_m` (the arguments are the same).
Outputs three 2D arrays of size ``N\times 6` containing the field derivatives with respect of `k*x`, `k*y` and `k*z`.
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
    d_point_dipole_e_m(krf, krd, dip, e0=1)
Computes the derivatives of an electromagnetic field generated with `point_dipole_e_m` (the arguments are the same).
Outputs three 2D arrays of size ``N\times 6` containing the field derivatives with respect of `k*x`, `k*y` and `k*z`.
"""
function d_point_dipole_e_m(krf, krd, dip; e0=1)
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
        dxGe, dxGm = GreenTensors.dxG_em_renorm(krf[i,:],krd)   
        dxG_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [dxGe im*dxGm; -im*dxGm dxGe]
        dyGe, dyGm = GreenTensors.dyG_em_renorm(krf[i,:],krd)   
        dyG_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [dyGe im*dyGm; -im*dyGm dyGe]
        dzGe, dzGm = GreenTensors.dzG_em_renorm(krf[i,:],krd)   
        dzG_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [dzGe im*dzGm; -im*dzGm dzGe]
    end
    dxe_dipole = dxG_tensor*dip*e0
    dxe_dipole = transpose(reshape(dxe_dipole,6,n_r0))
    dye_dipole = dyG_tensor*dip*e0
    dye_dipole = transpose(reshape(dye_dipole,6,n_r0))
    dze_dipole = dzG_tensor*dip*e0
    dze_dipole = transpose(reshape(dze_dipole,6,n_r0))

    return dxe_dipole, dye_dipole, dze_dipole        
end

@doc raw"""
    d_point_dipole_e(krf, krd, dip, e0=1)
Computes the derivatives of an electric field generated with `point_dipole_e` (the arguments are the same).
Outputs three 2D arrays of size ``N\times 3`` containing the field derivatives with respect `k*x`, `k*y` and `k*z`.
"""
function d_point_dipole_e(krf, krd, dip; e0=1)
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
        dxGe = GreenTensors.dxG_e_renorm(krf[i,:],krd)   
        dxG_tensor[3 * (i-1) + 1:3 * (i-1) + 3 , :] = dxGe
        dyGe = GreenTensors.dyG_e_renorm(krf[i,:],krd)   
        dyG_tensor[3 * (i-1) + 1:3 * (i-1) + 3 , :] = dyGe
        dzGe = GreenTensors.dzG_e_renorm(krf[i,:],krd)   
        dzG_tensor[3 * (i-1) + 1:3 * (i-1) + 3 , :] = dzGe
    end
    dxe_dipole = dxG_tensor*dip*e0
    dxe_dipole = transpose(reshape(dxe_dipole,3,n_r0))
    dye_dipole = dyG_tensor*dip*e0
    dye_dipole = transpose(reshape(dye_dipole,3,n_r0))
    dze_dipole = dzG_tensor*dip*e0
    dze_dipole = transpose(reshape(dze_dipole,3,n_r0))

    return dxe_dipole, dye_dipole, dze_dipole      
end

@doc raw"""
    d_gaussian_beam_e_m(krf, kbw0; n = 0, m = 0, kind = "hermite", e0 = 1, kmax = nothing, maxe=Int(5e3))
Computes the derivatives of an electromagnetic field generated with `gaussian_beam_e_m` (the arguments are the same).
Outputs three 2D arrays of size ``N\times 6` containing the field derivatives with respect of `k*x`, `k*y` and `k*z`.
"""
function d_gaussian_beam_e_m(krf, kbw0; n = 0, m = 0, kind = "hermite", e0 = 1, kmax = nothing, maxe=Int(5e3))
    if kmax===nothing
        kmax = 1
    elseif kmax>1
        kmax = 1
    end
    n_rf = length(krf[:,1])
    dxeh_gauss = zeros(ComplexF64,n_rf,6)
    dyeh_gauss = zeros(ComplexF64,n_rf,6)
    dzeh_gauss = zeros(ComplexF64,n_rf,6)
    f_gauss_kind = create_f_gauss_kind(kbw0,n,m,kind)
    for i = 1:n_rf
        kr = krf[i,:]
        dxGB(QT, dxGB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(1 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_beam = kbw0^2*exp(-kbw0^2*kp^2/4)/(4*pi)*f_gauss_kind(kx,ky,kp,theta)
            factor = exp(im*(kx*kr[1] + ky*kr[2] + Q*kr[3]))*f_beam*im*kx
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor
            Hr_y = (kx^2 + Q^2)*factor
            Hr_z = - ky*Q*factor

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
            kp = sqrt(1 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_beam = kbw0^2*exp(-kbw0^2*kp^2/4)/(4*pi)*f_gauss_kind(kx,ky,kp,theta)
            factor = exp(im*(kx*kr[1] + ky*kr[2] + Q*kr[3]))*f_beam*im*ky
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor
            Hr_y = (kx^2 + Q^2)*factor
            Hr_z = - ky*Q*factor

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
            kp = sqrt(1 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_beam = kbw0^2*exp(-kbw0^2*kp^2/4)/(4*pi)*f_gauss_kind(kx,ky,kp,theta)
            factor = exp(im*(kx*kr[1] + ky*kr[2] + Q*kr[3]))*f_beam*im*Q
            Er_x = Q*factor
            Er_z = - kx*factor
            Hr_x = -kx*ky*factor
            Hr_y = (kx^2 + Q^2)*factor
            Hr_z = - ky*Q*factor

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
    return dxeh_gauss*e0, dyeh_gauss*e0, dzeh_gauss*e0
end

@doc raw"""
    d_gaussian_beam_e(krf, kbw0; n = 0, m = 0, kind = "hermite", e0 = 1, kmax = nothing, maxe=Int(5e3))
Computes the derivatives of an electric field generated with `gaussian_beam_e` (the arguments are the same).
Outputs three 2D arrays of size ``N\times 3`` containing the field derivatives with respect `k*x`, `k*y` and `k*z`.
"""
function d_gaussian_beam_e(krf, kbw0; n = 0, m = 0, kind = "hermite", e0 = 1, kmax = nothing, maxe=Int(5e3))
    if kmax===nothing
        kmax = 1
    elseif kmax>1
        kmax = 1
    end
    n_rf = length(krf[:,1])
    dxe_gauss = zeros(ComplexF64,n_rf,3)
    dye_gauss = zeros(ComplexF64,n_rf,3)
    dze_gauss = zeros(ComplexF64,n_rf,3)
    f_gauss_kind = create_f_gauss_kind(kbw0,n,m,kind)
    for i = 1:n_rf
        kr = krf[i,:]
        dxGB(QT, dxGB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(1 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_beam = kbw0^2*exp(-kbw0^2*kp^2/4)/(4*pi)*f_gauss_kind(kx,ky,kp,theta)
            factor = exp(im*(kx*kr[1] + ky*kr[2] + Q*kr[3]))*f_beam*im*kx
            Er_x = Q*factor
            Er_z = - kx*factor

            dxGB_QT[1] = real(Er_x)
            dxGB_QT[2] = imag(Er_x)
            dxGB_QT[3] = real(Er_z)
            dxGB_QT[4] = imag(Er_z)
        end
        dyGB(QT, dyGB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(1 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_beam = kbw0^2*exp(-kbw0^2*kp^2/4)/(4*pi)*f_gauss_kind(kx,ky,kp,theta)
            factor = exp(im*(kx*kr[1] + ky*kr[2] + Q*kr[3]))*f_beam*im*ky
            Er_x = Q*factor
            Er_z = - kx*factor

            dyGB_QT[1] = real(Er_x)
            dyGB_QT[2] = imag(Er_x)
            dyGB_QT[3] = real(Er_z)
            dyGB_QT[4] = imag(Er_z)
        end
        dzGB(QT, dzGB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(1 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_beam = kbw0^2*exp(-kbw0^2*kp^2/4)/(4*pi)*f_gauss_kind(kx,ky,kp,theta)
            factor = exp(im*(kx*kr[1] + ky*kr[2] + Q*kr[3]))*f_beam*im*Q
            Er_x = Q*factor
            Er_z = - kx*factor

            dzGB_QT[1] = real(Er_x)
            dzGB_QT[2] = imag(Er_x)
            dzGB_QT[3] = real(Er_z)
            dzGB_QT[4] = imag(Er_z)
        end
        QT0 = [0, 0]
        QT1 = [kmax, 2*pi]
        (dxEg, dxerEg) = hcubature(4, dxGB, QT0, QT1, maxevals = maxe)
        (dyEg, dyerEg) = hcubature(4, dyGB, QT0, QT1, maxevals = maxe)
        (dzEg, dzerEg) = hcubature(4, dzGB, QT0, QT1, maxevals = maxe)
        dxE = [dxEg[1] + im*dxEg[2], 0, dxEg[3] + im*dxEg[4]]
        dxe_gauss[i,:] = dxE
        dyE = [dyEg[1] + im*dyEg[2], 0, dyEg[3] + im*dyEg[4]]
        dye_gauss[i,:] = dyE
        dzE = [dzEg[1] + im*dzEg[2], 0, dzEg[3] + im*dzEg[4]]
        dze_gauss[i,:] = dzE
    end
    return dxe_gauss*e0, dye_gauss*e0, dze_gauss*e0
end

@doc raw"""
    ghermite_amp(kbw0,n,m; kmax = nothing, maxe=Int(1e4), int_size = 5)

Computes the integral of the field amplitude (|E|^2) of the Hermite-Gaussian beam at the focal plane. 

# Arguments
- `kbw0`: float with the dimensionless beam waist radius (``k\omega_0``, where ``\omega_0`` is the beam waist radius).
- `n`: non-negative int with the radial order of the beam.
- `m`: int with the azimuthal order of the beam.
- `kmax`: float setting the limit of the radial integration (it shoud be `kmax < 1`).
- `maxe`: maximum number of evaluations in the adapative integral (see [Cubature.jl](https://github.com/JuliaMath/Cubature.jl) for more details).
- `int_size`: size of the integration area in units of ``kbw0``. For high-order beams this parameters should be ajusted.

# Outputs
- `int_amplitude`: integral of the field amplitude (|E|^2) in the area defined by int_size (x = [-kbw0*int_size, kbw0*int_size], y = [-kbw0*int_size, kbw0*int_size]).
"""
function ghermite_amp(kbw0,n,m; kmax = nothing, maxe=Int(1e4), int_size = 5)
    if kmax===nothing
        kmax = 1
    elseif kmax>1
        kmax = 1
    end
    function field_amp(kxy)
        krx = kxy[1]
        kry = kxy[2]
        GB(QT, GB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(1 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)
    
            f_gauss = kbw0^2*exp(-kbw0^2*kp^2/4)/(4*pi)
            f_hermite = (-sqrt(2)*im)^(n+m)*hermite_e_pol(kx*kbw0,n)*hermite_e_pol(ky*kbw0,m)*f_gauss
            factor = exp(im*(kx*krx + ky*kry))*f_hermite
            Er_x = Q*factor
            Er_z = - kx*factor
    
            GB_QT[1] = real(Er_x)
            GB_QT[2] = real(Er_z)
        end
        QT0 = [0, 0]
        QT1 = [kmax, 2*pi]
        (Eg, erEg) = hcubature(2, GB, QT0, QT1, maxevals = maxe)
        ampliude = abs2(Eg[1]) + abs2(Eg[2])
        return ampliude
    end
    kxy_inf = [-kbw0*int_size, -kbw0*int_size]
    kxy_sup = [kbw0*int_size, kbw0*int_size]
    (int_amplitude, erEg) = hcubature(field_amp, kxy_inf, kxy_sup, maxevals = maxe)
    return int_amplitude
end

@doc raw"""
    glaguerre_amp(kbw0,n,m; kmax = nothing, maxe=Int(1e4), int_size = 5)

Computes the integral of the field amplitude (|E|^2) of the Laguerre-Gaussian beam at the focal plane. 

# Arguments
- `kbw0`: float with the dimensionless beam waist radius (``k\omega_0``, where ``\omega_0`` is the beam waist radius).
- `n`: non-negative int with the radial order of the beam.
- `m`: int with the azimuthal order of the beam.
- `kmax`: float setting the limit of the radial integration (it shoud be `kmax < 1`).
- `maxe`: maximum number of evaluations in the adapative integral (see [Cubature.jl](https://github.com/JuliaMath/Cubature.jl) for more details).
- `int_size`: size of the integration area in units of ``kbw0``. For high-order beams this parameters should be ajusted.

# Outputs
- `int_amplitude`: integral of the field amplitude (|E|^2) in the area defined by int_size (x = [-kbw0*int_size, kbw0*int_size], y = [-kbw0*int_size, kbw0*int_size]).
"""
function glaguerre_amp(kbw0,n,m; kmax = nothing, maxe=Int(1e4), int_size = 5)
    if kmax===nothing
        kmax = 1
    elseif kmax>1
        kmax = 1
    end
    function field_amp(kxy)
        krx = kxy[1]
        kry = kxy[2]
        phi = atan(kry,krx)
        GB(QT, GB_QT) = begin
            Q = QT[1]
            theta = QT[2]
            kp = sqrt(1 - Q^2)
            kx = kp*cos(theta)
            ky = kp*sin(theta)

            f_gauss = kbw0^2*exp(-kbw0^2*kp^2/4)/(4*pi)
            f_laguerre = laguerre_pol(kp^2*kbw0^2/2,n,m)*exp(im*m*theta)*(-1)^(m+n)*sqrt(2)^(-m)*kbw0^m*kp^m*(im)^(m)*f_gauss
            factor = exp(im*(kx*krx + ky*kry))*f_laguerre
            Er_x = Q*factor
            Er_z = - kx*factor
    
            GB_QT[1] = real(Er_x*exp(-im*m*phi))
            GB_QT[2] = real(Er_z*exp(-im*m*phi))
        end
        QT0 = [0, 0]
        QT1 = [kmax, 2*pi]
        (Eg, erEg) = hcubature(2, GB, QT0, QT1, maxevals = maxe)
        ampliude = abs2(Eg[1]) + abs2(Eg[2])
        return ampliude
    end
    kxy_inf = [-kbw0*int_size, -kbw0*int_size]
    kxy_sup = [kbw0*int_size, kbw0*int_size]
    (int_amplitude, erEg) = hcubature(field_amp, kxy_inf, kxy_sup, maxevals = maxe)
    return int_amplitude
end


end