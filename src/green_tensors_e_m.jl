module GreenTensors

export G_em_renorm, G_em_far_field_renorm, G_em_renorm, G_m_renorm, G_e_renorm, dxG_e_renorm, dy_G_e_renorm, dz_G_e_renorm, dxG_m_renorm, dy_G_m_renorm, dz_G_m_renorm, dxG_em_renorm, dy_G_em_renorm, dz_G_em_renorm,denormalize_G_e,denormalize_G_m


#***********************
#imports
#***********************
using Base
using LinearAlgebra

@doc raw"""
    G_em_renorm(kr1,kr2)
Computes the renormalized electric and magnetic green tensors between two position `r1` and `r2`, where the imputs are the positions multiplied by the wave number `kr1` and `kr2`.
The outputs are two dimensionless 3x3 complex matrix.
"""
function G_em_renorm(kr1,kr2)

    kR_vec = kr1-kr2
    kR = norm(kR_vec)
    kR2 = kR^2
    Ur=kR_vec/kR

    term1=exp(im*kR)/(kR)
    term2=1+(im/(kR))-(1/(kR2))
    term3=1+(3*im/(kR))-(3/(kR2))
    term4=(im*kR-1)/kR
    matrix=kR_vec*transpose(kR_vec)/kR2
    id3=[1 0 0;0 1 0;0 0 1]
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]

    Ge = term1*(term2*id3-term3*matrix) 	#same as G_e_renorm(kr1,kr2)
    Gm = term1*term4*mat			#same as G_m_renorm(kr1,kr2)

    return Ge, Gm
end

@doc raw"""
    G_em_far_field_renorm(kr1,kr2)
Computes the renormalized electric and magnetic green tensors in the far field approximation between two dimensionless position `kr1` and `kr2`. Note that is is only valid for ``kr_1>>kr_2`` and ``kr_1>>1``
The outputs are two dimensionless 3x3 complex matrix.
"""
function G_em_far_field_renorm(kr1,kr2)
    kr1_norm=norm(kr1)
    kr1_vec=kr1/norm(kr1)
    common_factor=exp(im*kr1_norm)/kr1_norm*exp(-im*dot(kr2,kr1_vec))
    Ge=common_factor*(Matrix{ComplexF64}(I,3,3)-kr1_vec*transpose(kr1_vec))
    Gm=common_factor*im*[0 -kr1_vec[3] kr1_vec[2];kr1_vec[3] 0 -kr1_vec[1];-kr1_vec[2] kr1_vec[1] 0]
    return Ge,Gm
end

@doc raw"""
    G_m_renorm(kr1,kr2)
Computes the renormalized magnetic green tensor between two position `r1` and `r2`, where the imputs are the positions multiplied by the wave number `kr1` and `kr2`.
The output is a dimensionless 3x3 complex matrix.
"""
function G_m_renorm(kr1,kr2)
    #difference vector
    R_vec=kr1-kr2
    R=norm(R_vec)
    Ur=R_vec/R
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    #terms
    term1=exp(im*R)/R
    term2=(im*R-1)/R
    #return green tensor
    return term1*term2*mat
end

@doc raw"""
    G_e_renorm(kr1,kr2)
Computes the renormalized electric green tensor between two position `r1` and `r2`, where the imputs are the positions multiplied by the wave number `kr1` and `kr2`.
The output is a dimensionless 3x3 complex matrix.
"""
function G_e_renorm(kr1,kr2)
    #difference vector
    R_vec=kr1-kr2
    R=norm(R_vec)
    Ur=R_vec/R

    term1=exp(im*R)/R
    term2=(R^2+im*R-1)/R^2
    term3=(-R^2-3*im*R+3)/R^2

    matrix=Ur*transpose(Ur)
    id3=[1 0 0;0 1 0;0 0 1]
    return term1*(term2*id3+term3*matrix)
end

@doc raw"""
    dxG_e_renorm(kr1,kr2)
Computes the derivative of the renormalized electric green tensor (defined in `G_e_renorm(kr1,kr2)`) regarding the `k*x` component of `kr1` between two position  `kr1` and `kr2`.
The output is a dimensionless 3x3 complex matrix.
"""
function dxG_e_renorm(kr1,kr2)

    kR_vec = kr1-kr2
    kR = norm(kR_vec)
    kR2 = kR^2
    kR3 = kR^3
    Ur=kR_vec/kR

    x = kR_vec[1]
    y = kR_vec[2]
    z = kR_vec[3]

    term1 = exp(im*kR)/kR
    term2 = (im - 2/kR - 3*im/kR2 + 3/kR3)*x/kR
    term3 = - (im - 6/kR - 15*im/kR2 + 15/kR3)*x/kR
    term4 = - (1 + 3*im/kR - 3/kR2)/kR2
    Rmx = [2*x y z;y 0 0;z 0 0]
    matrix=Ur*transpose(Ur)
    id3=[1 0 0;0 1 0;0 0 1]

    return term1*(term2*id3 + term3*matrix + term4*Rmx)

end

@doc raw"""
    dyG_e_renorm(kr1,kr2)
Computes the derivative of the renormalized electric green tensor (defined in `G_e_renorm(kr1,kr2)`) regarding the `k*y` component of `kr1` between two position  `kr1` and `kr2`.
The output is a dimensionless 3x3 complex matrix.
"""
function dyG_e_renorm(kr1,kr2)

    kR_vec = kr1-kr2
    kR = norm(kR_vec)
    kR2 = kR^2
    kR3 = kR^3
    Ur=kR_vec/kR

    x = kR_vec[1]
    y = kR_vec[2]
    z = kR_vec[3]

    term1 = exp(im*kR)/kR
    term2 = (im - 2/kR - 3*im/kR2 + 3/kR3)*y/kR
    term3 = - (im - 6/kR - 15*im/kR2 + 15/kR3)*y/kR
    term4 = - (1 + 3*im/kR - 3/kR2)/kR2
    Rmy = [0 x 0;x 2*y z;0 z 0]
    matrix=Ur*transpose(Ur)
    id3=[1 0 0;0 1 0;0 0 1]

    return term1*(term2*id3 + term3*matrix + term4*Rmy)

end

@doc raw"""
    dzG_e_renorm(kr1,kr2)
Computes the derivative of the renormalized electric green tensor (defined in `G_e_renorm(kr1,kr2)`) regarding the `k*z` component of `kr1` between two position  `kr1` and `kr2`.
The output is a dimensionless 3x3 complex matrix.
"""
function dzG_e_renorm(kr1,kr2)

    kR_vec = kr1-kr2
    kR = norm(kR_vec)
    kR2 = kR^2
    kR3 = kR^3
    Ur=kR_vec/kR

    x = kR_vec[1]
    y = kR_vec[2]
    z = kR_vec[3]

    term1 = exp(im*kR)/kR
    term2 = (im - 2/kR - 3*im/kR2 + 3/kR3)*z/kR
    term3 = - (im - 6/kR - 15*im/kR2 + 15/kR3)*z/kR
    term4 = - (1 + 3*im/kR - 3/kR2)/kR2
    Rmz = [0 0 x;0 0 y;x y 2*z]
    matrix=Ur*transpose(Ur)
    id3=[1 0 0;0 1 0;0 0 1]

    return term1*(term2*id3 + term3*matrix + term4*Rmz)

end


@doc raw"""
    dxG_m_renorm(kr1,kr2)
Computes the derivative of the renormalized magnetic green tensor (defined in `G_m_renorm(kr1,kr2)`) regarding the `k*x` component of `kr1` between two position  `kr1` and `kr2`.
The output is a dimensionless 3x3 complex matrix.
"""
function dxG_m_renorm(kr1,kr2)
    #difference vector
    kR_vec=kr1-kr2
    kR=norm(kR_vec)
    Ur=kR_vec/kR
    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 0 0;0 0 -1;0 1 0]
    #terms
    term1=exp(im*kR)/kR^3
    term2=(im*kR-1)
    term3=(im*kR)^2 - 3*im*kR + 3
    #return green tensor
    return term1*(term3*mat*Ur[1] + term2*mat3)
end


@doc raw"""
    dyG_m_renorm(kr1,kr2)
Computes the derivative of the renormalized magnetic green tensor (defined in `G_m_renorm(kr1,kr2)`) regarding the `k*y` component of `kr1` between two position  `kr1` and `kr2`.
The output is a dimensionless 3x3 complex matrix.
"""
function dyG_m_renorm(kr1,kr2)
    #difference vector
    kR_vec=kr1-kr2
    kR=norm(kR_vec)
    Ur=kR_vec/kR
    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 0 1;0 0 0;-1 0 0]
    #terms
    term1=exp(im*kR)/kR^3
    term2=(im*kR-1)
    term3=(im*kR)^2 - 3*im*kR + 3
    #return green tensor
    return term1*(term3*mat*Ur[2] + term2*mat3)
end



@doc raw"""
    dzG_m_renorm(kr1,kr2)
Computes the derivative of the renormalized magnetic green tensor (defined in `G_m_renorm(kr1,kr2)`) regarding the `k*z` component of `kr1` between two position  `kr1` and `kr2`.
The output is a dimensionless 3x3 complex matrix.
"""
function dzG_m_renorm(kr1,kr2)
    #difference vector
    kR_vec=kr1-kr2
    kR=norm(kR_vec)
    Ur=kR_vec/kR
    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 -1 0;1 0 0;0 0 0]
    #terms
    term1=exp(im*kR)/kR^3
    term2=(im*kR-1)
    term3=(im*kR)^2 - 3*im*kR + 3
    #return green tensor
    return term1*(term3*mat*Ur[3] + term2*mat3)
end

@doc raw"""
    dxG_em_renorm(kr1,kr2)
Computes the derivative of the renormalized electric and magnetic green tensors (defined in `G_e_renorm(kr1,kr2)` and `G_m_renorm(kr1,kr2)`) regarding the `k*x` component of `kr1` between two position  `kr1` and `kr2`.
The outputs are two dimensionless 3x3 complex matrix.
"""
function dxG_em_renorm(kr1,kr2)

    kR_vec = kr1-kr2
    kR = norm(kR_vec)
    kR2 = kR^2
    kR3 = kR^3
    Ur = kR_vec/kR

    term1 = exp(im*kR)/kR
    term2 = (im - 2/kR - 3*im/kR2 + 3/kR3)*Ur[1]
    term3 = - (im - 6/kR - 15*im/kR2 + 15/kR3)*Ur[1]
    term4 = - (1 + 3*im/kR - 3/kR2)/kR
    Rmx = [2*Ur[1] Ur[2] Ur[3];Ur[2] 0 0;Ur[3] 0 0]
    matrix=Ur*transpose(Ur)
    id3=[1 0 0;0 1 0;0 0 1]
    dxGe =  term1*(term2*id3 + term3*matrix + term4*Rmx)

    mat=zeros(ComplexF64,3,3)
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 0 0;0 0 -1;0 1 0]
    term5=(im*kR-1)
    term6=(im*kR)^2 - 3*im*kR + 3
    dxGm = term1/kR2*(term6*mat*Ur[1] + term5*mat3)

    return dxGe, dxGm
end

@doc raw"""
    dyG_em_renorm(kr1,kr2)
Computes the derivative of the renormalized electric and magnetic green tensors (defined in `G_e_renorm(kr1,kr2)` and `G_m_renorm(kr1,kr2)`) regarding the `k*y` component of `kr1` between two position  `kr1` and `kr2`.
The outputs are two dimensionless 3x3 complex matrix.
"""
function dyG_em_renorm(kr1,kr2)

    kR_vec = kr1-kr2
    kR = norm(kR_vec)
    kR2 = kR^2
    kR3 = kR^3
    Ur = kR_vec/kR

    term1 = exp(im*kR)/kR
    term2 = (im - 2/kR - 3*im/kR2 + 3/kR3)*Ur[2]
    term3 = - (im - 6/kR - 15*im/kR2 + 15/kR3)*Ur[2]
    term4 = - (1 + 3*im/kR - 3/kR2)/kR
    Rmy = [0 Ur[1] 0;Ur[1] 2*Ur[2] Ur[3];0 Ur[3] 0]
    matrix=Ur*transpose(Ur)
    id3=[1 0 0;0 1 0;0 0 1]
    dyGe =  term1*(term2*id3 + term3*matrix + term4*Rmy)

    mat=zeros(ComplexF64,3,3)
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 0 1;0 0 0;-1 0 0]
    term5=(im*kR-1)
    term6=(im*kR)^2 - 3*im*kR + 3
    dyGm = term1/kR2*(term6*mat*Ur[2] + term5*mat3)

    return dyGe, dyGm
end

@doc raw"""
    dzG_em_renorm(kr1,kr2)
Computes the derivative of the renormalized electric and magnetic green tensors (defined in `G_e_renorm(kr1,kr2)` and `G_m_renorm(kr1,kr2)`) regarding the `k*z` component of `kr1` between two position  `kr1` and `kr2`.
The outputs are two dimensionless 3x3 complex matrix.
"""
function dzG_em_renorm(kr1,kr2)

    kR_vec = kr1-kr2
    kR = norm(kR_vec)
    kR2 = kR^2
    kR3 = kR^3
    Ur = kR_vec/kR

    term1 = exp(im*kR)/kR
    term2 = (im - 2/kR - 3*im/kR2 + 3/kR3)*Ur[3]
    term3 = - (im - 6/kR - 15*im/kR2 + 15/kR3)*Ur[3]
    term4 = - (1 + 3*im/kR - 3/kR2)/kR
    Rmz = [0 0 Ur[1];0 0 Ur[2];Ur[1] Ur[2] 2*Ur[3]]
    matrix=Ur*transpose(Ur)
    id3=[1 0 0;0 1 0;0 0 1]
    dzGe =  term1*(term2*id3 + term3*matrix + term4*Rmz)

    mat=zeros(ComplexF64,3,3)
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 -1 0;1 0 0;0 0 0]
    term5=(im*kR-1)
    term6=(im*kR)^2 - 3*im*kR + 3
    dzGm = term1/kR2*(term6*mat*Ur[3] + term5*mat3)

    return dzGe, dzGm
end

@doc raw"""
    denormalize_G_e(Ge,knorm)
Passes from a dimensionless electric green tensor `Ge` to a green tensor with units of length⁻¹. `knorm` is the wavenumber into the medium.
"""
function denormalize_G_e(Ge,knorm)
    return (knorm/4/pi).*Ge
end

@doc raw"""
    denormalize_G_m(Gm,knorm)
Passes from a dimensionless magnetic green tensor `Gm` to a green tensor with units of length⁻². `knorm` is the wavenumber into the medium.
"""
function denormalize_G_m(Gm,knorm)
    return (knorm^2/4/pi).*Gm
end
end
