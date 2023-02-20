module GreenTensors
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#code to compute the magnetic green tensor and it's derivative
#author: Augustin Muster
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#***********************
#imports
#***********************
using Base
using LinearAlgebra
using Test
#if you want to test the funcitons
testing=false

@doc raw"""
    G_em(r1,r2,k)
Compute the electric and magnetic green tensor between two position `r1` and `r2` (`r2` the origin and `r1` the observation) .
The output are two 3x3 complex matrix.
The electric green tensor (with unit of [1/m]) is defined as:
```math
\tilde{G}_e=\left(\vec{r_1},\vec{r_2},k\right)=\frac{e^{ikr}}{4 \pi r}\left(\frac{(kr)^2+ikr-1}{(kr)^2}I+\frac{-(kr)^2-3ikr+3}{(kr)^2}\vec{u_r}\otimes\vec{u_r}\right)
```
The magnetic green tensor (with unit of [1/m]) is defined as:
```math
\tilde{G}_m/k=\left(\vec{r_1},\vec{r_2},k\right)=\frac{e^{ikr}}{4 \pi r}\left(\frac{ikr-1}{ikr}\right)\vec{u_r}
```
"""
function G_em(r1,r2,k)

    kr1 = k*r1
    kr2 = k*r2
    kR_vec = kr1-kr2
    kR = norm(kR_vec)
    kR2 = kR^2
    Ur=kR_vec/kR

    term1=exp(im*kR)/(4*pi*kR)*k
    term2=1+(im/(kR))-(1/(kR2))
    term3=1+(3*im/(kR))-(3/(kR2))
    term4=(im*kR-1)/kR
    matrix=kR_vec*transpose(kR_vec)/kR2
    id3=[1 0 0;0 1 0;0 0 1]
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]

    Ge = term1*(term2*id3-term3*matrix) 	#same as G_e(r1,r2,knorm)
    Gm = term1*term4*mat			#same as G_m(r1,r2,knorm)/knorm

    return Ge, Gm
end


@doc raw"""
    G_em_renorm(kr1,kr2)
Compute the electric and magnetic green tensor between two position `r1` and `r2`, where the imputs are the positions multiplied by the wave number `kr1` and `kr2` (->dimensionless input).
The output are two 3x3 complex matrix.
The electric green tensor (with unit of [1]) is defined as:
```math
G_e=\frac{4*\pi}{k}\tilde{G}_e
```
The magnetic green tensor (with unit of [1]) is defined as:
```math
G_m=\frac{4*\pi}{k^2}\tilde{G}_m
```
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
    G_m(r1,r2,knorm)
Compute the magnetic green tensor between two position  `r1` and `r2` with wavenumber `knorm`.
The output is a 3x3 complex matrix.
The magnetic green tensor (with units [1/m^2]) is defined as:
```math
\tilde{G}_m=\left(\vec{r_1},\vec{r_2},k\right)=\frac{e^{ikr}}{4 \pi r}k\left(\frac{ikr-1}{ikr}\right)\vec{u_r}
```
with
```math
r=|r_1-r_2|, \vec{u_r}=\left(r_1-r_2\right)/r
```
"""
function G_m(r1,r2,knorm)
    #difference vector
    R_vec=r1-r2
    R=norm(R_vec)
    Ur=R_vec/R
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    #terms
    term1=exp(im*knorm*R)/4/pi/R
    term2=(im*knorm*R-1)/knorm/R
    #return green tensor
    return term1*term2*mat*knorm
end


@doc raw"""
    G_e(r1,r2,knorm)
Compute the electric green tensor between two position  `r1` and `r2` with wavenumber `knorm`.
The output is a 3x3 complex matrix.
The electric green tensor (with units [1/m]) is defined as:
```math
\tilde{G}_e=\left(\vec{r_1},\vec{r_2},k\right)=\frac{e^{ikr}}{4 \pi r}\left(\frac{(kr)^2+ikr-1}{(kr)^2}I+\frac{-(kr)^2-3ikr+3}{(kr)^2}\vec{u_r}\otimes\vec{u_r}\right)
```
with
```math
r=|r_1-r_2|, \vec{u_r}=\left(r_1-r_2\right)/r
```
"""
function G_e(r1,r2,knorm)
    R_vec=r1-r2
    R=norm(R_vec)
    term1=exp(im*knorm*R)/(4*pi*R)
    term2=1+(im/(knorm*R))-(1/(knorm^2*R^2))
    term3=1+(3*im/(knorm*R))-(3/(knorm^2*R^2))
    matrix=R_vec*transpose(R_vec)/R^2
    id3=[1 0 0;0 1 0;0 0 1]
    return term1*(term2*id3-term3*matrix)
end

@doc raw"""
    G_m_renorm(kr1,kr2)
Compute the magnetic green tensor in renormalized units (see Home page) between two position multiplied by the wave number `kr1` and `kr2` (->dimensionless input).
The output isd a 3x3 complex matrix.
The renormalized magnetic green tensor (with units [1]) is defined as:
```math
G_m=\frac{4*\pi}{k^2}\tilde{G}_m
```
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
Compute the electric green tensor in renormalized units (see Home page) between two position multiplied by the wave number `kr1` and `kr2` (->dimensionless input).
The output is a 3x3 complex matrix.
The renormalized electric green tensor (with units [1]) is defined as:
```math
G_e=\frac{4*\pi}{k}\tilde{G}_e
```
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
    dxG_e(r1,r2,knorm)
Compute the derivative electric green tensor (defined in G_e(r1,r2,knorm)) regarding the x component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix with units [1/m^2]
"""
function dxG_e(r1,r2,knorm)

    R = r1-r2
    r = norm(R)
    x = R[1]
    y = R[2]
    z = R[3]

    term1 = exp(im*knorm*r)/(4*pi*r)
    term2 = knorm*(im - 2/(knorm*r) - 3*im/(knorm*r)^2 + 3/(knorm*r)^3)*x/r
    term3 = - knorm*(im - 6/(knorm*r) - 15*im/(knorm*r)^2 + 15/(knorm*r)^3)*x/r
    term4 = - (1 + 3*im/(knorm*r) - 3/(knorm*r)^2)/r^2
    Rmx = [2*x y z;y 0 0;z 0 0]
    matrix=R*transpose(R)/r^2
    id3=[1 0 0;0 1 0;0 0 1]

    return term1*(term2*id3 + term3*matrix + term4*Rmx)

end

@doc raw"""
    dyG_e(r1,r2,knorm)
Compute the derivative electric green tensor (defined in G_e(r1,r2,knorm)) regarding the y component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix with units [1/m^2]
"""
function dyG_e(r1,r2,knorm)

    R = r1 - r2
    r = norm(R)
    x = R[1]
    y = R[2]
    z = R[3]

    term1 = exp(im*knorm*r)/(4*pi*r)
    term2 = knorm*(im - 2/(knorm*r) - 3*im/(knorm*r)^2 + 3/(knorm*r)^3)*y/r
    term3 = - knorm*(im - 6/(knorm*r) - 15*im/(knorm*r)^2 + 15/(knorm*r)^3)*y/r
    term4 = - (1 + 3*im/(knorm*r) - 3/(knorm*r)^2)/r^2
    Rmy = [0 x 0;x 2*y z;0 z 0]
    matrix=R*transpose(R)
    id3=[1 0 0;0 1 0;0 0 1]

    return term1*(term2*id3 + term3*matrix/r^2 + term4*Rmy)

end

@doc raw"""
    dzG_e(r1,r2,knorm)
Compute the derivativeelectric green tensor (defined in G_e(r1,r2,knorm)) regarding the z component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix with units [1/m^2]
"""
function dzG_e(r1,r2,knorm)

    R = r1 - r2
    r = norm(R)
    x = R[1]
    y = R[2]
    z = R[3]

    term1 = exp(im*knorm*r)/(4*pi*r)
    term2 = knorm*(im - 2/(knorm*r) - 3*im/(knorm*r)^2 + 3/(knorm*r)^3)*z/r
    term3 = - knorm*(im - 6/(knorm*r) - 15*im/(knorm*r)^2 + 15/(knorm*r)^3)*z/r
    term4 = - (1 + 3*im/(knorm*r) - 3/(knorm*r)^2)/r^2
    Rmz = [0 0 x;0 0 y;x y 2*z]
    matrix=R*transpose(R)
    id3=[1 0 0;0 1 0;0 0 1]

    return term1*(term2*id3 + term3*matrix/r^2 + term4*Rmz)

end

@doc raw"""
    dxG_m(r1,r2,knorm)
Compute the derivative magnetic green tensor (defined in G_m(r1,r2,knorm)) regarding the x component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix with units [1/m^3]
"""
function dxG_m(r1,r2,knorm)
    #difference vector
    R_vec=r1-r2
    R=norm(R_vec)
    Ur=R_vec/R
    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 0 0;0 0 -1;0 1 0]
    #terms
    term1=exp(im*knorm*R)/4/pi/(R^3)
    term2=(im*knorm*R-1)
    term3=(im*knorm*R)^2 - 3*im*knorm*R + 3
    #return green tensor
    return term1*(term3*mat*Ur[1] + term2*mat3)
end


@doc raw"""
    dyG_m(r1,r2,knorm)
Compute the derivative magnetic green tensor (defined in G_m(r1,r2,knorm)) regarding the y component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix with units [1/m^3]
"""
function dyG_m(r1,r2,knorm)
    #difference vector
    R_vec=r1-r2
    R=norm(R_vec)
    Ur=R_vec/R
    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 0 1;0 0 0;-1 0 0]
    #terms
    term1=exp(im*knorm*R)/4/pi/(R^3)
    term2=(im*knorm*R-1)
    term3=(im*knorm*R)^2 - 3*im*knorm*R + 3
    #return green tensor
    return term1*(term3*mat*Ur[2] + term2*mat3)
end


@doc raw"""
    dzG_m(r1,r2,knorm)
Compute the derivative magnetic green tensor (defined in G_m(r1,r2,knorm)) regarding the z component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix with units [1/m^3]
"""
function dzG_m(r1,r2,knorm)
    #difference vector
    R_vec=r1-r2
    R=norm(R_vec)
    Ur=R_vec/R
    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 -1 0;1 0 0;0 0 0]
    #terms
    term1=exp(im*knorm*R)/4/pi/(R^3)
    term2=(im*knorm*R-1)
    term3=(im*knorm*R)^2 - 3*im*knorm*R + 3
    #return green tensor
    return term1*(term3*mat*Ur[3] + term2*mat3)
end

@doc raw"""
    dxG_em(r1,r2,knorm)
Compute the derivative green tensor (defined in G_e(r1,r2,knorm) and G_m(r1,r2,knorm)/knorm) regarding the x component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output are two 3x3 complex matrix (with units of [1/m^2]).
"""
function dxG_em(r1,r2,knorm)

    R_vec = r1-r2
    R = norm(R_vec)
    Ur=R_vec/R

    x = R_vec[1]
    y = R_vec[2]
    z = R_vec[3]

    term1 = exp(im*knorm*R)/(4*pi*R)
    term2 = knorm*(im - 2/(knorm*R) - 3*im/(knorm*R)^2 + 3/(knorm*R)^3)*x/R
    term3 = - knorm*(im - 6/(knorm*R) - 15*im/(knorm*R)^2 + 15/(knorm*R)^3)*x/R
    term4 = - (1 + 3*im/(knorm*R) - 3/(knorm*R)^2)/R^2
    Rmx = [2*x y z;y 0 0;z 0 0]
    matrix=R_vec*transpose(R_vec)/R^2
    id3=[1 0 0;0 1 0;0 0 1]

    dxGe =  term1*(term2*id3 + term3*matrix + term4*Rmx)	# same as dxG_e(r1,r2,knorm)

    #difference vector

    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 0 0;0 0 -1;0 1 0]
    #terms
    term5=(im*knorm*R-1)
    term6=(im*knorm*R)^2 - 3*im*knorm*R + 3
    #return green tensor
    dxGm = term1/R^2*(term6*mat*Ur[1] + term5*mat3)/knorm		# same as dxG_m(r1,r2,knorm)/knorm

    return dxGe, dxGm
end

@doc raw"""
    dyG_em(r1,r2,knorm)
Compute the derivative green tensor (defined in G_e(r1,r2,knorm) and G_m(r1,r2,knorm)/knorm) regarding the y component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output are two 3x3 complex matrix (with units of [1/m^2]).
"""
function dyG_em(r1,r2,knorm)

    R_vec = r1-r2
    R = norm(R_vec)
    Ur=R_vec/R

    x = R_vec[1]
    y = R_vec[2]
    z = R_vec[3]

    term1 = exp(im*knorm*R)/(4*pi*R)
    term2 = knorm*(im - 2/(knorm*R) - 3*im/(knorm*R)^2 + 3/(knorm*R)^3)*y/R
    term3 = - knorm*(im - 6/(knorm*R) - 15*im/(knorm*R)^2 + 15/(knorm*R)^3)*y/R
    term4 = - (1 + 3*im/(knorm*R) - 3/(knorm*R)^2)/R^2
    Rmy = [0 x 0;x 2*y z;0 z 0]
    matrix=R_vec*transpose(R_vec)/R^2
    id3=[1 0 0;0 1 0;0 0 1]

    dyGe =  term1*(term2*id3 + term3*matrix + term4*Rmy)	# same as dxG_e(r1,r2,knorm)

    #difference vector

    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 0 1;0 0 0;-1 0 0]
    #terms
    term5=(im*knorm*R-1)
    term6=(im*knorm*R)^2 - 3*im*knorm*R + 3
    #return green tensor
    dyGm = term1/R^2*(term6*mat*Ur[2] + term5*mat3)/knorm		# same as dxG_m(r1,r2,knorm)/knorm

    return dyGe, dyGm
end

@doc raw"""
    dzG_em(r1,r2,knorm)
Compute the derivative green tensor (defined in G_e(r1,r2,knorm) and G_m(r1,r2,knorm)/knorm) regarding the x component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output are two 3x3 complex matrix (with units of [1/m^2]).
"""
function dzG_em(r1,r2,knorm)

    R_vec = r1-r2
    R = norm(R_vec)
    Ur=R_vec/R

    x = R_vec[1]
    y = R_vec[2]
    z = R_vec[3]

    term1 = exp(im*knorm*R)/(4*pi*R)
    term2 = knorm*(im - 2/(knorm*R) - 3*im/(knorm*R)^2 + 3/(knorm*R)^3)*z/R
    term3 = - knorm*(im - 6/(knorm*R) - 15*im/(knorm*R)^2 + 15/(knorm*R)^3)*z/R
    term4 = - (1 + 3*im/(knorm*R) - 3/(knorm*R)^2)/R^2
    Rmz = [0 0 x;0 0 y;x y 2*z]
    matrix=R_vec*transpose(R_vec)/R^2
    id3=[1 0 0;0 1 0;0 0 1]

    dzGe =  term1*(term2*id3 + term3*matrix + term4*Rmz)	# same as dxG_e(r1,r2,knorm)

    #difference vector

    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #cross product matrix
    mat=[0 -Ur[3] Ur[2];Ur[3] 0 -Ur[1];-Ur[2] Ur[1] 0]
    mat3=[0 -1 0;1 0 0;0 0 0]
    #terms
    term5=(im*knorm*R-1)
    term6=(im*knorm*R)^2 - 3*im*knorm*R + 3
    #return green tensor
    dzGm = term1/R^2*(term6*mat*Ur[3] + term5*mat3)/knorm		# same as dxG_m(r1,r2,knorm)/knorm

    return dzGe, dzGm
end

@doc raw"""
    Sigma(n)
Sigma represent the base-change-matrix [E, ZH] to [E, iZH]
"""
function Sigma(n)

    Sigma = zeros(ComplexF64,6*n,6*n)
    Sigma_i = zeros(ComplexF64,6*n,6)
    id = [1 0 0;0 1 0;0 0 1]

    for i = 1:n
        Sigma_i[(i-1)*6+1:(i-1)*6+6,:] = [id id*0; id*0 im*id]
    end
    for i = 1:n
        Sigma[:,(i-1)*6+1:(i-1)*6+6] = copy(Sigma_i)
    end
    return Sigma

end


@doc raw"""
    G_e_m(r1,r2,knorm)
Compute the big electric and magnetic green tensor between two position  `r1` and `r2` with wavenumber `knorm`.
The output is a 6x6 complex matrix.
The big electric and magnetic green tensor is defined as:
```math
 \tilde{G}_{em}=\left(\begin{matrix}
 \tilde{G}_e &  \tilde{G}_m/k\\
 \tilde{G}_m/k &  \tilde{G}_e
\end{matrix}\right)
```
"""
function G_e_m(r1,r2,knorm)
    Gem=zeros(ComplexF64,6,6)
    Gem[1:3,1:3]=G_e(r1,r2,knorm)
    Gem[4:6,4:6]=G_e(r1,r2,knorm)
    Gem[1:3,4:6]=G_m(r1,r2,knorm)/knorm
    Gem[4:6,1:3]=G_m(r1,r2,knorm)/knorm
    return Gem
end


@doc raw"""
    G_e_m_renorm(r1,r2,knorm)
Compute the big electric and magnetic green tensor in renormalized units (see Home page) between two position  `r1` and `r2` with wavenumber `knorm`.
The output is a 6x6 complex matrix.
The big electric and magnetic green tensor in renormalized units is defined as:
```math
 G_{em}=\left(\begin{matrix}
 G_e &  G_m\\
 G_m &  G_e
\end{matrix}\right)
```
"""
function G_e_m_renorm(r1,r2)
    Gem=zeros(ComplexF64,6,6)
    Gem[1:3,1:3]=G_e_renorm(r1,r2)
    Gem[4:6,4:6]=G_e_renorm(r1,r2)
    Gem[1:3,4:6]=G_m_renorm(r1,r2)
    Gem[4:6,1:3]=G_m_renorm(r1,r2)
    return Gem
end

function dxG_e_m_renorm(r1,r2,knorm)
    Gem=zeros(ComplexF64,6,6)
    Gem[1:3,1:3]=dxG_e(r1,r2,knorm)*4*pi/knorm
    Gem[4:6,4:6]=dxG_e(r1,r2,knorm)*4*pi/knorm
    Gem[1:3,4:6]=dxG_m(r1,r2,knorm)*4*pi/knorm^2
    Gem[4:6,1:3]=dxG_m(r1,r2,knorm)*4*pi/knorm^2
    return Gem
end

function dyG_e_m_renorm(r1,r2,knorm)
    Gem=zeros(ComplexF64,6,6)
    Gem[1:3,1:3]=dyG_e(r1,r2,knorm)*4*pi/knorm
    Gem[4:6,4:6]=dyG_e(r1,r2,knorm)*4*pi/knorm
    Gem[1:3,4:6]=dyG_m(r1,r2,knorm)*4*pi/knorm^2
    Gem[4:6,1:3]=dyG_m(r1,r2,knorm)*4*pi/knorm^2
    return Gem
end

function dzG_e_m_renorm(r1,r2,knorm)
    Gem=zeros(ComplexF64,6,6)
    Gem[1:3,1:3]=dzG_e(r1,r2,knorm)*4*pi/knorm
    Gem[4:6,4:6]=dzG_e(r1,r2,knorm)*4*pi/knorm
    Gem[1:3,4:6]=dzG_m(r1,r2,knorm)*4*pi/knorm^2
    Gem[4:6,1:3]=dzG_m(r1,r2,knorm)*4*pi/knorm^2
    return Gem
end
@doc raw"""
    dxG_e_m(r1,r2,knorm)
Compute the derivative of the big electric and magnetic green tensor regarding the x component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 6x6 complex matrix
"""
function dxG_e_m(r1,r2,knorm)
    Gem=zeros(ComplexF64,6,6)
    Gem[1:3,1:3]=dxG_e(r1,r2,knorm)
    Gem[4:6,4:6]=dxG_e(r1,r2,knorm)
    Gem[1:3,4:6]=dxG_m(r1,r2,knorm)/knorm
    Gem[4:6,1:3]=dxG_m(r1,r2,knorm)/knorm
    return Gem
end

@doc raw"""
    dyG_e_m(r1,r2,knorm)
Compute the derivative of the big electric and magnetic green tensor regarding the y component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 6x6 complex matrix
"""
function dyG_e_m(r1,r2,knorm)
    Gem=zeros(ComplexF64,6,6)
    Gem[1:3,1:3]=dyG_e(r1,r2,knorm)
    Gem[4:6,4:6]=dyG_e(r1,r2,knorm)
    Gem[1:3,4:6]=dyG_m(r1,r2,knorm)/knorm
    Gem[4:6,1:3]=dyG_m(r1,r2,knorm)/knorm
    return Gem
end

@doc raw"""
    dzG_e_m(r1,r2,knorm)
Compute the derivative of the big electric and magnetic green tensor regarding the z component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 6x6 complex matrix
"""
function dzG_e_m(r1,r2,knorm)
    Gem=zeros(ComplexF64,6,6)
    Gem[1:3,1:3]=dzG_e(r1,r2,knorm)
    Gem[4:6,4:6]=dzG_e(r1,r2,knorm)
    Gem[1:3,4:6]=dzG_m(r1,r2,knorm)/knorm
    Gem[4:6,1:3]=dzG_m(r1,r2,knorm)/knorm
    return Gem
end


if testing
    ###########################
    #TESTS
    ###########################
    println("")
    println("*************************")
    println("Testing")
    println("*************************")
    println("")
    #***********************
    #TEST THE X DERIVATIVE OF THE MAGNETIC GREEN TENSOR
    #***********************
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[x,2,3]*1e-6
    x1bis=[x+h,2,3]*1e-6
    #
    x2=[2,3,4]*1e-6
    knorm=1e9
    #numerical one
    num=(G_m(x1bis,x2,knorm)-G_m(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-G_m([x+2*h,2,3]*1e-6,x2,knorm)+8*G_m([x+h,2,3]*1e-6,x2,knorm)-8*G_m([x-h,2,3]*1e-6,x2,knorm)+G_m([x-2*h,2,3]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=dxG_m(x1,x2,knorm)

    println("testing the function for the x-derivative of the magnetic green tensor...")
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<1e-6
    @test real((num2-func)[1,3])/real(num2[1,3])<1e-6
    @test real((num2-func)[2,1])/real(num2[2,1])<1e-6
    @test real((num2-func)[2,3])/real(num2[2,3])<1e-6
    @test real((num2-func)[3,1])/real(num2[3,1])<1e-6
    @test real((num2-func)[3,2])/real(num2[3,2])<1e-6
    @test imag((num2-func)[1,2])/imag(num2[1,2])<1e-6
    @test imag((num2-func)[1,3])/imag(num2[1,3])<1e-6
    @test imag((num2-func)[2,1])/imag(num2[2,1])<1e-6
    @test imag((num2-func)[2,3])/imag(num2[2,3])<1e-6
    @test imag((num2-func)[3,1])/imag(num2[3,1])<1e-6
    @test imag((num2-func)[3,2])/imag(num2[3,2])<1e-6
    println("test passed ;) ")
    println("")


    #***********************
    #TEST THE Y DERIVATIVE OF THE MAGNETIC GREEN TENSOR
    #***********************
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[1,x,3]*1e-6
    x1bis=[1,x+h,3]*1e-6
    #
    x2=[7,8,9]*1e-6
    knorm=1e9
    #numerical one
    num=(G_m(x1bis,x2,knorm)-G_m(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-G_m([1,x+2*h,3]*1e-6,x2,knorm)+8*G_m([1,x+h,3]*1e-6,x2,knorm)-8*G_m([1,x-h,3]*1e-6,x2,knorm)+G_m([1,x-2*h,3]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=dyG_m(x1,x2,knorm)
    println("")
    println("testing the function for the y-derivative of the magnetic green tensor...")
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<1e-6
    @test real((num2-func)[1,3])/real(num2[1,3])<1e-6
    @test real((num2-func)[2,1])/real(num2[2,1])<1e-6
    @test real((num2-func)[2,3])/real(num2[2,3])<1e-6
    @test real((num2-func)[3,1])/real(num2[3,1])<1e-6
    @test real((num2-func)[3,2])/real(num2[3,2])<1e-6
    @test imag((num2-func)[1,2])/imag(num2[1,2])<1e-6
    @test imag((num2-func)[1,3])/imag(num2[1,3])<1e-6
    @test imag((num2-func)[2,1])/imag(num2[2,1])<1e-6
    @test imag((num2-func)[2,3])/imag(num2[2,3])<1e-6
    @test imag((num2-func)[3,1])/imag(num2[3,1])<1e-6
    @test imag((num2-func)[3,2])/imag(num2[3,2])<1e-6
    println("test passed ;) ")
    println("")

    #***********************
    #TEST THE Z DERIVATIVE OF THE MAGNETIC GREEN TENSOR
    #***********************
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[1,2,x]*1e-6
    x1bis=[1,2,x+h]*1e-6
    #
    x2=[7,8,9]*1e-6
    knorm=1e9
    #numerical one
    num=(G_m(x1bis,x2,knorm)-G_m(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-G_m([1,2,x+2*h]*1e-6,x2,knorm)+8*G_m([1,2,x+h]*1e-6,x2,knorm)-8*G_m([1,2,x-h]*1e-6,x2,knorm)+G_m([1,2,x-2*h]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=dzG_m(x1,x2,knorm)
    println("")
    println("testing the function for the z-derivative of the magnetic green tensor...")
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<1e-6
    @test real((num2-func)[1,3])/real(num2[1,3])<1e-6
    @test real((num2-func)[2,1])/real(num2[2,1])<1e-6
    @test real((num2-func)[2,3])/real(num2[2,3])<1e-6
    @test real((num2-func)[3,1])/real(num2[3,1])<1e-6
    @test real((num2-func)[3,2])/real(num2[3,2])<1e-6
    @test imag((num2-func)[1,2])/imag(num2[1,2])<1e-6
    @test imag((num2-func)[1,3])/imag(num2[1,3])<1e-6
    @test imag((num2-func)[2,1])/imag(num2[2,1])<1e-6
    @test imag((num2-func)[2,3])/imag(num2[2,3])<1e-6
    @test imag((num2-func)[3,1])/imag(num2[3,1])<1e-6
    @test imag((num2-func)[3,2])/imag(num2[3,2])<1e-6
    println("test passed ;) ")
    println("")

    #***********************
    #TEST THE X DERIVATIVE OF THE ELECTRIC GREEN TENSOR
    #***********************
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[x,2,3]*1e-6
    x1bis=[x+h,2,3]*1e-6
    #
    x2=[2,3,4]*1e-6
    knorm=1e9
    #numerical one
    num=(G_e(x1bis,x2,knorm)-G_e(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-G_e([x+2*h,2,3]*1e-6,x2,knorm)+8*G_e([x+h,2,3]*1e-6,x2,knorm)-8*G_e([x-h,2,3]*1e-6,x2,knorm)+G_e([x-2*h,2,3]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=dxG_e(x1,x2,knorm)

    println("testing the function for the x-derivative of the electric green tensor...")
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<1e-6
    @test real((num2-func)[1,3])/real(num2[1,3])<1e-6
    @test real((num2-func)[2,1])/real(num2[2,1])<1e-6
    @test real((num2-func)[2,3])/real(num2[2,3])<1e-6
    @test real((num2-func)[3,1])/real(num2[3,1])<1e-6
    @test real((num2-func)[3,2])/real(num2[3,2])<1e-6
    @test imag((num2-func)[1,2])/imag(num2[1,2])<1e-6
    @test imag((num2-func)[1,3])/imag(num2[1,3])<1e-6
    @test imag((num2-func)[2,1])/imag(num2[2,1])<1e-6
    @test imag((num2-func)[2,3])/imag(num2[2,3])<1e-6
    @test imag((num2-func)[3,1])/imag(num2[3,1])<1e-6
    @test imag((num2-func)[3,2])/imag(num2[3,2])<1e-6
    println("test passed ;) ")
    println("")


    #***********************
    #TEST THE Y DERIVATIVE OF THE ELECTRIC GREEN TENSOR
    #***********************
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[1,x,3]*1e-6
    x1bis=[1,x+h,3]*1e-6
    #
    x2=[7,8,9]*1e-6
    knorm=1e9
    #numerical one
    num=(G_e(x1bis,x2,knorm)-G_e(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-G_e([1,x+2*h,3]*1e-6,x2,knorm)+8*G_e([1,x+h,3]*1e-6,x2,knorm)-8*G_e([1,x-h,3]*1e-6,x2,knorm)+G_e([1,x-2*h,3]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=dyG_e(x1,x2,knorm)
    println("")
    println("testing the function for the y-derivative of the electric green tensor...")
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<1e-6
    @test real((num2-func)[1,3])/real(num2[1,3])<1e-6
    @test real((num2-func)[2,1])/real(num2[2,1])<1e-6
    @test real((num2-func)[2,3])/real(num2[2,3])<1e-6
    @test real((num2-func)[3,1])/real(num2[3,1])<1e-6
    @test real((num2-func)[3,2])/real(num2[3,2])<1e-6
    @test imag((num2-func)[1,2])/imag(num2[1,2])<1e-6
    @test imag((num2-func)[1,3])/imag(num2[1,3])<1e-6
    @test imag((num2-func)[2,1])/imag(num2[2,1])<1e-6
    @test imag((num2-func)[2,3])/imag(num2[2,3])<1e-6
    @test imag((num2-func)[3,1])/imag(num2[3,1])<1e-6
    @test imag((num2-func)[3,2])/imag(num2[3,2])<1e-6
    println("test passed ;) ")
    println("")

    #***********************
    #TEST THE Z DERIVATIVE OF THE ELECTRIC GREEN TENSOR
    #***********************
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[1,2,x]*1e-6
    x1bis=[1,2,x+h]*1e-6
    #
    x2=[7,8,9]*1e-6
    knorm=1e9
    #numerical one
    num=(G_e(x1bis,x2,knorm)-G_e(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-G_e([1,2,x+2*h]*1e-6,x2,knorm)+8*G_e([1,2,x+h]*1e-6,x2,knorm)-8*G_e([1,2,x-h]*1e-6,x2,knorm)+G_e([1,2,x-2*h]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=dzG_e(x1,x2,knorm)
    println("")
    println("testing the function for the z-derivative of the electric green tensor...")
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<1e-6
    @test real((num2-func)[1,3])/real(num2[1,3])<1e-6
    @test real((num2-func)[2,1])/real(num2[2,1])<1e-6
    @test real((num2-func)[2,3])/real(num2[2,3])<1e-6
    @test real((num2-func)[3,1])/real(num2[3,1])<1e-6
    @test real((num2-func)[3,2])/real(num2[3,2])<1e-6
    @test imag((num2-func)[1,2])/imag(num2[1,2])<1e-6
    @test imag((num2-func)[1,3])/imag(num2[1,3])<1e-6
    @test imag((num2-func)[2,1])/imag(num2[2,1])<1e-6
    @test imag((num2-func)[2,3])/imag(num2[2,3])<1e-6
    @test imag((num2-func)[3,1])/imag(num2[3,1])<1e-6
    @test imag((num2-func)[3,2])/imag(num2[3,2])<1e-6
    println("test passed ;) ")
    println("")
end
end
