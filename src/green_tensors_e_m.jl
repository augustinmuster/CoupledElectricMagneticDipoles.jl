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
    G_m(r1,r2,knorm)
Compute the magnetic green tensor between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix.
The magnetic green tensor is defined as:
```math
\tilde{G}_m\left(\vec{r_1},\vec{r_2},k\right)=\frac{e^{ikr}}{4 \pi r}k\left(\frac{ikr-1}{ikr}\right)\vec{u_r}
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
    term1=knorm*exp(im*knorm*R)/4/pi/R
    term2=(im*knorm*R-1)/knorm/R
    #return green tensor
    return term1*term2*mat
end
@doc raw"""
    dxG_m(r1,r2,knorm)
Compute the derivative magnetic green tensor regarding the x component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix
"""
function dxG_m(r1,r2,knorm)
    #difference vector
    R_vec=r1-r2
    R=norm(R_vec)
    Ur=R_vec/R
    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #compute all the off diagonal terms (the diagonal terms are equal to zero)
    #common terms
    t1=3*exp(im*knorm*R)*(-1+im*knorm*R)/R^5
    t2=im*exp(im*knorm*R)*knorm/R^4
    t3=im*exp(im*knorm*R)*knorm*(-1+im*knorm*R)/R^4
    t4=exp(im*knorm*R)*(-1+im*knorm*R)/R^3
    #component 12
    com=Ur[1]*Ur[3]*R*R
    m1=t1*com
    m2=-t2*com
    m3=-t3*com
    mat[1,2]=(m1+m2+m3)/4/pi
    #component 13
    com=Ur[1]*Ur[2]*R*R
    m1=-t1*com
    m2=t2*com
    m3=t3*com
    mat[1,3]=(m1+m2+m3)/4/pi
    #component 21
    com=Ur[1]*Ur[3]*R*R
    m1=-t1*com
    m2=t2*com
    m3=t3*com
    mat[2,1]=(m1+m2+m3)/4/pi
    #component 31
    com=Ur[1]*Ur[2]*R*R
    m1=t1*com
    m2=-t2*com
    m3=-t3*com
    mat[3,1]=(m1+m2+m3)/4/pi
    #component 23
    com=Ur[1]*Ur[1]*R*R
    m1=t1*com
    m2=-t2*com
    m3=-t3*com
    mat[2,3]=(m1+m2+m3-t4)/4/pi
    #component 32
    com=Ur[1]*Ur[1]*R*R
    m1=-t1*com
    m2=+t2*com
    m3=+t3*com
    mat[3,2]=(m1+m2+m3+t4)/4/pi
    #return the magnetic green tensor
    return mat
end

@doc raw"""
    dyG_m(r1,r2,knorm)
Compute the derivative magnetic green tensor regarding the y component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix
"""
function dyG_m(r1,r2,knorm)
    #difference vector
    R_vec=r1-r2
    R=norm(R_vec)
    Ur=R_vec/R
    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #compute all the off diagonal terms (the diagonal terms are equal to zero)
    #common terms
    t1=3*exp(im*knorm*R)*(-1+im*knorm*R)/R^5
    t2=im*exp(im*knorm*R)*knorm/R^4
    t3=im*exp(im*knorm*R)*knorm*(-1+im*knorm*R)/R^4
    t4=exp(im*knorm*R)*(-1+im*knorm*R)/R^3
    #component 12
    com=Ur[2]*Ur[3]*R*R
    m1=t1*com
    m2=-t2*com
    m3=-t3*com
    mat[1,2]=(m1+m2+m3)/4/pi
    #component 13
    com=Ur[2]*Ur[2]*R*R
    m1=-t1*com
    m2=t2*com
    m3=t3*com
    mat[1,3]=(m1+m2+m3+t4)/4/pi
    #component 21
    com=Ur[2]*Ur[3]*R*R
    m1=-t1*com
    m2=t2*com
    m3=t3*com
    mat[2,1]=(m1+m2+m3)/4/pi
    #component 31
    com=Ur[2]*Ur[2]*R*R
    m1=+t1*com
    m2=-t2*com
    m3=-t3*com
    mat[3,1]=(m1+m2+m3-t4)/4/pi
    #component 23
    com=Ur[1]*Ur[2]*R*R
    m1=t1*com
    m2=-t2*com
    m3=-t3*com
    mat[2,3]=(m1+m2+m3)/4/pi
    #component 32
    com=Ur[1]*Ur[2]*R*R
    m1=-t1*com
    m2=+t2*com
    m3=+t3*com
    mat[3,2]=(m1+m2+m3)/4/pi
    #return the magnetic green tensor
    return mat
end

@doc raw"""
    dzG_m(r1,r2,knorm)
Compute the derivative magnetic green tensor regarding the z component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix
"""
function dzG_m(r1,r2,knorm)
    #difference vector
    R_vec=r1-r2
    R=norm(R_vec)
    Ur=R_vec/R
    #create empty 3x3 matirx to store the green tensor
    mat=zeros(ComplexF64,3,3)
    #compute all the off diagonal terms (the diagonal terms are equal to zero)
    #common terms
    t1=3*exp(im*knorm*R)*(-1+im*knorm*R)/R^5
    t2=im*exp(im*knorm*R)*knorm/R^4
    t3=im*exp(im*knorm*R)*knorm*(-1+im*knorm*R)/R^4
    t4=exp(im*knorm*R)*(-1+im*knorm*R)/R^3
    #component 12
    com=Ur[3]*Ur[3]*R*R
    m1=t1*com
    m2=-t2*com
    m3=-t3*com
    mat[1,2]=(m1+m2+m3-t4)/4/pi
    #component 13
    com=Ur[2]*Ur[3]*R*R
    m1=-t1*com
    m2=t2*com
    m3=t3*com
    mat[1,3]=(m1+m2+m3)/4/pi
    #component 21
    com=Ur[3]*Ur[3]*R*R
    m1=-t1*com
    m2=t2*com
    m3=t3*com
    mat[2,1]=(m1+m2+m3+t4)/4/pi
    #component 31
    com=Ur[2]*Ur[3]*R*R
    m1=+t1*com
    m2=-t2*com
    m3=-t3*com
    mat[3,1]=(m1+m2+m3)/4/pi
    #component 23
    com=Ur[1]*Ur[3]*R*R
    m1=t1*com
    m2=-t2*com
    m3=-t3*com
    mat[2,3]=(m1+m2+m3)/4/pi
    #component 32
    com=Ur[1]*Ur[3]*R*R
    m1=-t1*com
    m2=+t2*com
    m3=+t3*com
    mat[3,2]=(m1+m2+m3)/4/pi
    #return the magnetic green tensor
    return mat
end

@doc raw"""
    G_e(r1,r2,knorm)
Compute the electric green tensor between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix.
The electric green tensor is defined as:
```math
\tilde{G}_e\left(\vec{r_1},\vec{r_2},k\right)=\frac{e^{ikr}}{4 \pi r}k\left(\frac{(kr)^2+ikr-1}{(kr)^2}I+\frac{-(kr)^2-3ikr+3}{(kr)^2}\vec{u_r}\vec{u_r}\right)
```
with
```math
r=|r_1-r_2|, \vec{u_r}=\left(r_1-r_2\right)/r
```
"""
function G_e(r1,r2,knorm)
    R=r1-r2
    term1=exp(im*knorm*norm(R))/(4*pi*norm(R))
    term2=1+(im/(knorm*norm(R)))-(1/(knorm^2*norm(R)^2))
    term3=1+(3*im/(knorm*norm(R)))-(3/(knorm^2*norm(R)^2))
    matrix=R*transpose(R)
    id3=[1 0 0;0 1 0;0 0 1]
    return term1*(term2*id3-term3*matrix/norm(R)^2)
end


@doc raw"""
    dxG_e(r1,r2,knorm)
Compute the derivative magnetic green tensor regarding the x component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix
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
    matrix=R*transpose(R)
    id3=[1 0 0;0 1 0;0 0 1]

    return term1*(term2*id3 + term3*matrix/r^2 + term4*Rmx)

end

@doc raw"""
    dyG_e(r1,r2,knorm)
Compute the derivative magnetic green tensor regarding the y component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix
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
Compute the derivative magnetic green tensor regarding the z component of `r1` between two position  `r1` and `r2` with wavenumber `knorm`.
The output isd a 3x3 complex matrix
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
    G_m_renorm(kr1,kr2)
Compute the magnetic green tensor in renormalized units (see Home page) between two position multiplied by the wave number `kr1` and `kr2` (->dimensionless input).
The output isd a 3x3 complex matrix.
The renormalized magnetic green tensor is defined as:
```math
G_m=\frac{4*\pi}{k}\tilde{G}_m
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
The renormalized electric green tensor is defined as:
```math
G_e=\frac{4*\pi}{k^2}\tilde{G}_e
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
