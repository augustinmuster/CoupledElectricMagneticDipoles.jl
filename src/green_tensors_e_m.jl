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
    dxG_e_renorm(kr1,kr2)
Compute the derivative of the renormalized green tensor (defined in G_e_renorm(kr1,kr2)) regarding the k*x component of `kr1` between two position  `kr1` and `kr2`.
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
Compute the derivative of the renormalized green tensor (defined in G_e_renorm(kr1,kr2)) regarding the k*y component of `kr1` between two position  `kr1` and `kr2`.
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
Compute the derivative of the renormalized green tensor (defined in G_e_renorm(kr1,kr2)) regarding the k*z component of `kr1` between two position  `kr1` and `kr2`.
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
Compute the derivative of the renormalized magnetic green tensor (defined in G_m_renorm(kr1,kr2)) regarding the k*x component of `kr1` between two position  `kr1` and `kr2`.
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
Compute the derivative of the renormalized magnetic green tensor (defined in G_m_renorm(kr1,kr2)) regarding the k*y component of `kr1` between two position  `kr1` and `kr2`.
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
Compute the derivative of the renormalized magnetic green tensor (defined in G_m_renorm(kr1,kr2)) regarding the k*z component of `kr1` between two position  `kr1` and `kr2`.
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
Compute the x-derivative of the renormalized electric and magnetic green tensor regarding the components of `kr1` between two position  `kr1` and `kr2`.
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
Compute the y-derivative of the renormalized electric and magnetic green tensor regarding the components of `kr1` between two position  `kr1` and `kr2`.
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
Compute the z-derivative of the renormalized electric and magnetic green tensor regarding the components of `kr1` between two position  `kr1` and `kr2`.
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
    #numerical one
    num=(G_m_renorm(x1bis,x2)-G_m_renorm(x1,x2))/(h*1e-6)
    #numerical highest order
    num2=-G_m_renorm([1,2,x+2*h]*1e-6,x2)+8*G_m_renorm([1,2,x+h]*1e-6,x2)-8*G_m_renorm([1,2,x-h]*1e-6,x2)+G_m_renorm([1,2,x-2*h]*1e-6,x2)
    num2=num2/12/h/1e-6
    #analytical one
    func=dzG_m_renorm(x1,x2)
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
    #numerical one
    num=(G_e_renorm(x1bis,x2)-G_e_renorm(x1,x2))/(h*1e-6)
    #numerical highest order
    num2=-G_e_renorm([x+2*h,2,3]*1e-6,x2)+8*G_e_renorm([x+h,2,3]*1e-6,x2)-8*G_e_renorm([x-h,2,3]*1e-6,x2)+G_e_renorm([x-2*h,2,3]*1e-6,x2)
    num2=num2/12/h/1e-6
    #analytical one
    func=dxG_e_renorm(x1,x2)

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
    #numerical one
    num=(G_e_renorm(x1bis,x2)-G_e_renorm(x1,x2))/(h*1e-6)
    #numerical highest order
    num2=-G_e_renorm([1,x+2*h,3]*1e-6,x2)+8*G_e_renorm([1,x+h,3]*1e-6,x2)-8*G_e_renorm([1,x-h,3]*1e-6,x2)+G_e_renorm([1,x-2*h,3]*1e-6,x2)
    num2=num2/12/h/1e-6
    #analytical one
    func=dyG_e_renorm(x1,x2)
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
    #numerical one
    num=(G_e_renorm(x1bis,x2)-G_e_renorm(x1,x2))/(h*1e-6)
    #numerical highest order
    num2=-G_e_renorm([1,2,x+2*h]*1e-6,x2)+8*G_e_renorm([1,2,x+h]*1e-6,x2)-8*G_e_renorm([1,2,x-h]*1e-6,x2)+G_e_renorm([1,2,x-2*h]*1e-6,x2)
    num2=num2/12/h/1e-6
    #analytical one
    func=dzG_e_renorm(x1,x2)
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
