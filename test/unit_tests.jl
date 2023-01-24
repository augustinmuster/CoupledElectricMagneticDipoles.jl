#imports
using Test
using Base
using LinearAlgebra

#include the modules
println("Importing the library...")
include("../src/DDA.jl")
include("../src/alpha.jl")
include("../src/input_fields.jl")
include("../src/processing.jl")
include("../src/green_tensors_e_m.jl")
#FLAGS FOR TESTING DIFFERENT MODULES
TEST_GREENTENSORS=true
TEST_DDACORE=true


if TEST_GREENTENSORS
    ###########################
    #TESTS
    ###########################
    println("")
    println("*************************")
    println("Testing GreenTensors")
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
    num=(GreenTensors.G_m(x1bis,x2,knorm)-GreenTensors.G_m(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-GreenTensors.G_m([x+2*h,2,3]*1e-6,x2,knorm)+8*GreenTensors.G_m([x+h,2,3]*1e-6,x2,knorm)-8*GreenTensors.G_m([x-h,2,3]*1e-6,x2,knorm)+GreenTensors.G_m([x-2*h,2,3]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=GreenTensors.dxG_m(x1,x2,knorm)

    print("testing the function for the x-derivative of the magnetic green tensor: ")
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
    println("passed ")



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
    num=(GreenTensors.G_m(x1bis,x2,knorm)-GreenTensors.G_m(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-GreenTensors.G_m([1,x+2*h,3]*1e-6,x2,knorm)+8*GreenTensors.G_m([1,x+h,3]*1e-6,x2,knorm)-8*GreenTensors.G_m([1,x-h,3]*1e-6,x2,knorm)+GreenTensors.G_m([1,x-2*h,3]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=GreenTensors.dyG_m(x1,x2,knorm)

    print("testing the function for the y-derivative of the magnetic green tensor: ")
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
    println("passed")

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
    num=(GreenTensors.G_m(x1bis,x2,knorm)-GreenTensors.G_m(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-GreenTensors.G_m([1,2,x+2*h]*1e-6,x2,knorm)+8*GreenTensors.G_m([1,2,x+h]*1e-6,x2,knorm)-8*GreenTensors.G_m([1,2,x-h]*1e-6,x2,knorm)+GreenTensors.G_m([1,2,x-2*h]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=GreenTensors.dzG_m(x1,x2,knorm)

    print("testing the function for the z-derivative of the magnetic green tensor: ")
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
    println("passed")


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
    num=(GreenTensors.G_e(x1bis,x2,knorm)-GreenTensors.G_e(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-GreenTensors.G_e([x+2*h,2,3]*1e-6,x2,knorm)+8*GreenTensors.G_e([x+h,2,3]*1e-6,x2,knorm)-8*GreenTensors.G_e([x-h,2,3]*1e-6,x2,knorm)+GreenTensors.G_e([x-2*h,2,3]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=GreenTensors.dxG_e(x1,x2,knorm)

    print("testing the function for the x-derivative of the electric green tensor: ")
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
    println("passed")


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
    num=(GreenTensors.G_e(x1bis,x2,knorm)-GreenTensors.G_e(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-GreenTensors.G_e([1,x+2*h,3]*1e-6,x2,knorm)+8*GreenTensors.G_e([1,x+h,3]*1e-6,x2,knorm)-8*GreenTensors.G_e([1,x-h,3]*1e-6,x2,knorm)+GreenTensors.G_e([1,x-2*h,3]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=GreenTensors.dyG_e(x1,x2,knorm)

    print("testing the function for the y-derivative of the electric green tensor: ")
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
    println("passed")

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
    num=(GreenTensors.G_e(x1bis,x2,knorm)-GreenTensors.G_e(x1,x2,knorm))/(h*1e-6)
    #numerical highest order
    num2=-GreenTensors.G_e([1,2,x+2*h]*1e-6,x2,knorm)+8*GreenTensors.G_e([1,2,x+h]*1e-6,x2,knorm)-8*GreenTensors.G_e([1,2,x-h]*1e-6,x2,knorm)+GreenTensors.G_e([1,2,x-2*h]*1e-6,x2,knorm)
    num2=num2/12/h/1e-6
    #analytical one
    func=GreenTensors.dzG_e(x1,x2,knorm)

    print("testing the function for the z-derivative of the electric green tensor: ")
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
    println("passed")
end

if TEST_DDACORE
    ###########################
    #TESTS
    ###########################
    println("")
    println("*************************")
    println("Testing DDACore")
    println("*************************")
    println("")


    println("")
    println("***")
    println("Only electric DDA")
    println("***")
    println("")

    #------Consevation of energy--------
    #ŦEST: test if energy is conserved by the DDA process by comparing the cross sections
    print("Testing conservation of energy: ")
    #lattice parameter
    d=1e-8
    #position of one dipole
    r=[0 0 0;0 0 d]
    #dielectric constant
    e=(3.5+im*0.01)^2
    #wavelength
    lambda=1000e-9
    knorm=2*pi/lambda
    #computing polarisabilitites
    L=Alphas.depolarisation_tensor(d,d,d,d^3)

    a0=zeros(ComplexF64,length(r[:,1]),3,3)
    a=zeros(ComplexF64,length(r[:,1]),3,3)
    for i=1:length(r[:,1])
        a0[i,:,:]=Alphas.alpha_0(e,1,L,d^3)
        a[i,:,:]=Alphas.alpha_radiative(a0[i,:,:],knorm)
    end
    #DDA solving--dimensional inputs
    p,e_inc=DDACore.solve_DDA_e(knorm,r,a,InputFields.plane_wave,verbose=false)
    #computing cross sections
    res=PostProcessing.compute_cross_sections(knorm,r,p,e_inc,a0,verbose=false)
    #testing
    @test real(res[2])-real(res[3])-real(res[4])<10^(-10)
    #DDA solving--Adimensional inputs
    p,e_inc=DDACore.solve_DDA_e(knorm*r,a*knorm^3/4*pi,InputFields.plane_wave_renorm,verbose=false)
    #computing cross sections
    res=PostProcessing.compute_cross_sections(knorm,r,p,e_inc,a0,verbose=false)
    #testing
    @test real(res[2])-real(res[3])-real(res[4])<10^(-10)
    println("passed")

    #------Alpha dynamic input--------
    #ŦEST: test if alpha dynamic input is function
    print("Testing dynamic input of alpha: ")
    #scalar polarisability
    a_scalar=a[:,1,1]

    #DDA solving-tensor
    p,e_inc=DDACore.solve_DDA_e(knorm,r,a,InputFields.plane_wave,verbose=false)
    #DDA solving-tensor
    p_scalar,e_inc_scalar=DDACore.solve_DDA_e(knorm,r,a_scalar,InputFields.plane_wave,verbose=false)
    #testing
    @test maximum(abs.(p).-abs.(p_scalar))<1e-10
    @test maximum(abs.(e_inc).-abs.(e_inc_scalar))<1e-10
    println("passed")

    #------Dimentionless with or without input field get the same--------
    #ŦEST: test if
    print("Testing that dimensional and adimensional version gives the same: ")
    #with input field
    #DDA solving-tensor
    p,e_inc=DDACore.solve_DDA_e(knorm,r,a,InputFields.plane_wave,verbose=false)
    #DDA solving-tensor
    p_dl,e_inc_dl=DDACore.solve_DDA_e(knorm*r,knorm^3/4/pi*a,InputFields.plane_wave_renorm,verbose=false)
    #testing
    @test maximum(abs.(knorm^3/4/pi*p).-abs.(p_dl))<1e-10
    @test maximum(abs.(e_inc).-abs.(e_inc_dl))<1e-10
    #without input field
    #DDA solving-tensor
    mat=DDACore.solve_DDA_e(knorm,r,a,verbose=false)
    #DDA solving-tensor
    mat_dl=DDACore.solve_DDA_e(knorm*r,knorm^3/4/pi*a,verbose=false)
    #testing
    @test maximum(abs.(mat).-abs.(mat_dl))<1e-10
    println("passed")



    println("")
    println("***")
    println("electric and magnetic DDA")
    println("***")
    println("")

    #------Consevation of energy--------
    #ŦEST: test if energy is conserved by the DDA process by comparing the cross sections
    print("Testing conservation of energy: ")
    #lattice parameter
    d=1e-8
    rad=230e-9
    #position of one dipole
    latt=[0 0 0;0 0 d]
    #dielectric constant
    e=(3.5+im*0.01)
    #norm of the wave vector
    knorm=2*pi/1500e-9
    #generate polarisabilities
    n=length(latt[:,1])
    alpha_e=zeros(ComplexF64,n,3,3)
    alpha_m=zeros(ComplexF64,n,3,3)
    for j=1:n
        ae,am=Alphas.alpha_e_m_mie_renorm(knorm,rad,sqrt(e),1)
        alpha_e[j,:,:]=copy(ae*[1 0 0;0 1 0;0 0 1])
        alpha_m[j,:,:]=copy(am*[1 0 0;0 1 0;0 0 1])
    end
    #computing polarisabilitite

    #DDA solving--adimensional inputs
    p,m,e_inc,h_inc,e_inp,h_inp=DDACore.solve_DDA_e_m(knorm*latt,alpha_e,alpha_m,InputFields.plane_wave_e_m_renorm,verbose=false)
    #computing cross sections
    res=PostProcessing.compute_cross_sections_e_m(knorm,latt[:,1:3],p,m,e_inc,h_inc,e_inp,h_inp,alpha_e,alpha_m,verbose=false)
    #testing
    @test real(res[2])-real(res[3])-real(res[4])<10^(-10)
    println("passed")

    #------Alpha dynamic input--------
    #ŦEST: test if alpha dynamic input is function
    print("Testing dynamic input of alpha: ")
    #scalar polarisability
    ae_scalar=alpha_e[:,1,1]
    am_scalar=alpha_m[:,1,1]
    #DDA solving-tensor
    p,m,e_inc,h_inc,e_inp,h_inp=DDACore.solve_DDA_e_m(knorm*latt,alpha_e,alpha_m,InputFields.plane_wave_e_m_renorm,verbose=false)
    #DDA solving-tensor
    p_s,m_s,e_inc_s,h_inc_s,e_inp_s,h_inp_s=DDACore.solve_DDA_e_m(knorm*latt,ae_scalar,am_scalar,InputFields.plane_wave_e_m_renorm,verbose=false)
    #testing
    @test maximum(abs.(p).-abs.(p_s))<1e-10
    @test maximum(abs.(e_inc).-abs.(e_inc_s))<1e-10
    println("passed")

    #------Dimentionless with or without input field get the same--------
    #ŦEST: test if
    print("Testing that dimensional and adimensional version gives the same: ")
    #with input field
    #DDA solving-tensor
    p,m,e_inc,h_inc,e_inp,h_inp=DDACore.solve_DDA_e_m(knorm*latt,alpha_e,alpha_m,InputFields.plane_wave_e_m_renorm,verbose=false)
    #DDA solving-tensor
    p_dl,e_inc_dl=DDACore.solve_DDA_e(knorm*r,knorm^3/4/pi*a,InputFields.plane_wave_renorm,verbose=false)
    #testing
    @test maximum(abs.(knorm^3/4/pi*p).-abs.(p_dl))<1e-10
    println(knorm^3/4/pi*p)
    println(p_dl)
    @test maximum(abs.(e_inc).-abs.(e_inc_dl))<1e-10
    println(e_inc)
    println(e_inc_dl)
    #without input field
    #DDA solving-tensor
    mat=DDACore.solve_DDA_e(knorm,r,a,verbose=false)
    #DDA solving-tensor
    mat_dl=DDACore.solve_DDA_e(knorm*r,knorm^3/4/pi*a,verbose=false)
    #testing
    @test maximum(abs.(mat).-abs.(mat_dl))<1e-10
    println("passed")
end
#--------------------------------