include("../src/CoupledElectricMagneticDipoles.jl")
using .CoupledElectricMagneticDipoles
using Test
using LinearAlgebra
using Lebedev
#FLAGS
#in order to skip a test group, set the appropriate falg to false
TEST_GREENTENSORS=true
TEST_DDACORE=true
TEST_CS=true
TEST_LDOS=true
#parameters for the test
N=10
epsilon=1e-8

#useful constants
const id3=[1 0 0;0 1 0;0 0 1]

#testin green tensors
if TEST_GREENTENSORS

    ##########################
    print("Testing x derivative of the magnetic green tensor: ")
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[x,2,3]
    x1bis=[x+h,2,3]
    #
    x2=[2,3,4]
    #numerical one
    num=(GreenTensors.G_m_renorm(x1bis,x2)-GreenTensors.G_m_renorm(x1,x2))/(h)
    #numerical highest order
    num2=-GreenTensors.G_m_renorm([x+2*h,2,3],x2)+8*GreenTensors.G_m_renorm([x+h,2,3],x2)-8*GreenTensors.G_m_renorm([x-h,2,3],x2)+GreenTensors.G_m_renorm([x-2*h,2,3],x2)
    num2=num2/12/h
    #analytical one
    func=GreenTensors.dxG_m_renorm(x1,x2)
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<epsilon
    @test real((num2-func)[1,3])/real(num2[1,3])<epsilon
    @test real((num2-func)[2,1])/real(num2[2,1])<epsilon
    @test real((num2-func)[2,3])/real(num2[2,3])<epsilon
    @test real((num2-func)[3,1])/real(num2[3,1])<epsilon
    @test real((num2-func)[3,2])/real(num2[3,2])<epsilon
    @test imag((num2-func)[1,2])/imag(num2[1,2])<epsilon
    @test imag((num2-func)[1,3])/imag(num2[1,3])<epsilon
    @test imag((num2-func)[2,1])/imag(num2[2,1])<epsilon
    @test imag((num2-func)[2,3])/imag(num2[2,3])<epsilon
    @test imag((num2-func)[3,1])/imag(num2[3,1])<epsilon
    @test imag((num2-func)[3,2])/imag(num2[3,2])<epsilon
    println("passed")
    ##############################

    ##########################
    print("Testing y derivative of the magnetic green tensor: ")
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[1,x,3]
    x1bis=[1,x+h,3]
    #
    x2=[7,8,9]

    #numerical one
    num=(GreenTensors.G_m_renorm(x1bis,x2)-GreenTensors.G_m_renorm(x1,x2))/(h)
    #numerical highest order
    num2=-GreenTensors.G_m_renorm([1,x+2*h,3],x2)+8*GreenTensors.G_m_renorm([1,x+h,3],x2)-8*GreenTensors.G_m_renorm([1,x-h,3],x2)+GreenTensors.G_m_renorm([1,x-2*h,3],x2)
    num2=num2/12/h
    #analytical one
    func=GreenTensors.dyG_m_renorm(x1,x2)
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<epsilon
    @test real((num2-func)[1,3])/real(num2[1,3])<epsilon
    @test real((num2-func)[2,1])/real(num2[2,1])<epsilon
    @test real((num2-func)[2,3])/real(num2[2,3])<epsilon
    @test real((num2-func)[3,1])/real(num2[3,1])<epsilon
    @test real((num2-func)[3,2])/real(num2[3,2])<epsilon
    @test imag((num2-func)[1,2])/imag(num2[1,2])<epsilon
    @test imag((num2-func)[1,3])/imag(num2[1,3])<epsilon
    @test imag((num2-func)[2,1])/imag(num2[2,1])<epsilon
    @test imag((num2-func)[2,3])/imag(num2[2,3])<epsilon
    @test imag((num2-func)[3,1])/imag(num2[3,1])<epsilon
    @test imag((num2-func)[3,2])/imag(num2[3,2])<epsilon
    println("passed")
    ##########################

    ##########################
    print("Testing z derivative of the magnetic green tensor: ")
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[1,2,x]
    x1bis=[1,2,x+h]
    #
    x2=[7,8,9]

    #numerical one
    num=(GreenTensors.G_m_renorm(x1bis,x2)-GreenTensors.G_m_renorm(x1,x2))/(h)
    #numerical highest order
    num2=-GreenTensors.G_m_renorm([1,2,x+2*h],x2)+8*GreenTensors.G_m_renorm([1,2,x+h],x2)-8*GreenTensors.G_m_renorm([1,2,x-h],x2)+GreenTensors.G_m_renorm([1,2,x-2*h],x2)
    num2=num2/12/h
    #analytical one
    func=GreenTensors.dzG_m_renorm(x1,x2)
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<epsilon
    @test real((num2-func)[1,3])/real(num2[1,3])<epsilon
    @test real((num2-func)[2,1])/real(num2[2,1])<epsilon
    @test real((num2-func)[2,3])/real(num2[2,3])<epsilon
    @test real((num2-func)[3,1])/real(num2[3,1])<epsilon
    @test real((num2-func)[3,2])/real(num2[3,2])<epsilon
    @test imag((num2-func)[1,2])/imag(num2[1,2])<epsilon
    @test imag((num2-func)[1,3])/imag(num2[1,3])<epsilon
    @test imag((num2-func)[2,1])/imag(num2[2,1])<epsilon
    @test imag((num2-func)[2,3])/imag(num2[2,3])<epsilon
    @test imag((num2-func)[3,1])/imag(num2[3,1])<epsilon
    @test imag((num2-func)[3,2])/imag(num2[3,2])<epsilon
    println("passed")
    ##########################

    ##########################
    print("Testing x derivative of the electric green tensor: ")
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[x,2,3]
    x1bis=[x+h,2,3]
    #
    x2=[2,3,4]
    knorm=1e9
    #numerical one
    num=(GreenTensors.G_e_renorm(x1bis,x2)-GreenTensors.G_e_renorm(x1,x2))/(h)
    #numerical highest order
    num2=-GreenTensors.G_e_renorm([x+2*h,2,3],x2)+8*GreenTensors.G_e_renorm([x+h,2,3],x2)-8*GreenTensors.G_e_renorm([x-h,2,3],x2)+GreenTensors.G_e_renorm([x-2*h,2,3],x2)
    num2=num2/12/h
    #analytical one
    func=GreenTensors.dxG_e_renorm(x1,x2)
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<epsilon
    @test real((num2-func)[1,3])/real(num2[1,3])<epsilon
    @test real((num2-func)[2,1])/real(num2[2,1])<epsilon
    @test real((num2-func)[2,3])/real(num2[2,3])<epsilon
    @test real((num2-func)[3,1])/real(num2[3,1])<epsilon
    @test real((num2-func)[3,2])/real(num2[3,2])<epsilon
    @test imag((num2-func)[1,2])/imag(num2[1,2])<epsilon
    @test imag((num2-func)[1,3])/imag(num2[1,3])<epsilon
    @test imag((num2-func)[2,1])/imag(num2[2,1])<epsilon
    @test imag((num2-func)[2,3])/imag(num2[2,3])<epsilon
    @test imag((num2-func)[3,1])/imag(num2[3,1])<epsilon
    @test imag((num2-func)[3,2])/imag(num2[3,2])<epsilon
    println("passed")
    ##########################

    ##########################
    print("Testing x derivative of the electric green tensor: ")
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[1,x,3]
    x1bis=[1,x+h,3]
    #
    x2=[7,8,9]
    knorm=1e9
    #numerical one
    num=(GreenTensors.G_e_renorm(x1bis,x2)-GreenTensors.G_e_renorm(x1,x2))/(h)
    #numerical highest order
    num2=-GreenTensors.G_e_renorm([1,x+2*h,3],x2)+8*GreenTensors.G_e_renorm([1,x+h,3],x2)-8*GreenTensors.G_e_renorm([1,x-h,3],x2)+GreenTensors.G_e_renorm([1,x-2*h,3],x2)
    num2=num2/12/h
    #analytical one
    func=GreenTensors.dyG_e_renorm(x1,x2)
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<epsilon
    @test real((num2-func)[1,3])/real(num2[1,3])<epsilon
    @test real((num2-func)[2,1])/real(num2[2,1])<epsilon
    @test real((num2-func)[2,3])/real(num2[2,3])<epsilon
    @test real((num2-func)[3,1])/real(num2[3,1])<epsilon
    @test real((num2-func)[3,2])/real(num2[3,2])<epsilon
    @test imag((num2-func)[1,2])/imag(num2[1,2])<epsilon
    @test imag((num2-func)[1,3])/imag(num2[1,3])<epsilon
    @test imag((num2-func)[2,1])/imag(num2[2,1])<epsilon
    @test imag((num2-func)[2,3])/imag(num2[2,3])<epsilon
    @test imag((num2-func)[3,1])/imag(num2[3,1])<epsilon
    @test imag((num2-func)[3,2])/imag(num2[3,2])<epsilon
    println("passed")
    ##########################

    ##########################
    print("Testing x derivative of the electric green tensor: ")
    #small step
    h=0.000001
    #value of the component
    x=4
    #
    x1=[1,2,x]
    x1bis=[1,2,x+h]
    #
    x2=[7,8,9]
    knorm=1e9
    #numerical one
    num=(GreenTensors.G_e_renorm(x1bis,x2)-GreenTensors.G_e_renorm(x1,x2))/(h)
    #numerical highest order
    num2=-GreenTensors.G_e_renorm([1,2,x+2*h],x2)+8*GreenTensors.G_e_renorm([1,2,x+h],x2)-8*GreenTensors.G_e_renorm([1,2,x-h],x2)+GreenTensors.G_e_renorm([1,2,x-2*h],x2)
    num2=num2/12/h
    #analytical one
    func=GreenTensors.dzG_e_renorm(x1,x2)
    #unit test
    @test real((num2-func)[1,2])/real(num2[1,2])<epsilon
    @test real((num2-func)[1,3])/real(num2[1,3])<epsilon
    @test real((num2-func)[2,1])/real(num2[2,1])<epsilon
    @test real((num2-func)[2,3])/real(num2[2,3])<epsilon
    @test real((num2-func)[3,1])/real(num2[3,1])<epsilon
    @test real((num2-func)[3,2])/real(num2[3,2])<epsilon
    @test imag((num2-func)[1,2])/imag(num2[1,2])<epsilon
    @test imag((num2-func)[1,3])/imag(num2[1,3])<epsilon
    @test imag((num2-func)[2,1])/imag(num2[2,1])<epsilon
    @test imag((num2-func)[2,3])/imag(num2[2,3])<epsilon
    @test imag((num2-func)[3,1])/imag(num2[3,1])<epsilon
    @test imag((num2-func)[3,2])/imag(num2[3,2])<epsilon
    println("passed")
    ##########################
    println("")
end

#test DDACORE
if TEST_DDACORE
    ##########################
    print("Testing DDA solving e: ")
    #params
    kr=rand(N,3)
    alpha_e=rand(ComplexF64,N)
    input_field=rand(ComplexF64,N,3)
    #solving
    Ainv=DDACore.solve_DDA_e(kr,alpha_e,verbose=false)
    e_inc=DDACore.solve_DDA_e(kr,alpha_e,input_field=input_field,verbose=false)
    #test
    @test sum(abs.(transpose(reshape(Ainv*reshape(transpose(input_field),3*N),3,N)).-e_inc).<epsilon)==3*N
    println("passed")
    ##############################

    ##########################
    print("Testing DDA solving em: ")
    #params
    kr=rand(N,3)
    alpha_e=rand(ComplexF64,N)
    alpha_m=rand(ComplexF64,N)
    input_field=rand(ComplexF64,N,6)
    #solving
    Ainv=DDACore.solve_DDA_e_m(kr,alpha_e,alpha_m,verbose=false)
    e_inc=DDACore.solve_DDA_e_m(kr,alpha_e,alpha_m,input_field=input_field,verbose=false)
    #test1
    @test sum(abs.(transpose(reshape(Ainv*reshape(transpose(input_field),6*N),6,N)).-e_inc).<epsilon)==6*N
    #6x6 pol input
    alpha=zeros(ComplexF64,N,6,6)
    for i=1:N
        alpha[i,1:3,1:3]=alpha_e[i]*id3
        alpha[i,4:6,4:6]=alpha_m[i]*id3
    end
    Ainv2=DDACore.solve_DDA_e_m(kr,alpha,verbose=false)
    #test2
    @test norm(Ainv-Ainv2)<epsilon
    println("passed")
    ##############################
    println("")
end

#testing cross sections
if TEST_CS
    ##########################
    print("Testing CS e: ")
    #params
    knorm=rand()
    kr=rand(N,3)
    alpha_e=rand(ComplexF64,N)
    input_field=InputFields.plane_wave_e(kr)
    #solve DDA
    e_inc=DDACore.solve_DDA_e(kr,alpha_e,input_field=input_field,verbose=false)
    #compute cross section
    res=PostProcessing.compute_cross_sections_e(knorm,kr,e_inc,alpha_e,input_field;explicit_scattering=true,verbose=false)
    #test optical theorem
    @test (res[1]-res[2]-res[3])<epsilon
    println("passed")
    ###########################

    ##########################
    print("Testing diff CS e: ")
    x,y,z,w = lebedev_by_order(13)
    csca_int=4 * pi * dot(w,PostProcessing.diff_scattering_cross_section_e(knorm,kr,e_inc,alpha_e,input_field,[x y z],verbose=false))
    @test abs(res[3]-csca_int)<epsilon
    println("passed")
    ###########################

    ##########################
    print("Testing CS em: ")
    #params
    kr=rand(N,3)
    alpha_e=rand(ComplexF64,N)
    alpha_m=rand(ComplexF64,N)
    input_field=InputFields.plane_wave_e_m(kr)
    #solve DDA
    phi_inc=DDACore.solve_DDA_e_m(kr,alpha_e,alpha_m,input_field=input_field,verbose=false)
    #compute cross section
    res=PostProcessing.compute_cross_sections_e_m(knorm,kr,phi_inc,alpha_e,alpha_m,input_field;explicit_scattering=true,verbose=false)
    #test optical theorem
    @test (res[1]-res[2]-res[3])<epsilon
    println("passed")
    ###########################

    ##########################
    print("Testing diff CS em: ")
    x,y,z,w = lebedev_by_order(13)
    csca_int=4 * pi * dot(w,PostProcessing.diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_e,alpha_m,input_field,[x y z],verbose=false))
    @test abs(res[3]-csca_int)<epsilon
    println("passed")
    ###########################
    println("")
end 


#testing LDOS
if TEST_LDOS

    ##########################
    print("Testing LDOS e: ")
    #system parameters
    kr=rand(N,3)
    alpha_e=rand(ComplexF64,N)
    #compute LDOS
    A_inv=DDACore.solve_DDA_e(kr,alpha_e,verbose=false)
    krd=rand(1,3)
    LDOS=PostProcessing.ldos_e(kr, alpha_e, A_inv, krd; dip=1,verbose=false)
    #compute rad and nonrad LDOS
    inp=InputFields.point_dipole_e(kr,krd[1,:],1)
    e_inc=DDACore.solve_DDA_e(kr,alpha_e,input_field=inp,verbose=false)
    p=PostProcessing.compute_dipole_moment(alpha_e,e_inc[:,1:3])
    radLDOS=PostProcessing.rad_ldos_e(kr,krd,p,[1,0,0],verbose=false)
    nonradLDOS=PostProcessing.nonrad_ldos_e(p,e_inc,[1,0,0],verbose=false)
    #energy balance test
    @test (LDOS-radLDOS-nonradLDOS)<epsilon
    println("passed")
    ##############################

    ##########################
    print("Testing LDOS em: ")
    #system parameters
    kr=rand(N,3)
    alpha_e=rand(ComplexF64,N)
    alpha_m=rand(ComplexF64,N)
    #compute LDOS
    A_inv=DDACore.solve_DDA_e_m(kr,alpha_e,alpha_m,verbose=false)
    krd=rand(1,3)
    LDOS=PostProcessing.ldos_e_m(kr, alpha_e, alpha_m, A_inv, krd; dip=1,verbose=false)
    #compute rad and nonrad LDOS
    inp=InputFields.point_dipole_e_m(kr,krd[1,:],1)
    phi_inc=DDACore.solve_DDA_e_m(kr,alpha_e,alpha_m,input_field=inp,verbose=false)
    p=PostProcessing.compute_dipole_moment(alpha_e,phi_inc[:,1:3])
    m=PostProcessing.compute_dipole_moment(alpha_m,phi_inc[:,4:6])
    radLDOS=PostProcessing.rad_ldos_e_m(kr,krd,p,m,[1,0,0,0,0,0],verbose=false)
    nonradLDOS=PostProcessing.nonrad_ldos_e_m(p,m,phi_inc,[1,0,0,0,0,0],verbose=false)
    #energy balance test
    @test (LDOS-radLDOS-nonradLDOS)<epsilon
    println("passed")
    ##############################
    println("")
end
