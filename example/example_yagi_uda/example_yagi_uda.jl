#imports
using PyCall
using LaTeXStrings
using Lebedev
using LinearAlgebra
@pyimport matplotlib.pyplot as plt
@pyimport numpy as np
include("../../src/DDA.jl")
include("../../src/geometries.jl")
include("../../src/processing.jl")
include("../../src/alpha.jl")
include("../../src/input_fields.jl")
include("../../src/mie_coeff.jl")


##################### Parameters ########################################
#radius of the sphere
a_refl=245e-9
a_dir=200e-9
N_dir=20
#dielectric constant of the particle
eps=12
#number of wavelengths to compute
lambda=1550e-9
##########################################################################

#setting the structure
#creates an array to contain the positions
r=zeros(N_dir+1,3)

#spacing between reflector and first director
spacing_ref_dir=a_refl+355e-9+800e-9+a_dir
#spacing between directors
spacing_dirs=4*a_dir

#sets the position of the directors (reflector is at the origin)
for i=2:N_dir+1
    r[i,3]=spacing_ref_dir+(i-2)*spacing_dirs
    println(i,": ",r[i,3])
end

println(r)
#creates an array containing the radius of each sphere
as=a_dir*ones(N_dir+1)
as[1]=a_refl



#------------------modelling silicon particles---------------
#parameter
lambdas=LinRange(1200e-9,1600e-9,100)
knorms=2*pi./lambdas
a=230e-9
ka = knorms*a
eps=12
#scattering cross sections
mie_sca=MieCoeff.mie_scattering.(ka,eps,1,cutoff=20)
dipole_sca=(6*pi)./knorms.^2 .*(abs2.(MieCoeff.mie_an.(ka, eps, 1, n=1)).+abs2.(MieCoeff.mie_bn.(ka, eps, 1, n=1)))
#plotting
fig1,ax1=plt.subplots()
ax1.set_xlabel(L"\lambda\ (nm)")
ax1.set_ylabel(L"Q_{sca}")
ax1.plot(lambdas.*1e9,mie_sca,color="black",label="Mie")
ax1.plot(lambdas.*1e9,dipole_sca./(pi*a^2),color="red",label="Dipoles")
fig1.savefig("mie_dipole_qsca.svg")
#------------------------------------------------------------


#computes the wavenumber
knorm=2*pi/lambda
#computes the polarizabilities using first mie coefficients
alpha_e=zeros(ComplexF64,N_dir+1)
alpha_m=zeros(ComplexF64,N_dir+1)
for i=1:N_dir+1 
    alpha_e[i],alpha_m[i]=Alphas.alpha_e_m_mie(knorm*as[i],eps,1)
end
println(alpha_m)
#computes the input input_field
input_field=InputFields.point_dipole_e_m(knorm*r,knorm*[0,0,355e-9],2)

println(input_field)
println(size(input_field))
#solves DDA electric and magnetic
phi_inc=DDACore.solve_DDA_e_m(knorm*r,alpha_e,alpha_m,input_field=input_field,solver="CPU")

println(phi_inc)

thetas=LinRange(0,2*pi,200)
krf=zeros(200,3)
krf[:,3]=100e-5*knorm*cos.(thetas)
krf[:,2]=100e-5*knorm*sin.(thetas)

phi_krf=InputFields.point_dipole_e_m(krf,knorm*[0,0,355e-9],2)

res=PostProcessing.diff_emitted_power_e_m(knorm,knorm*r,phi_inc,alpha_e,alpha_m,krf,phi_krf)

println(res)

#plotting
fig2=plt.figure()
ax2 = fig2.add_subplot(projection="polar")
ax2.set_title(L"d P/ d \Omega")
ax2.plot(thetas,res,label="y-z plane")
ax2.legend()
fig2.savefig("diff_P.svg")



