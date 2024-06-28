#imports
using CoupledElectricMagneticDipoles
using PyCall
using LaTeXStrings
using Lebedev
using LinearAlgebra
@pyimport matplotlib.pyplot as plt


##################### Parameters ########################################
#length of edge of the cube (in nm)
D=250
#dielectric constant of the particle
eps=(1.6+0.01*im)^2
#dielectric constant of the medium
eps_h=1
#wavelengths to compute (in nm)
lambda0=2*pi*D./0.1
lambda=lambda0/sqrt(eps_h)
##########################################################################

#discretizes a sphere in small cubes
latt,dx=Geometries.discretize_cube(D,10)
#getting number of cubes in the discretized sphere
n=length(latt[:,1])


#create an array to store results
res=zeros(Float64,3)

#wavenumber in medium
knorm=2*pi/lambda
#computes polarizability for each dipoles using effective dielectric constant 
alpha=zeros(ComplexF64,n)
for j=1:n
    eps_eff=latt[j,4]*eps+(1-latt[j,4])*eps_h
    alpha[j]=Alphas.alpha_radiative(Alphas.alpha0_cube(dx,eps_eff,eps_h),knorm)
end
#computes input_field, an x-polarized plane-wave propagating along z
input_field=InputFields.plane_wave_e(knorm*latt[:,1:3])
#solves DDA
e_inc=DDACore.solve_DDA_e(knorm*latt[:,1:3],alpha,input_field=input_field,solver="CPU")
#computes cross section and save it in folder
res=PostProcessing.compute_cross_sections_e(knorm,knorm*latt[:,1:3],e_inc,alpha,input_field;explicit_scattering=true,verbose=true)

#radius of volume equivalent sphere
a=(D^3*3/4/pi)^(1/3)
#print ext, abs. and sca. efficiencies
println(res./pi/a^2)


