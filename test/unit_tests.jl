#imports
using Test
using Base
using LinearAlgebra

#include the modules
println("Importing the library...")
include("../DDA.jl")
include("../alpha.jl")
include("../input_fields.jl")
include("../processing.jl")

#------Consevation of energy--------
#ŦEST: test if energy is conserved by the DDA process by comparing the cross sections
print("Testing conservation of energy: ")
#lattice parameter
d=1e-8
#position of one dipole
r=[0 0 0;0 0 d]
#dielectric constant
eps=(3.5+im*0.01)^2
#wavelength
lambda=1000e-9
knorm=2*pi/lambda
#computing polarisabilitites
L=depolarisation_tensor(d,d,d,d^3)

a0=zeros(ComplexF64,length(r[:,1]),3,3)
a=zeros(ComplexF64,length(r[:,1]),3,3)
for i=1:length(r[:,1])
    a0[i,:,:]=alpha_0(eps,1,L,d^3)
    a[i,:,:]=alpha_radiative(a0[i,:,:],knorm)
end
#DDA solving
p,e_inc=solve_DDA(knorm,r,a,plane_wave,verbose=false)
#computing cross sections
res=compute_cross_sections(knorm,p,e_inc,a0,r,verbose=false)
#testing
@test real(res[2])-real(res[3])-real(res[4])<10^(-10)
println("passed")
#--------------------------------