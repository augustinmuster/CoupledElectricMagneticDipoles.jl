#imports
using CoupledElectricMagneticDipoles
using PyCall
using LaTeXStrings
using LinearAlgebra
@pyimport matplotlib.pyplot as plt


##################### Parameters ########################################
#radius of the sphere (in μm)
a_refl=0.245 #reflector radius
a_dir=0.200 #director radius
N_dir=10 #number of directors
#dielectric constant of the particle
eps=12
#wavelength (in μm) 
lambda=1.550
##########################################################################

#setting the structure
#creates an array to contain the positions
r=zeros(N_dir+1,3)

#spacing between reflector and first director
spacing_ref_dir=a_refl+0.355+0.800+a_dir
#spacing between directors
spacing_dirs=4*a_dir

#sets the position of the directors (reflector is at the origin)
for i=2:N_dir+1
    r[i,3]=spacing_ref_dir+(i-2)*spacing_dirs
end

#creates an array containing the radius of each sphere
as=a_dir*ones(N_dir+1)
as[1]=a_refl



#------------------modelling silicon particles---------------
#parameter
lambdas=LinRange(1.200,1.600,100)
knorms=2*pi./lambdas
a=0.230
ka = knorms*a
eps=12
#scattering cross sections
mie_sca=MieCoeff.mie_scattering.(ka,eps,1,cutoff=20)
dipole_sca=(6*pi)./knorms.^2 .*(abs2.(MieCoeff.mie_an.(ka, eps, 1, n=1)).+abs2.(MieCoeff.mie_bn.(ka, eps, 1, n=1)))
#plotting
fig1,ax1=plt.subplots()
ax1.set_xlabel(L"\lambda\ (\mu m)")
ax1.set_ylabel(L"Q_{sca}")
ax1.plot(lambdas,mie_sca,color="black",label="Mie")
ax1.plot(lambdas,dipole_sca./(pi*a^2),color="red",label="Dipoles")
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
#computes the input input_field
krd=knorm*[0,0,0.355+0.245] #position of the emitter
input_field=InputFields.point_dipole_e_m(knorm*r,krd,2)

#solves DDA electric and magnetic
phi_inc=DDACore.solve_DDA_e_m(knorm*r,alpha_e,alpha_m,input_field=input_field,solver="CPU")

#sample directions in the y-z plane
thetas=LinRange(0,2*pi,200)
ur=zeros(200,3)
ur[:,3]=knorm*cos.(thetas)
ur[:,2]=knorm*sin.(thetas)

#emission pattern of the antenna
res=PostProcessing.emission_pattern_e_m(knorm*r,phi_inc,alpha_e,alpha_m,ur,krd,2)

#plotting
fig2=plt.figure()
ax2 = fig2.add_subplot(projection="polar")
ax2.set_title(L"d P/ d \Omega\ (P_0)")
ax2.plot(thetas,res,label="y-z plane")
ax2.legend()
fig2.savefig("diff_P.svg")



