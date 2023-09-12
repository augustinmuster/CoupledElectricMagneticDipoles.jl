#imports
using CoupledElectricMagneticDipoles
using PyCall
using LaTeXStrings
using Lebedev
using LinearAlgebra
@pyimport matplotlib.pyplot as plt


##################### Parameters ########################################
#radius (in nm)
a=250
#dielectric constant of the particle
eps=(1.59)^2
#dielectric constant of the medium
eps_h=(1.33)^2
#number of wavelengths to compute (in nm)
N_lambda=10
lambda_min=1000
lambda_max=1100
#wavelengths to compute
lambdas0=LinRange(lambda_min,lambda_max,N_lambda)
lambdas=lambdas0/sqrt(eps_h)
##########################################################################

#discretizes a sphere in small cubes
latt,dx=Geometries.discretize_sphere(a,10)

#getting number of cubes in the discretized sphere
n=length(latt[:,1])


#create an array to store results
res=zeros(Float64,N_lambda,3)


#solves DDA problem for each wavelength
for i=1:N_lambda
    #wavenumber in medium
    knorm=2*pi/lambdas[i]
    #computes polarizability for each dipoles using effective dielectric constant 
    alpha=zeros(ComplexF64,n,3,3)
    for j=1:n
        eps_eff=latt[j,4]*eps+(1-latt[j,4])*eps_h
        alpha[j,:,:]=Alphas.alpha_radiative(Alphas.alpha0_parallelepiped(dx,dx,dx,eps_eff,eps_h),knorm)
    end
    #computes input_field, an x-polarized plane-wave propagating along z
    input_field=InputFields.plane_wave_e(knorm*latt[:,1:3])
    #solves DDA
    e_inc=DDACore.solve_DDA_e(knorm*latt[:,1:3],alpha,input_field=input_field,solver="CPU")
    #computes cross section and save it in folder
    res[i,1:end]=PostProcessing.compute_cross_sections_e(knorm,knorm*latt[:,1:3],e_inc,alpha,input_field;explicit_scattering=true,verbose=true)
end


#scattering cross section from the Mie theory
res_mie=MieCoeff.mie_scattering.(2 .*pi./lambdas*a,eps,eps_h;cutoff=50)

#plotting the cross sections using matplotlib
fig1,ax1=plt.subplots(2,sharex=true)
#sets axis labels
ax1[1].set_ylabel(L"Q_{sca}")
ax1[2].set_ylabel(L"(Q_{ext}-Q_{abs}-Q_{abs})/Q_{ext}")
ax1[2].set_xlabel(L"\lambda_0/a")
#plot
cst=pi*a^2
ax1[1].plot(lambdas0./a,res[:,3]./cst,color="black",label="DDA, N="*string(n),marker="o")
ax1[1].plot(lambdas0./a,res_mie,color="red",label="Mie",marker="o")
ax1[2].plot(lambdas0./a,(res[:,1].-res[:,2].-res[:,3])./res[:,1],color="black",marker="o")
#legend and save
ax1[1].legend()
plt.tight_layout()
fig1.savefig("Q_sca.svg")



#computes polarizability for each dipoles using effective dielectric constant 
knorm=2*pi/lambdas[1]
alpha=zeros(ComplexF64,n,3,3)
for j=1:n
    eps_eff=latt[j,4]*eps+(1-latt[j,4])*eps_h
    global  alpha[j,:,:]=Alphas.alpha_radiative(Alphas.alpha0_parallelepiped(dx,dx,dx,eps_eff,eps_h),knorm)
end

#computes input_field, an x-polarized plane-wave propagating along z
input_field=InputFields.plane_wave_e(knorm*latt[:,1:3])
#solves DDA
e_inc=DDACore.solve_DDA_e(knorm*latt[:,1:3],alpha,input_field=input_field,solver="CPU")
#computes cross section
cs=PostProcessing.compute_cross_sections_e(knorm,knorm*latt[:,1:3],e_inc,alpha,input_field;explicit_scattering=true,verbose=true)


#sampling direction an plotting 
thetas=LinRange(0,2*pi,100)
ur=zeros(100,3)
ur[:,3]=cos.(thetas)
ur[:,2]=sin.(thetas)

#computes differential cross section
res=PostProcessing.diff_scattering_cross_section_e(knorm,knorm*latt[:,1:3],e_inc,alpha,input_field,ur)

#plotting
fig2=plt.figure()
ax2 = fig2.add_subplot(projection="polar")
ax2.set_title(L"log(d Q_{sca}/ d \Omega)")
ax2.plot(thetas,log10.(res/pi/(a^2)),label="y-z plane")
plt.tight_layout()
fig2.savefig("diff_Q_sca.svg")

#Compare total scattering cross section and integral of the differential one
x,y,z,w = lebedev_by_order(13)
csca_int=4 * pi * dot(w,PostProcessing.diff_scattering_cross_section_e(knorm,knorm*latt[:,1:3],e_inc,alpha,input_field,[x y z]))
println(cs[3]," : ",csca_int)
