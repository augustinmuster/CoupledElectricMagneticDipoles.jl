#importing the julia library in python
from julia.api import Julia
jl=Julia(compiled_modules=False)
from julia import CoupledElectricMagneticDipoles as CEMD
#importing matplotlib and numpy
import numpy as np
import matplotlib.pyplot as plt

##################### Parameters ########################################
#radius (in nm)
a=250
#dielectric constant of the particle
eps=1.59**2
#dielectric constant of the medium
eps_h=1.33**2
#number of wavelengths to compute (in nm)
N_lambda=10
lambda_min=1000
lambda_max=1100
#wavelengths to compute
lambdas0=np.linspace(lambda_min,lambda_max,N_lambda)
lambdas=lambdas0/np.sqrt(eps_h)
##########################################################################

#discretizes a sphere in small cubes
latt,dx=CEMD.Geometries.discretize_sphere(a,10)
#converting latt to numpy array
latt=np.array(latt)

#getting number of cubes in the discretized sphere
n=len(latt)

#create an array to store results
res=np.zeros((N_lambda,3))

#solves DDA problem for each wavelength
for i in range(N_lambda):
    #wavenumber in medium
    knorm=2*np.pi/lambdas[i]
    #computes polarizability for each dipoles using effective dielectric constant 
    alpha=np.zeros((n,3,3),complex)
    for j in range(n):
        eps_eff=latt[j,3]*eps+(1-latt[j,3])*eps_h
        alpha[j,:,:]=np.array(CEMD.Alphas.alpha_radiative(CEMD.Alphas.alpha0_parallelepiped(dx,dx,dx,eps_eff,eps_h),knorm))

    #computes input_field, an x-polarized plane-wave propagating along z
    input_field=np.array(CEMD.InputFields.plane_wave_e(knorm*latt[:,0:3]))
    #solves DDA
    e_inc=np.array(CEMD.DDACore.solve_DDA_e(knorm*latt[:,0:3],alpha,input_field=input_field,solver="CPU"))
    #computes cross section and save it in folder
    res[i,:]=CEMD.PostProcessing.compute_cross_sections_e(knorm,knorm*latt[:,0:3],e_inc,alpha,input_field,explicit_scattering=True,verbose=True)




#plotting the cross sections using matplotlib
fig1,ax1=plt.subplots(2,sharex=True)
#sets axis labels
ax1[0].set_ylabel("$Q_{sca}$")
ax1[1].set_ylabel("$(Q_{ext}-Q_{abs}-Q_{abs})/Q_{ext}$")
ax1[1].set_xlabel("$\lambda_0/a$")
#plot
cst=np.pi*a**2
ax1[0].plot(lambdas0/a,res[:,2]/cst,color="black",label="DDA, N="+str(n),marker="o")
ax1[1].plot(lambdas0/a,(res[:,0]-res[:,1]-res[:,2])/res[:,0],color="black",marker="o")
#legend and save
ax1[0].legend()
plt.tight_layout()
fig1.savefig("Q_sca_py.svg")

