using Base
using DelimitedFiles
#using LinearAlgebra
include("../../src/CoupledElectricMagneticDipoles.jl")
using .CoupledElectricMagneticDipoles

using PyCall
#@pyimport numpy
@pyimport matplotlib.pyplot as plt

# analytic solution R. Carminati et. al., Opt. Comm. 261, 368 (2006)
function ldos_analytic(kz, alp_dl)
    ldos_z = 1 + 6*imag(alp_dl*exp(2*im*kz)*(-1/kz^4 - 2*im/kz^5 + 1/kz^6) )
    ldos_x = 1 + 3/2*imag(alp_dl*exp(2*im*kz)*(1/kz^2 + 2*im/kz^3 - 3/kz^4 - 2*im/kz^5 + 1/kz^6) )
    return ldos_z, ldos_x
end

# physical properties
# particle radius
a = 5
# wavelengths
lamb = [612, 354] 
# dielectric constant of the particle
eps=[-15.04 + im*1.02, -2.03 + im*0.6] 
# distante between particle and dipole 
nz = 91 
z = LinRange(10,100,nz)

# variables to store the calcualtions
ldos_z = zeros(nz,2) 
ldos_x = zeros(nz,2)
ldos_z_analytic = zeros(nz,2) 
ldos_x_analytic = zeros(nz,2)

# ldos calculation at both wavelengths and all distances
for i=1:2 # loop in wavelength
    # wavevector
    knorm = 2*pi/lamb[i] 
    # permittivity
    eps_i = eps[i] 
    # calculation of the polarizability
    alp_0 = Alphas.alpha0_sphere(a,eps_i,1) # static polarizability
    alp_e_dl = Alphas.alpha_radiative(alp_0,knorm) # dimensionless polarizability with radiative corrections
    for j=1:nz # loop in distance
        # distance
        z_j = z[j] 
        # renormalized distance
        kz = knorm*z_j
        # renormalized position of the particle (at the origin of coordinates) 
        kr = zeros(1,3) 
        # renormalized position of the diple (z-component at kz)
        krd = zeros(1,3) 
        krd[3] = kz
        # analytic ldos
        global ldos_z_analytic[j,i], ldos_x_analytic[j,i] =  ldos_analytic(kz, alp_e_dl) 
        # numerical ldos
        Ainv = DDACore.solve_DDA_e(kr,alp_e_dl) # calculation inverse dda matrix
        global ldos_z[j,i] = PostProcessing.ldos_e(kr, alp_e_dl, Ainv, krd, dip = 3) # ldos z-axis
        global ldos_x[j,i] = PostProcessing.ldos_e(kr, alp_e_dl, Ainv, krd, dip = 1) # ldos x-axis
    end
end

# plot ldos
for ind_l = 1:2
    fig,axs=plt.subplots()
    fig.suptitle("LDOSx lambda = "*string(Int(lamb[ind_l]))*" nm")
    axs.plot(z,ldos_x_analytic[:,ind_l],"--",label="LDOSx analytic")
    axs.plot(z,ldos_x[:,ind_l],"o",label="LDOSx DDA")
    axs.set_xlabel("z (nm)")
    axs.set_ylabel("LDOS_x")
    axs.set_yscale("log")
    fig.tight_layout()
    axs.legend()
    plt.savefig("LDOSx"*string(Int(lamb[ind_l]))*".svg")

    fig,axs=plt.subplots()
    fig.suptitle("LDOSz lambda = "*string(Int(lamb[ind_l]))*" nm")
    axs.plot(z,ldos_z_analytic[:,ind_l],"--",label="LDOSz analytic")
    axs.plot(z,ldos_z[:,ind_l],"o",label="LDOSz DDA")
    axs.set_xlabel("z (nm)")
    axs.set_ylabel("LDOS_z")
    axs.set_yscale("log")
    fig.tight_layout()
    axs.legend()
    plt.savefig("LDOSz"*string(Int(lamb[ind_l]))*".svg")
end