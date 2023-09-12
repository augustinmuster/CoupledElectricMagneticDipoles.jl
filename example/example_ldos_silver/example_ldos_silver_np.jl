using Base
using DelimitedFiles
#using LinearAlgebra
include("../../src/CoupledElectricMagneticDipoles.jl")
using .CoupledElectricMagneticDipoles

using PyCall
#@pyimport numpy
@pyimport matplotlib.pyplot as plt

# analytic solutions R. Carminati et. al., Opt. Comm. 261, 368 (2006)
function ldos_analytic(kz, alp_dl)
    ldos_z = 1 + 6*imag(alp_dl*exp(2*im*kz)*(-1/kz^4 - 2*im/kz^5 + 1/kz^6) )
    ldos_x = 1 + 3/2*imag(alp_dl*exp(2*im*kz)*(1/kz^2 + 2*im/kz^3 - 3/kz^4 - 2*im/kz^5 + 1/kz^6) )
    return ldos_z, ldos_x
end
function nonrad_ldos_analytic_shortdistance(kz, alp_dl)
    ldos_z = 6*(imag(alp_dl) - 2/3*abs2(alp_dl))*(1/kz^4 + 1/kz^6) 
    ldos_x = 3/2*(imag(alp_dl) - 2/3*abs2(alp_dl))*(1/kz^2 - 1/kz^4 + 1/kz^6) 
    return ldos_z, ldos_x
end
function rad_ldos_analytic_shortdistance(kz, alp_dl)
    ldos_z = 1 + 4*abs2(alp_dl)*(1/kz^4 + 1/kz^6) + 4*real(alp_dl)/kz^3
    ldos_x = 1 + abs2(alp_dl)*(-1/kz^4 + 1/kz^6) - 2*real(alp_dl)/kz^3
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
rad_ldos_z = zeros(nz,2) 
rad_ldos_x = zeros(nz,2)
rad_ldos_z_analytic_shortdistance = zeros(nz,2) 
rad_ldos_x_analytic_shortdistance = zeros(nz,2)
nonrad_ldos_z = zeros(nz,2) 
nonrad_ldos_x = zeros(nz,2)
nonrad_ldos_z_analytic_shortdistance = zeros(nz,2) 
nonrad_ldos_x_analytic_shortdistance = zeros(nz,2)

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

        # analitic ldos at shortdistances for radiative and non radiative components
        global rad_ldos_z_analytic_shortdistance[j,i], rad_ldos_x_analytic_shortdistance[j,i] =  rad_ldos_analytic_shortdistance(kz, alp_e_dl)
        global nonrad_ldos_z_analytic_shortdistance[j,i], nonrad_ldos_x_analytic_shortdistance[j,i] =  nonrad_ldos_analytic_shortdistance(kz, alp_e_dl)

        # numerical ldos at shortdistances for radiative and non radiative components
        dip_z = zeros(3)
        dip_z[3] = 1
        dipole_field = InputFields.point_dipole_e(kr, krd[:], dip_z) # field of the point dipole with dipole moment "dip_z"
        phi_inc = DDACore.solve_DDA_e(kr,alp_e_dl;input_field=dipole_field)     # incident field at the silver partice 
        dipole_moment = PostProcessing.compute_dipole_moment(alp_e_dl,phi_inc) # dipole moment at the silver particle
        global rad_ldos_z[j,i] = PostProcessing.rad_ldos_e(kr,krd,dipole_moment,dip_z)          # radiative ldos z-axis
        global nonrad_ldos_z[j,i] = PostProcessing.nonrad_ldos_e(dipole_moment,phi_inc,dip_z)   # non-radiative ldos z-axis

        dip_x = zeros(3)
        dip_x[1] = 1
        dipole_field = InputFields.point_dipole_e(kr, krd[:], dip_x) # field of the point dipole with dipole moment "dip_x"
        phi_inc = DDACore.solve_DDA_e(kr,alp_e_dl;input_field=dipole_field)     # incident field at the silver partice 
        dipole_moment = PostProcessing.compute_dipole_moment(alp_e_dl,phi_inc) # dipole moment at the silver particle
        global rad_ldos_x[j,i] = PostProcessing.rad_ldos_e(kr,krd,dipole_moment,dip_x)          # radiative ldos x-axis
        global nonrad_ldos_x[j,i] = PostProcessing.nonrad_ldos_e(dipole_moment,phi_inc,dip_x)   # non-radiative ldos x-axis

    end
end

# plot ldos
pas = 5 
for ind_l = 1:2
    fig,axs=plt.subplots()
    fig.suptitle("LDOSx lambda = "*string(Int(lamb[ind_l]))*" nm")
    axs.plot(z,ldos_x_analytic[:,ind_l],"--b",label="total analytic")
    axs.plot(z,rad_ldos_x_analytic_shortdistance[:,ind_l],"--r",label="radiative analytic")
    axs.plot(z,nonrad_ldos_x_analytic_shortdistance[:,ind_l],"--g",label="non radiative analytic")
    axs.plot(z[1:pas:end],ldos_x[1:pas:end,ind_l],"ob",label="total DDA")
    axs.plot(z[1:pas:end],rad_ldos_x[1:pas:end,ind_l],"or",label="radiative DDA")
    axs.plot(z[1:pas:end],nonrad_ldos_x[1:pas:end,ind_l],"og",label="non radiative DDA")
    axs.set_xlabel("z (nm)")
    axs.set_ylabel("LDOS_x")
    axs.set_yscale("log")
    fig.tight_layout()
    axs.legend()
    plt.savefig("LDOSx"*string(Int(lamb[ind_l]))*".svg")

    fig,axs=plt.subplots()
    fig.suptitle("LDOSz lambda = "*string(Int(lamb[ind_l]))*" nm")
    axs.plot(z,ldos_z_analytic[:,ind_l],"--b",label="total analytic")
    axs.plot(z,rad_ldos_z_analytic_shortdistance[:,ind_l],"--r",label="radiative analytic")
    axs.plot(z,nonrad_ldos_z_analytic_shortdistance[:,ind_l],"--g",label="non radiative analytic")
    axs.plot(z[1:pas:end],ldos_z[1:pas:end,ind_l],"ob",label="total DDA")
    axs.plot(z[1:pas:end],rad_ldos_z[1:pas:end,ind_l],"or",label="radiative DDA")
    axs.plot(z[1:pas:end],nonrad_ldos_z[1:pas:end,ind_l],"og",label="non radiative DDA")
    axs.set_xlabel("z (nm)")
    axs.set_ylabel("LDOS_z")
    axs.set_yscale("log")
    fig.tight_layout()
    axs.legend()
    plt.savefig("LDOSz"*string(Int(lamb[ind_l]))*".svg")

end