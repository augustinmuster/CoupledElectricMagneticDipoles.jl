# Local density of states for a metallic nanoparticle

This example aims to teach the user how to use other advanced utilities such as the local density of states (LDOS) by reproducing the results of R. Carminati et. al., Opt. Comm. 261, 368 (2006). The system under study is a silver particle of radius 5 nm around its plasmon-resonance frequency (lambda = 354 nm) and at out-of-resonance (lambda = 612 nm). The numerical projected LDOS is then compared with the analytical results derived in the manuscript.  

If you want to run this example, copy it or download it on the github (`example_ldos_silver_np.jl`) and run it using 

```bash
julia example_ldos_silver_np.jl

```
Let's start by importing CoupledElectricMagneticDipoles.jl. Note that we also use PyCall, because we will use the python matplotlib library in order to plot the results.

```julia
#imports
using CoupledElectricMagneticDipoles
using PyCall
@pyimport matplotlib.pyplot as plt

```
## Defining analytical solutions from R. Carminati et. al., Opt. Comm. 261, 368 (2006)

Let first define the analytical solution of the LDOS, that it takes as inputs the dimensionless distance `kz` and polarizability `alp_dl`, and the outpus are the projected LDOS along the z- and x-axis. By definition, the silver nanoparticle will be placed at the origin of the coordinate system, and the LDOS is measured along the z-axis.

```julia
# analytic solution R. Carminati et. al., Opt. Comm. 261, 368 (2006)
function ldos_analytic(kz, alp_dl)
    ldos_z = 1 + 6*imag(alp_dl*exp(2*im*kz)*(-1/kz^4 - 2*im/kz^5 + 1/kz^6) )
    ldos_x = 1 + 3/2*imag(alp_dl*exp(2*im*kz)*(1/kz^2 + 2*im/kz^3 - 3/kz^4 - 2*im/kz^5 + 1/kz^6) )
    return ldos_z, ldos_x
end
```

## Setting physical properties

Now let's set the parameters of the system, as well as the variables where the LDOS will be stored.

```julia
# physical properties
# particle radius
a = 5e-9 
# wavelengths
lamb = [612e-9, 354e-9] 
# dielectric constant of the particle
eps=[-15.04 + im*1.02, -2.03 + im*0.6] 
# distance between particle and dipole 
nz = 91 
z = LinRange(10,100,nz)*1e-9 
```
The values of the permittivity is taken directly from the manuscript, that correspond with the permittivity at the specific wavelengths, as can be checked in E.W. Palik, Handbook of Optical Constants of Solids, Academic Press, San Diego, 1985 for bulk silver. 

## Computing the LDOS

We can then open a loop and computes the LDOS as follows:

```julia
# ldos calculation at both wavelengths and all distances
for i=1:2 # loop in wavelength
    # wavevector
    k = 2*pi/lamb[i] 
    # permittivity
    eps_i = eps[i] 
    # calculation of the polarizability
    alp_0 = 4*pi*a^3*(eps_i - 1)/(eps_i + 2) # static polarizability
    alp = alp_0/(1 - im*k^3/(6*pi)*alp_0) # radiative correction to the polarizability
    alp_e_dl = alp*k^3/(4*pi) # dimensionless polarizability
    for j=1:nz # loop in distance
        # distance
        z_j = z[j] 
        # renormalized distance
        kz = k*z_j
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
```
Here, the polarizability has been calculated using Eq. 5-6 of the manuscript. The selection of the projection of the LDOS is done by the `dip` argument. It is also possible to pass an array as an argument, defining the dipole moment of the testing dipolar source as

```julia
dip_vec = zeros(3)
dip_vec[3] = 1 
global ldos_z[j,i] = PostProcessing.ldos_e(kr, alp_e_dl, Ainv, krd, dip = dip_vec) # ldos z-axis
```
This way of calculating the projection along the z-axis would lead into the same result. Also, `dip` could be any three (of six for electric and magnetic dipoles) dimensional vector.

It is now possible to plot the LDOS, comparing the numerical and analytical calculations. The plot is made using the python library matplotlib called in julia by the intermediate of the PyCall library, but you can plot it with any software of your choice.

```julia
# plot ldos
for ind_l = 1:2
    fig,axs=plt.subplots()
    fig.suptitle("LDOSx lambda = "*string(Int(lamb[ind_l]*1e9))*" nm")
    axs.plot(z*1e9,ldos_x_analytic[:,ind_l],"--",label="LDOSx analytic")
    axs.plot(z*1e9,ldos_x[:,ind_l],"o",label="LDOSx DDA")
    axs.set_xlabel("z (nm)")
    axs.set_ylabel("LDOS_x")
    axs.set_yscale("log")
    fig.tight_layout()
    axs.legend()
    plt.savefig("LDOSx"*string(Int(lamb[ind_l]*1e9))*".png")

    fig,axs=plt.subplots()
    fig.suptitle("LDOSz lambda = "*string(Int(lamb[ind_l]*1e9))*" nm")
    axs.plot(z*1e9,ldos_z_analytic[:,ind_l],"--",label="LDOSz analytic")
    axs.plot(z*1e9,ldos_z[:,ind_l],"o",label="LDOSz DDA")
    axs.set_xlabel("z (nm)")
    axs.set_ylabel("LDOS_z")
    axs.set_yscale("log")
    fig.tight_layout()
    axs.legend()
    plt.savefig("LDOSz"*string(Int(lamb[ind_l]*1e9))*".png")
end
```
This is what we get:

```@raw html
<img src="../assets/LDOSx354.svg">
```
```@raw html
<img src="../assets/LDOSz354.svg">
```
```@raw html
<img src="../assets/LDOSx612.svg">
```
```@raw html
<img src="../assets/LDOSz612.svg">
```

