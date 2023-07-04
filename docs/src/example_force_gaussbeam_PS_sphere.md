# Electromagentic forces of a Gaussian Beam on a Polystyrene Sphere

This example aims to teach the user how to calculate optical forces, as well as to calculate the field and its derivatives of a tight-focused Gaussian beam. Note that the derivaties are implemented as adimensional. Thus, both the field and the derivatives would have the same units.

The forces will be calculated along the three axes, x-, y- and z-axis. For example, for the forces along the x-axis, we will take y = z = 0 (and the same for the other axis). Also, with the forces along a given axis, we mean that the input field is a Gaussian beam focus at the origin of coordinates, while the force is calculated as a function of the position of the center of the Polystyrene Sphere. Nonetheless, in order to avoid the recalculation of the DDA matrix, in the calculation the particle is fixed at the center of coordinates, while the focus of the Gaussian Beam is moving.

If you want to run this example, copy it or download it on the github (`example_force_gaussbeam_PS_sphere.jl`) and run it using 

```bash
julia example_force_gaussbeam_PS_spher.jl

```
If you can, it is recommended to run it in parallel, using the `--threads` option. 

Let's start by importing CoupledElectricMagneticDipoles.jl. Note that we also use LaTeXStrings and PyCall, because we will use the python matplotlib library in order to plot the results. LinearAlgebra is used for....

```julia
#imports
using CoupledElectricMagneticDipoles
using PyCall
using LaTeXStrings
#using DelimitedFiles 
@pyimport matplotlib.pyplot as plt

```
## Discretizing the Sphere and polarizability

We then need to start modeling our particle in water. The parameters are the same used for `example_PS_sphere.jl`. Thus, please visit this example for more details.
```julia
##################### Parameters ########################################
#radius of the sphere
a=250e-9
#dielectric constant of the particle
eps=(1.59)^2
#dielectric constant of the medium
eps_h=(1.33)^2
#number of wavelengths to compute
N_lambda=10
lambda_min=1000e-9
lambda_max=1100e-9
#wavelengths to compute
lambdas0=LinRange(1000e-9,1100e-9,N_lambda)
lambdas=lambdas0/sqrt(eps_h)
##########################################################################

#discretizes a sphere in small cubes
latt,dx=Geometries.discretize_sphere(a,10)
n=length(latt[:,1])

# Parameters for the forces at lambda_0 = 1000 nm
# wavelength
lamb = lambdas[1]
# wavevector
knorm=2*pi/lamb
# renormalized distance of the dipoles
kr = knorm*latt[:,1:3]

#computes polarizability for each dipoles using effective dielectric constant 
alpha=zeros(ComplexF64,n,3,3)
for j=1:n
    eps_eff=latt[j,4]*eps+(1-latt[j,4])*eps_h
    alpha[j,:,:]=Alphas.alpha_radiative(Alphas.alpha0_parallelepiped(dx,dx,dx,eps_eff,eps_h),knorm)
end
# calculation of the inverse DDA matrix
Ainv = DDACore.solve_DDA_e(kr,alpha)
```
Note that in this example `DDACore.solve_DDA_e` has no `input_field argument`. Then, the output is the inverse DDA matrix. 

## Setting incoming field and particle position

As an incoming field, we will use a Gaussian Beam with radius beam waist w0 = \lambda/2, and intensity at the focus I0 = 25 GW/m^2. Also, the forces will be calculated along the three axes, between [-2\lambda, 2\lambda], discretizing the space in 51 points. For convenience, it is better to use an odd number of points in order to take the 0.

```julia
# parameters of the Gaussian Beam
# radius beam waist
bw0 = lamb/2
# intensity at focus (25 GW/m^2)
intensity = 25e9
# electromagnetic constants
eps0 = 8.8541878128e-12
c_const = 3e8
# electric field amplitude at focus (V/m)
e0 = sqrt(2*intensity/c_const/eps0/eps_h)

# discretization of the position of the particle
ndis = 51 # odd number in order to mesh the "0" position
dis = LinRange(-2*lamb,2*lamb,ndis)
# variable where the force will be stored
force = zeros(ndis,3)
```
For knowing what is the field distribution of this field, the file `example_plot_gauss_beam.jl` can be runned.

## Calculating the forces

Finally, we can then open a loop and computes the forces as follow:

```julia
# loop on distances 
# note that, instead of moving the particle (and avoiding to recalculate the inverse DDA matrix), the position of the focus 
# of the Gaussian beam is changed.
for i=1:ndis 
    # forces along the x-axis when the particle is moving along the same axis (with y = z = 0)
    # evaluation of the Gaussian beam and its derivatives 
    rf = latt[:,1:3] .+ [dis[i] 0 0]
    e_0inc = InputFields.gauss_beam_e(rf,knorm,bw0,e0 = e0)
    dxe_0inc, dye_0inc, dze_0inc = InputFields.d_gauss_beam_e(rf,knorm,bw0,e0 = e0)
    # calculation of forces 
    fx, fy, fz = Forces.force_e(knorm,kr,alpha, Ainv, e_0inc, dxe_0inc, dye_0inc, dze_0inc)
    global force[i,1] = sum(fx)

    # forces along the y-axis when the particle is moving along the same axis (with z = x = 0)
    # evaluation of the Gaussian beam and its derivatives 
    rf = latt[:,1:3] .+ [0 dis[i] 0]
    e_0inc = InputFields.gauss_beam_e(rf,knorm,bw0,e0 = e0)
    dxe_0inc, dye_0inc, dze_0inc = InputFields.d_gauss_beam_e(rf,knorm,bw0,e0 = e0)
    # calculation of forces 
    fx, fy, fz = Forces.force_e(knorm,kr,alpha, Ainv, e_0inc, dxe_0inc, dye_0inc, dze_0inc)
    global force[i,2] = sum(fy)

    # forces along the z-axis when the particle is moving along the same axis (with x = y = 0)
    # evaluation of the Gaussian beam and its derivatives 
    rf = latt[:,1:3] .+ [0 0 dis[i]]
    e_0inc = InputFields.gauss_beam_e(rf,knorm,bw0,e0 = e0)
    dxe_0inc, dye_0inc, dze_0inc = InputFields.d_gauss_beam_e(rf,knorm,bw0,e0 = e0)
    # calculation of forces
    fx, fy, fz = Forces.force_e(knorm,kr,alpha, Ainv, e_0inc, dxe_0inc, dye_0inc, dze_0inc)
    global force[i,3] = sum(fz)
end
#=
# save data
fout=open("dis.dat","w")
writedlm(fout,dis)
close(fout)
fout=open("force.dat","w")
writedlm(fout,force)
close(fout)
=#
```
As it is explained above, for the calculation the Polystyrene Sphere is keeped at the origin of coordinates, while the focus of the derivatives beam is changed. However, (for reciprocity) we interpret the results as the forces on the Polystyrene Sphere as it is moving out the focus. 

To save the data, uncomment the last lines and include "using DelimitedFiles".

## Calculating stiffnesses

Since the tight-focused Gaussian beam is going to trap the particle, the forces around the equilibrium position can be approximated by a linear model. The depth of the trap can be characterized by the stiffnesses along the different axis, a simple spring model. To estimate the stiffnesses, a linear fit can be done.

```julia
# calculation of the stiffnesses of the trap by a linear fit around the zero force position
# for kx and ky, we directly assume that the zero force position is at the minimum of the "dis" array (at dis = 0)
# find the position of the minimum
val, ind_min_xy = findmin(abs.(dis))
# calculation of the stiffnesses along the x- and y-axis
kx = -(force[ind_min_xy+1,1]-force[ind_min_xy-1,1])/(dis[ind_min_xy+1] - dis[ind_min_xy-1])
ky = -(force[ind_min_xy+1,2]-force[ind_min_xy-1,2])/(dis[ind_min_xy+1] - dis[ind_min_xy-1])
# for kz the minimum is found as the first minimum along the z-axis (the minimum is not at "z=0")
ind_min_z = ind_min_xy
while abs(force[ind_min_z,3]) > abs(force[ind_min_z+1,3])
    min_z = ind_min_z + 1
    global ind_min_z = min_z
end
# calculation of the stiffnesses along the z-axis
kz = -(force[ind_min_z+1,3]-force[ind_min_z-1,3])/(dis[ind_min_z+1] - dis[ind_min_z-1])
# linear calculation of for the position of the minimum
zmin = dis[ind_min_z] + force[ind_min_z,3]/kz
# shorter array for plotting the linear approximation of the forces
dis_short = LinRange(-lamb/4,lamb/4,ndis)
# linear approximation of the force around the zero
fx_lin = -kx*dis_short
fy_lin = -ky*dis_short
fz_lin = -kz*(dis_short)
# rounding the value of the stiffnesses for the legend
kx = round(kx*1e6,sigdigits=3)
ky = round(ky*1e6,sigdigits=3)
kz = round(kz*1e6,sigdigits=3)
```

## Plotting forces

Once the forces and the stiffnesses are calculated, they can be plotted using the next lines:

```julia
# plotting results
fig, axs = plt.subplots()
axs.set_title(L"x-axis, \lambda_0 = 1000 nm, \lambda = \lambda_0/1.33, w0 = \lambda/ 2, I_0 = 25 GW/m^2")
axs.plot(dis*1e9,force[:,1]*1e15,label="")
axs.plot(dis_short*1e9,fx_lin*1e15,"--",label="kx = "*string(kx)*" fN/nm")
axs.set_xlabel("x (nm)")
axs.set_ylabel("Fx (fN)")
axs.legend()
fig.savefig("fx.svg")

fig, axs = plt.subplots()
axs.set_title(L"y-axis, \lambda_0 = 1000 nm, \lambda = \lambda_0/1.33, w0 = \lambda/ 2, I_0 = 25 GW/m^2")
axs.plot(dis*1e9,force[:,2]*1e15)
axs.plot(dis_short*1e9,fy_lin*1e15,"--",label="ky = "*string(ky)*" fN/nm")
axs.set_xlabel("y (nm)")
axs.set_ylabel("Fy (fN)")
axs.legend()
fig.savefig("fy.svg")

fig, axs = plt.subplots()
axs.set_title(L"z-axis, \lambda_0 = 1000 nm, \lambda = \lambda_0/1.33, w0 = \lambda/ 2, I_0 = 25 GW/m^2")
axs.plot(dis*1e9,force[:,3]*1e15)
axs.plot((dis_short .+ zmin)*1e9,fz_lin*1e15,"--",label="kz = "*string(kz)*" fN/nm")
axs.plot(dis*1e9,force[:,3]*0,"k--")
axs.set_xlabel("z (nm)")
axs.set_ylabel("Fz (fN)")
axs.legend()
fig.savefig("fz.svg")
```

```@raw html
<img src="../assets/fx.svg">
```

```@raw html
<img src="../assets/fy.svg">
```

```@raw html
<img src="../assets/fz.svg">
```
