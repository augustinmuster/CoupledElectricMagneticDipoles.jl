using DelimitedFiles
using LaTeXStrings
#using LinearAlgebra
include("../../src/green_tensors_e_m.jl")
include("../../src/processing.jl")
include("../../src/DDA.jl")
include("../../src/mie_coeff.jl")
include("../../src/input_fields.jl")
include("../../src/alpha.jl")
include("../../src/geometries.jl")
include("../../src/forces.jl")

using PyCall
@pyimport matplotlib.pyplot as plt

##################### Parameters ########################################
#radius of the sphere
a=0.250
#dielectric constant of the particle
eps=(1.59)^2
#dielectric constant of the medium
eps_h=(1.33)^2
#number of wavelengths to compute
N_lambda=10
lambda_min=1
lambda_max=1.1
#wavelengths to compute
lambdas0=LinRange(lambda_min,lambda_max,N_lambda)
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

# parameters of the Gaussian Beam
# beam waist radius is set to lamb/2
kbw0 = pi # (2*pi/lambda)*(lamb/2)

# discretization of the position of the particle
ndis = 51 # odd number in order to mesh the "0" position
dis = LinRange(-2*lamb,2*lamb,ndis)
# variable where the force will be stored
force = zeros(ndis,3)

# loop on distances 
# note that, instead of moving the particle (and avoiding to recalculate the inverse DDA matrix), the position of the focus 
# of the Gaussian beam is changed.
for i=1:ndis 
    # forces along the x-axis when the particle is moving along the same axis (with y = z = 0)
    # evaluation of the Gaussian beam and its derivatives 
    krf = (latt[:,1:3] .+ [dis[i] 0 0])*knorm
    e_0inc = InputFields.gauss_beam_e(krf,kbw0)
    dxe_0inc, dye_0inc, dze_0inc = InputFields.d_gauss_beam_e(krf,kbw0)
    # calculation of forces 
    fx, fy, fz = Forces.force_e(kr,alpha, Ainv, e_0inc, dxe_0inc, dye_0inc, dze_0inc)
    global force[i,1] = sum(fx)

    # forces along the y-axis when the particle is moving along the same axis (with z = x = 0)
    # evaluation of the Gaussian beam and its derivatives 
    krf = (latt[:,1:3] .+ [0 dis[i] 0])*knorm
    e_0inc = InputFields.gauss_beam_e(krf,kbw0)
    dxe_0inc, dye_0inc, dze_0inc = InputFields.d_gauss_beam_e(krf,kbw0)
    # calculation of forces 
    fx, fy, fz = Forces.force_e(kr,alpha, Ainv, e_0inc, dxe_0inc, dye_0inc, dze_0inc)
    global force[i,2] = sum(fy)

    # forces along the z-axis when the particle is moving along the same axis (with x = y = 0)
    # evaluation of the Gaussian beam and its derivatives 
    krf = (latt[:,1:3] .+ [0 0 dis[i]])*knorm
    e_0inc = InputFields.gauss_beam_e(krf,kbw0)
    dxe_0inc, dye_0inc, dze_0inc = InputFields.d_gauss_beam_e(krf,kbw0)
    # calculation of forces
    fx, fy, fz = Forces.force_e(kr,alpha, Ainv, e_0inc, dxe_0inc, dye_0inc, dze_0inc)
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

# converse forces in Newtons
# laser intensity (10 mW)
power = 10e-3
# speed of light in the media
c_const = 3e8/sqrt(eps_h)
# factor for getting the forces in Newtons using the paraxial approximation for the Gaussian beam
factor = 16*power/c_const/kbw0^2
# force in Newtons
force = force*factor

# calculation of the stiffnesses of the trap by a linear fit around the zero force position
# for kx and ky, we directly assume that the zero force position is at the minimum of the "dis" array (at dis = 0)
# find the position of the minimum
val, ind_min_xy = findmin(abs.(dis))
# calculation of the stiffnesses along the x- and y-axis (N/um)
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
kx = round(kx*1e12,sigdigits=3)
ky = round(ky*1e12,sigdigits=3)
kz = round(kz*1e12,sigdigits=3)

# plotting results
fig, axs = plt.subplots()
axs.set_title(L"x-axis, bw_0 = \lambda/2, P = 10 mW")
axs.plot(dis,force[:,1]*1e12,label="")
axs.plot(dis_short,fx_lin*1e12,"--",label="kx = "*string(kx)*" pN/um")
axs.set_xlabel("x (um)")
axs.set_ylabel("Fx (pN)")
axs.legend()
fig.savefig("fx.svg")

fig, axs = plt.subplots()
axs.set_title(L"y-axis,bw_0 = \lambda/2, P = 10 mW")
axs.plot(dis,force[:,2]*1e12)
axs.plot(dis_short,fy_lin*1e12,"--",label="ky = "*string(ky)*" pN/um")
axs.set_xlabel("y (um)")
axs.set_ylabel("Fy (pN)")
axs.legend()
fig.savefig("fy.svg")

fig, axs = plt.subplots()
axs.set_title(L"z-axis, bw_0 = \lambda/2, P = 10 mW")
axs.plot(dis,force[:,3]*1e12)
axs.plot((dis_short .+ zmin),fz_lin*1e12,"--",label="kz = "*string(kz)*" pN/um")
axs.plot(dis,force[:,3]*0,"k--")
axs.set_xlabel("z (um)")
axs.set_ylabel("Fz (pN)")
axs.legend()
fig.savefig("fz.svg")


