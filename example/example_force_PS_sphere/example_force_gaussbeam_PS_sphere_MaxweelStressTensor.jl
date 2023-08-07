using DelimitedFiles
using LaTeXStrings
using LinearAlgebra
using Lebedev
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

# definition of functions to calculate the forces using the Maxwell stress tensor
function maxwell_stress_tensor(phi)
"""
function for calcualting the Maxwell stress tensor
"""
    e = phi[1:3]
    h = phi[4:6]
    i3 = Matrix{ComplexF64}(I,3,3)
    mst = e*e' + h*h'- 1/2*i3*(e'*e + h'*h)
    return 1/2*real(mst)
end
function field_sca_e_em(kr, alpha_e_dl, phi_inc, krf)
"""
function for calcualting the electric and magnetic field scattered the essamble of electric dipolar particles.
"""
    n_particles = length(kr[:,1]) 
    n_r0 = length(krf[:,1]) 

    alp_e = Alphas.dispatch_e(alpha_e_dl,n_particles)

    G_tensor_fr = zeros(ComplexF64,n_r0*6,n_particles*3)
    Threads.@threads for i = 1:n_particles
        for j = 1:n_r0
            Ge, Gm = GreenTensors.G_em_renorm(krf[j,:],kr[i,:])   
            G_tensor_fr[6 * (j-1) + 1:6 * (j-1) + 6 , 3 * (i-1) + 1:3 * (i-1) + 3] = [Ge*alp_e[i]; -im*Gm*alp_e[i]]
	    end
    end
    phi_inc = reshape(transpose(phi_inc),n_particles*3,)
    field_r = G_tensor_fr*phi_inc
    return transpose(reshape(field_r,6,n_r0))       
end

function forces_gaussbeam(kr,alpha_e_dl,Ainv,kdis,kbw0; l_order=11, r_factor = 1.1)
#=
function for calculating the forces in a dipolar system when the incident field is a "x" polarized electric Gaussian beam propagating along the z-axis
with focus at "- kdis" (instead of moving the particle, the focus of the Gaussian beam is moved).
# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `Ainv`: (inverse) DDA matrix.
- `kdis`: renormalized position of the (negative of) focus of the Gaussian beam (the focus is at "- kdis").
- `kbw0`: float with the dimensionless beam waist radius (``k\omega_0``, where ``\omega_0`` is the beam waist radius).
- `l_order`: order for the Lebedev quadrature.
- `r_factor`: multiplicative factor to the radius of the sphere where the integration is done. It must be higher than one in order to enclose all particles

# Outputs
- `fx`: float array of Size ``N`` with the value of the force along the ``x``-axis at each dipole.
- `fy`: float array of Size ``N`` with the value of the force along the ``y``-axis at each dipole.
- `fz`: float array of Size ``N`` with the value of the force along the ``z``-axis at each dipole.
- `force_mst`: float of Size ``3`` with the value of the sum of the forces along x, y, z- axis calculated by integrating the Maxwell stress tensor.
=#
    if r_factor < 1
        r_factor = 1.1
    end
    # for integrating the Maxwell stress tensor, the discretitation of the sphere is done by the Lebedev quadrature
    x,y,z,w = lebedev_by_order(l_order)
    # normalized radius of the sphere (bigger than the distance of the further particle to the center)
    r_int = maximum(sum(abs.(kr),dims=2))*r_factor
    # point where the Maxwell stress tensor will be evaluated
    krl = [x y z]*r_int
    # variable where the force calculated wit the Maxwell stress tensor will be stored
    force_mst = zeros(ndis,3)

    # forces along the x-axis when the particle is moving along the same axis (with y = z = 0)
    # possitions where the Gaussian beam is calculated (instead of moving the particle, the focus of the Gaussian beam is moved)
    krf = kr .+ kdis
    # evaluation of the Gaussian beam and its derivatives
    # the function "InputFields.gauss_beam_e" can be simply used for getting only the electric field of the Gaussian Beam, but the magnetic components are needed for the Maxwell stress tensor
    e_0inc = InputFields.gauss_beam_e_m(krf,kbw0)
    e_0inc_e = e_0inc[:,1:3]
    dxe_0inc, dye_0inc, dze_0inc = InputFields.d_gauss_beam_e(krf,kbw0)
    # calculation of forces using the dipole approximation (arrays with the value of the forces in each particle/dipole)
    fx, fy, fz = Forces.force_e(kr,alpha, Ainv, e_0inc_e, dxe_0inc, dye_0inc, dze_0inc)

    # forces calculated by the integral in the Maxwell stress tensor (flux of linear momentum)
    # calculation of the total inciden field on the dipoles
    e_0inc = reshape(transpose(e_0inc_e),n*3,)
    e_inc = Ainv*e_0inc
    e_inc = transpose(reshape(e_inc,3,n))
    # calculation of the total field at the position of an sphere that surround the particles (external field plus field scattered by the particles)
    # possitions where the Gaussian beam is calculated (instead of moving the particle and sphere that surround them, the focus of the Gaussian beam is moved)
    krf_l = krl .+ kdis
    # evaluation of the Gaussian beam
    e_0inc_l = InputFields.gauss_beam_e_m(krf_l,kbw0)
    # evaluation of the field (electric and magnetic) scattered by the (only electric) particles. 
    e_sca_r = field_sca_e_em(kr, alpha, e_inc, krl)
    # total field
    e_t = e_0inc_l + e_sca_r
    # variable for storing the acumulate sum
    force_mst = zeros(3)
    # loop in the sphere discretitation for doing the integral as a sum.
    for i = 1:length(x)
        # sphericall coordinate angles
        theta = acos(z[i])
        phi = atan(y[i],x[i])
        # evaluation of the Maxwell stress tennsor
        mst = maxwell_stress_tensor(e_t[i,:])
        # unit vector of the differential surface
        nr = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
        # sum of the forces
        f = force_mst + mst*nr*w[i]
        force_mst = f
    end
    # r^2 factor of the spherical surface integral
    force_mst = force_mst*r_int^2

    return fx, fy, fz, force_mst
end
# Code for the optical forces
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
r = latt[:,1:3]
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
# order for the Lebedev quadrature
l_order = 53
# variable where the force will be stored
force = zeros(ndis,3)
# variable where the force calculated with the Maxwell stress tensor will be stored
force_mst = zeros(ndis,3)

# loop on distances 
# note that, instead of moving the particle (and avoiding to recalculate the inverse DDA matrix), the position of the focus 
# of the Gaussian beam is changed.
for i=1:ndis   
    # forces along the x-axis when the particle is moving along the same axis (with y = z = 0)
    kdis = [dis[i] 0 0]*knorm
    fx, fy, fz, force_i = forces_gaussbeam(kr,alpha,Ainv,kdis,kbw0; l_order=l_order)
    global force[i,1] = sum(fx)
    global force_mst[i,1] = force_i[1]
    # forces along the y-axis when the particle is moving along the same axis (with z = x = 0)
    kdis = [0 dis[i] 0]*knorm
    fx, fy, fz, force_i = forces_gaussbeam(kr,alpha,Ainv,kdis,kbw0; l_order=l_order)
    global force[i,2] = sum(fy)
    global force_mst[i,2] = force_i[2]
    # forces along the z-axis when the particle is moving along the same axis (with x = y = 0)
    kdis = [0 0 dis[i]]*knorm
    fx, fy, fz, force_i = forces_gaussbeam(kr,alpha,Ainv,kdis,kbw0; l_order=l_order)
    global force[i,3] = sum(fz)
    global force_mst[i,3] = force_i[3]
end

#=
# save data
fout=open("dis.dat","w")
writedlm(fout,dis)
close(fout)
fout=open("force_101.dat","w")
writedlm(fout,force)
close(fout)
fout=open("force_mst_lebedev53.dat","w")
writedlm(fout,force_mst)
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
force_mst = force_mst*factor

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

# number the point to skip for plotting the forces calcualted by integrating the Maxwell stress tensor
pas = 4
# plotting results
fig, axs = plt.subplots()
axs.set_title(L"x-axis, bw_0 = \lambda/2, P = 10 mW")
axs.plot(dis,force[:,1]*1e12, label="dipole aprox")
axs.plot(dis[1:pas:end],force_mst[1:pas:end,1]*1e12,"*", label="Maxwell stress tensor")
axs.plot(dis_short,fx_lin*1e12,"--",label="kx = "*string(kx)*" pN/um")
axs.set_xlabel("x (um)")
axs.set_ylabel("Fx (pN)")
axs.legend()
fig.savefig("fx_mst.svg")

fig, axs = plt.subplots()
axs.set_title(L"y-axis,bw_0 = \lambda/2, P = 10 mW")
axs.plot(dis,force[:,2]*1e12, label="dipole aprox")
axs.plot(dis[1:pas:end],force_mst[1:pas:end,2]*1e12,"*", label="Maxwell stress tensor")
axs.plot(dis_short,fy_lin*1e12,"--",label="ky = "*string(ky)*" pN/um")
axs.set_xlabel("y (um)")
axs.set_ylabel("Fy (pN)")
axs.legend()
fig.savefig("fy_mst.svg")

fig, axs = plt.subplots()
axs.set_title(L"z-axis, bw_0 = \lambda/2, P = 10 mW")
axs.plot(dis,force[:,3]*1e12, label="dipole aprox")
axs.plot(dis[1:pas:end],force_mst[1:pas:end,3]*1e12,"*", label="Maxwell stress tensor")
axs.plot((dis_short .+ zmin),fz_lin*1e12,"--",label="kz = "*string(kz)*" pN/um")
axs.plot(dis,force[:,3]*0,"k--")
axs.set_xlabel("z (um)")
axs.set_ylabel("Fz (pN)")
axs.legend()
fig.savefig("fz_mst.svg")
