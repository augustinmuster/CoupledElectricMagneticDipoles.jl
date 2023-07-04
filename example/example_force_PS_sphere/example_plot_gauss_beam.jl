
using Base
using PyCall
using LinearAlgebra
using DelimitedFiles
include("../../CoupledElectricMagneticDipoles/src/input_fields.jl")

pushfirst!(pyimport("sys")."path", "")

@pyimport numpy
@pyimport matplotlib.pyplot as plt

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

# Parameters at lambda_0 = 1000 nm
# wavelength
lamb = lambdas[1]
# wavevector
knorm=2*pi/lamb
# radius beam waist
bw0 = lamb/2

# discretitation of the position
nz = 201
z = LinRange(-2*lamb,2*lamb,nz)

# xy plane cut
# variables for storing the field
ex = zeros(ComplexF64,nz,nz)
ez = zeros(ComplexF64,nz,nz)
for i=1:nz # loop in x
    for j=1:nz # loop in y
        rf = [z[i],z[j],0]'
        e_0inc = InputFields.gauss_beam_e(rf,knorm,bw0)
        global ex[i,j] = e_0inc[1,1]
        global ez[i,j] = e_0inc[1,3]
    end
end

# plot field
fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_x|, xy-plane, w0 = lambda/2")
CS = axs.contourf(z*1e9,z*1e9,abs.(ex),50)
axs.set_xlabel("y (nm)")
axs.set_ylabel("x (nm)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_x|")
fig.savefig("gb_Ex_xy.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_z|, xy-plane, w0 = lambda/2")
CS = axs.contourf(z*1e9,z*1e9,abs.(ez),50)
axs.set_xlabel("y (nm)")
axs.set_ylabel("x (nm)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_z|")
fig.savefig("gb_Ez_xy.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E|, xy-plane, w0 = lambda/2")
CS = axs.contourf(z*1e9,z*1e9,sqrt.(abs2.(ex) + abs2.(ez)),50)
axs.set_xlabel("y (nm)")
axs.set_ylabel("x (nm)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E|")
fig.savefig("gb_absE_xy.svg")
plt.show()

# yz plane cut
# variables for storing the field
ex = zeros(ComplexF64,nz,nz)
ez = zeros(ComplexF64,nz,nz)
for i=1:nz # loop in y
    for j=1:nz # loop in z
        rf = [0,z[i],z[j]]'
        e_0inc = InputFields.gauss_beam_e(rf,knorm,bw0)
        global ex[i,j] = e_0inc[1,1]
        global ez[i,j] = e_0inc[1,3]
    end
end

# plot field
fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_x|, yz-plane, w0 = lambda/2")
CS = axs.contourf(z*1e9,z*1e9,abs.(ex),50)
axs.set_xlabel("z (nm)")
axs.set_ylabel("y (nm)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_x|")
fig.savefig("gb_Ex_yz.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_z|, yz-plane, w0 = lambda/2")
CS = axs.contourf(z*1e9,z*1e9,abs.(ez),50)
axs.set_xlabel("z (nm)")
axs.set_ylabel("y (nm)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_z|")
fig.savefig("gb_Ez_yz.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E|, yz-plane, w0 = lambda/2")
CS = axs.contourf(z*1e9,z*1e9,sqrt.(abs2.(ex) + abs2.(ez)),50)
axs.set_xlabel("z (nm)")
axs.set_ylabel("y (nm)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E|")
fig.savefig("gb_absE_yz.svg")
plt.show()

# zx plane cut
# variables for storing the field
ex = zeros(ComplexF64,nz,nz)
ez = zeros(ComplexF64,nz,nz)
for i=1:nz # loop in z
    for j=1:nz # loop in x
        rf = [z[j],0,z[i]]'
        e_0inc = InputFields.gauss_beam_e(rf,knorm,bw0)
        global ex[i,j] = e_0inc[1,1]
        global ez[i,j] = e_0inc[1,3]
    end
end

# plot field
fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_x|, zx-plane, w0 = lambda/2")
CS = axs.contourf(z*1e9,z*1e9,abs.(ex),50)
axs.set_xlabel("x (nm)")
axs.set_ylabel("z (nm)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_x|")
fig.savefig("gb_Ex_zx.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_z|, zx-plane, w0 = lambda/2")
CS = axs.contourf(z*1e9,z*1e9,abs.(ez),50)
axs.set_xlabel("x (nm)")
axs.set_ylabel("z (nm)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_z|")
fig.savefig("gb_Ez_zx.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E|, zx-plane, w0 = lambda/2")
CS = axs.contourf(z*1e9,z*1e9,sqrt.(abs2.(ex) + abs2.(ez)),50)
axs.set_xlabel("x (nm)")
axs.set_ylabel("z (nm)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E|")
fig.savefig("gb_absE_zx.svg")
plt.show()