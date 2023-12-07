
using Base
using PyCall
using LinearAlgebra
using DelimitedFiles
include("../../src/CoupledElectricMagneticDipoles.jl")
using .CoupledElectricMagneticDipoles

pushfirst!(pyimport("sys")."path", "")

using PyCall
#@pyimport numpy as np
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

# Parameters at lambda_0 = 1000 nm
# wavelength
lamb = lambdas[1]
# wavevector
knorm=2*pi/lamb
# beam waist radius is set to lamb/2
kbw0 = pi # (2*pi/lambda)*(lamb/2)

# discretitation of the position
nz = 201
kr = LinRange(-2*lamb,2*lamb,nz)*knorm

# xy plane cut
# variables for storing the field
ex = zeros(ComplexF64,nz,nz)
ez = zeros(ComplexF64,nz,nz)
for i=1:nz # loop in x
    for j=1:nz # loop in y
        krf = [kr[i],kr[j],0]'
        e_0inc = InputFields.gaussian_beam_e(krf,kbw0)
        global ex[i,j] = e_0inc[1,1]
        global ez[i,j] = e_0inc[1,3]
    end
end

# plot field
fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_x|, xy-plane, w0 = lambda/2")
CS = axs.contourf(kr,kr,abs.(ex),50)
axs.set_xlabel("y (um)")
axs.set_ylabel("x (um)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_x|")
fig.savefig("gb_Ex_xy.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_z|, xy-plane, w0 = lambda/2")
CS = axs.contourf(kr,kr,abs.(ez),50)
axs.set_xlabel("y (um)")
axs.set_ylabel("x (um)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_z|")
fig.savefig("gb_Ez_xy.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E|, xy-plane, w0 = lambda/2")
CS = axs.contourf(kr,kr,sqrt.(abs2.(ex) + abs2.(ez)),50)
axs.set_xlabel("y (um)")
axs.set_ylabel("x (um)")
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
        krf = [0,kr[i],kr[j]]'
        e_0inc = InputFields.gaussian_beam_e(krf,kbw0)
        global ex[i,j] = e_0inc[1,1]
        global ez[i,j] = e_0inc[1,3]
    end
end

# plot field
fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_x|, yz-plane, w0 = lambda/2")
CS = axs.contourf(kr,kr,abs.(ex),50)
axs.set_xlabel("z (um)")
axs.set_ylabel("y (um)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_x|")
fig.savefig("gb_Ex_yz.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_z|, yz-plane, w0 = lambda/2")
CS = axs.contourf(kr,kr,abs.(ez),50)
axs.set_xlabel("z (um)")
axs.set_ylabel("y (um)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_z|")
fig.savefig("gb_Ez_yz.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E|, yz-plane, w0 = lambda/2")
CS = axs.contourf(kr,kr,sqrt.(abs2.(ex) + abs2.(ez)),50)
axs.set_xlabel("z (um)")
axs.set_ylabel("y (um)")
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
        krf = [kr[j],0,kr[i]]'
        e_0inc = InputFields.gaussian_beam_e(krf,kbw0)
        global ex[i,j] = e_0inc[1,1]
        global ez[i,j] = e_0inc[1,3]
    end
end

# plot field
fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_x|, zx-plane, w0 = lambda/2")
CS = axs.contourf(kr,kr,abs.(ex),50)
axs.set_xlabel("x (um)")
axs.set_ylabel("z (um)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_x|")
fig.savefig("gb_Ex_zx.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E_z|, zx-plane, w0 = lambda/2")
CS = axs.contourf(kr,kr,abs.(ez),50)
axs.set_xlabel("x (um)")
axs.set_ylabel("z (um)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E_z|")
fig.savefig("gb_Ez_zx.svg")
plt.show()

fig, axs = plt.subplots()
axs.set_title("Gauss Beam |E|, zx-plane, w0 = lambda/2")
CS = axs.contourf(kr,kr,sqrt.(abs2.(ex) + abs2.(ez)),50)
axs.set_xlabel("x (um)")
axs.set_ylabel("z (um)")
axs.set_aspect("equal", "box")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("|E|")
fig.savefig("gb_absE_zx.svg")
plt.show()