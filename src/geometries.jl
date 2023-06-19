"""
CoupledElectricMagneticDipoles.jl : Geometries Module
This module contains a small library to discretize spheres on a cubic lattice.
Author: Augustin Muster, November 2022, augustin@must-r.com
"""
module Geometries
###########################
# imports
###########################
using Base
using LinearAlgebra
using PyCall
@pyimport matplotlib.pyplot as plt
###########################
# FUNCTIONS
###########################

function discretize_sphere(a,N,N_sub)
    #lattice parameter
    dx=2*a/N
    #sublattice parameter
    dx_sub=dx/N_sub
    #
    center=[0.,0.,0.]
    #
    latt=[]
    #
    total_sub=N_sub^3
    #
    for i=1:N
        xc=(i-0.5)*dx-a
        for j=1:N
            yc=(j-0.5)*dx-a
            for k=1:N
                zc=(k-0.5)*dx-a
                inside=0
                for n1=1:N_sub
                    center[1]=(n1-0.5)*dx_sub+xc-dx/2
                    for n2=1:N_sub
                        center[2]=(n2-0.5)*dx_sub+yc-dx/2
                        for n3=1:N_sub
                            center[3]=(n3-0.5)*dx_sub+zc-dx/2
                            if norm(center)<a
                                inside+=1
                            end
                        end
                    end
                end
                if inside!=0
                    append!(latt,[xc yc zc inside/total_sub 1])
                end
            end
        end
    end

    latt=transpose(reshape(latt,(5,:)))
    return latt, dx
    #=
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    n=length(latt[:,1])
    color=zeros(n,4)
    color[:,3].=1
    color[:,4]=latt[:,4]
    ax.voxels(latt[:,1],latt[:,2],latt[:,3],ones(n,n,n),facecolor=color,edgecolor="grey")
    plt.show()
    =#
end



function discretize_cube(L,N)
     #lattice parameter
     dx=L/N
     #
     latt=[]
     #
     for i=1:N
         xc=(i-0.5)*dx-L/2
         for j=1:N
             yc=(j-0.5)*dx-L/2
             for k=1:N
                zc=(k-0.5)*dx-L/2
                append!(latt,[xc yc zc 1])
             end
         end
     end
     latt=transpose(reshape(latt,(4,:)))
     #=
     fig = plt.figure(figsize=(12, 12))
     ax = fig.add_subplot(projection="3d")
     ax.scatter(latt[:,1],latt[:,2],latt[:,3])
     plt.show()
     =#
     return latt,dx
end
end
