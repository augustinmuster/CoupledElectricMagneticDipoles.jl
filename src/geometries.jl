module Geometries

export discretize_sphere, discretize_cube

###########################
# imports
###########################
using Base
using LinearAlgebra
###########################
# FUNCTIONS
###########################

@doc raw"""
    discretize_sphere(a,N;N_sub=10)

Discretizes the volume of a sphere of radius `a` in small cubes of edge `dx=2*a/N`. `N_sub` is a parameter to set the super sampling anti-aliasing accuracy of the discretization. It is by default set to 10.
Let ``N_d`` be the obtained number of cubes, the output is a tuple with an ``N_d \times 4`` array containing the 3D position of the centers of the cubes (first three entries) and their filling fraction (fourth entry), as well as the size of the edge of the cubes `dx`.
"""
function discretize_sphere(a,N;N_sub=10)
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
end


@doc raw"""
    discretize_cube(L,N)

Discretizes the volume of a cube of edge `L` in small cubes of edge `dx=L/N`.
Let ``N_d`` be the obtained number of cubes, the output is a tuple with an ``N_d \times 4`` array containing the 3D position of the centers of the cubes (first three entries) and their filling fraction (fourth entry), as well as the size of the edge of the cubes `dx`.
"""
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
     return latt,dx
end
end
