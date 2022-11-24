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
###########################
# FUNCTIONS
###########################
@doc raw"""
    gen_sphere_lattice_cubes(s,r;verbose=true)
Discretizes a sphere of radius `R` in cubes on a cubic lattice where the radius of this sphere is divided in `s` lattice constants.
`ref_id` is the refractive index id (we suppose that the entire sphere has the same refractive index.)
Returns 2D array with first dimension is the `N` units that compose the sphere and the second is
1) x-component of the position of the unit
2) y-component of the position of the unit
3) z-component of the position of the unit
4) distance from the origin
5) refractive index id
6) lattice parameter
7) volume of the cubic unit
"""
function gen_sphere_lattice_cubes(s,r;ref_id=1,verbose=true)
    #logging
    if verbose
        println()
        println("Generating a Sphere Lattice")
        println()
    end
    #computing volume
    V=(4/3)*pi*R^3
    #printing
    if verbose
        println("you choose a sphere of radius ",R)
        println("volume of the sphere:", V)
        println("radius in units of lattice parameter=",s)
    end
    #counter
    m=0
    #iterate over all lattice state
    for x=-(s+1):(s+1)
        for y=-(s+1):(s+1)
            for z=-(s+1):(s+1)
                #positions in real dimensions
                px=x
                py=y
                pz=z
                #test if it is in the sphere
                dis=sqrt(px^2+py^2+pz^2)
                if dis<s
                    #write if in the sphere
                    global m=m+1
                end
            end
        end
    end
    if verbose
        println("numberof dipoles: ",m)
    end
    #opening output file
    latt=zeros(Float64,m,7)
    #computing lattice parameter
    d=(V/m)^(1/3)
    #iterate over all lattice state
    m=1
    for x=-(s+1):(s+1)
        for y=-(s+1):(s+1)
            for z=-(s+1):(s+1)
                #positions in real dimensions
                px=x
                py=y
                pz=z
                #test if it is in the sphere
                dis=sqrt(px^2+py^2+pz^2)
                if dis<s
                    #format: x,y,z,norm,real(alpha),imag(alpha): delimiter: tab
                    latt[m,:]=[px*d,py*d,pz*d,dis*d,ref_id,d,d^3]
                    global m=m+1
                end
            end
        end
    end

    if verbose
        println("lattice gegenerated with ", m-1, " dipoles")
    end
    return latt
end

@doc raw"""
    gen_sphere_lattice_spheres(s,r;verbose=true)
Discretizes a sphere of radius `R` in spheres on a cubic lattice where the radius of this sphere is divided in `s` lattice constants.
`ref_id` is the refractive index id (we suppose that the entire sphere has the same refractive index.)
Returns 2D array with first dimension is the `N` units that compose the sphere and the second is
1) x-component of the position of the unit
2) y-component of the position of the unit
3) z-component of the position of the unit
4) distance from the origin
5) refractive index id
6) lattice parameter
7) volume of the spherical unit
"""
function gen_sphere_lattice_spheres(s,R;ref_id=1,verbose=true)
    #logging
    if verbose
        println()
        println("Generating a Sphere Lattice")
        println()
    end
    #computing volume
    V=(4/3)*pi*R^3
    #printing
    if verbose
        println("you choose a sphere of radius ",R)
        println("volume of the sphere:", V)
        println("radius in units of lattice parameter=",s)
    end
    #counter
    m=0
    #iterate over all lattice state
    for x=-(s+1):(s+1)
        for y=-(s+1):(s+1)
            for z=-(s+1):(s+1)
                #positions in real dimensions
                px=x
                py=y
                pz=z
                #test if it is in the sphere
                dis=sqrt(px^2+py^2+pz^2)
                if dis<s
                    #write if in the sphere
                    m=m+1
                end
            end
        end
    end
    if verbose
        println("numberof dipoles: ",m)
    end
    #opening output file
    latt=zeros(Float64,m,7)
    #computing lattice parameter, diameter of the unit spheres
    d=2*(3*V/m/4/pi)^(1/3)
    #iterate over all lattice state
    m=1
    for x=-(s+1):(s+1)
        for y=-(s+1):(s+1)
            for z=-(s+1):(s+1)
                #positions in real dimensions
                px=x
                py=y
                pz=z
                #test if it is in the sphere
                dis=sqrt(px^2+py^2+pz^2)
                if dis<s
                    #format: x,y,z,norm,real(alpha),imag(alpha): delimiter: tab
                    latt[m,:]=[px*d,py*d,pz*d,dis*d,ref_id,d,4/3*pi*d^3]
                    m=m+1
                end
            end
        end
    end

    if verbose
        println("lattice gegenerated with ", m-1, " dipoles")
    end
    return latt
end
end
