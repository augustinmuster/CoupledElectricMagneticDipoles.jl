###########################
# imports
###########################
using Base
using DelimitedFiles
using LinearAlgebra
###########################
# RUN PARAMETERS
###########################
#radius in term of lattice parameter
s=4
#radius of the sphere
R=230e-9
#verbose
verbose=true
###########################
# MAIN CODE
###########################
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
fout=open("sphere_lattice.dat","w")
#computing lattice parameter (diameter of the spheres)
d=2*(3*V/m/4/pi)^(1/3)
#iterate over all lattice state
alpha=zeros(ComplexF64,3,3)
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
                writedlm(fout,[px*d py*d pz*d dis*d 1 d d^3])
            end
        end
    end
end

if verbose
    println("lattice gegenerated with ", m, " dipoles")
end
#close ouput file
close(fout)
println(m)
