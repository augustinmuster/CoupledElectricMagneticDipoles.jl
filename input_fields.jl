###########################
# IMPORTS
#######m####################
using Base
###########################
# FUNCTIONS
#######m####################
#*************************************************
#PLANE WAVE
#INPUTS:  norm of the wave vector, position vector,direction of the wave vector,polaristaion,
#OUTPUT: plane wave vector
#*************************************************
function plane_wave(knorm,r,khat=[0,0,1],e0=[1,0,0])
    return exp(im*dot(knorm*khat,r))*e0
end

#*************************************************
#PLANE WAVE,dimensionless input
#INPUTS:  norm of the wave vector, position vector,direction of the wave vector,polaristaion,
#OUTPUT: plane wave vector
#*************************************************
function plane_wave_dl(kr,khat=[0,0,1],e0=[1,0,0])
    return exp(im*dot(khat,kr))*e0
end