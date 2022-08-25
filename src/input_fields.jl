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

#*************************************************
#PLANE WAVE ELECTRIC AND MAGNETIC
#INPUTS:  norm of the wave vector, position vector,direction of the wave vector,polaristaion,
#OUTPUT: plane wave vector
#*************************************************
function plane_wave_e_m(knorm,r,khat=[0,0,1],e0=[1,0,0])
    E=exp(im*dot(knorm*khat,r))*e0
    H=cross(khat,E)
    return E,H
end

#*************************************************
#PLANE WAVE ELECTRIC AND MAGNETIC, RENORM
#INPUTS:   position vector,direction of the wave vector,polaristaion,
#OUTPUT: plane wave vector
#*************************************************
function plane_wave_e_m_renorm(r,khat=[0,0,1],e0=[1,0,0])
    E=exp(im*dot(khat,r))*e0
    H=cross(khat,E)
    return E,H
end