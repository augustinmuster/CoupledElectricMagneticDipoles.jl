###########################
# IMPORTS
###########################
using Base
using LinearAlgebra
###########################
# FUNCTIONS
#######m####################
#*************************************************
#DEPOLARISTAION TENSOR
#INPUTS: dimensions of the element of volume, volume
#OUTPUT: depolarisation tensor
#*************************************************
function depolarisation_tensor(lx,ly,lz,Vn)
    xx=2/pi*atan(1/lx^2*Vn/sqrt(lx^2+ly^2+lz^2))
    yy=2/pi*atan(1/ly^2*Vn/sqrt(lx^2+ly^2+lz^2))
    zz=2/pi*atan(1/lz^2*Vn/sqrt(lx^2+ly^2+lz^2))
    return[xx 0 0;0 yy 0;0 0 zz]
end
#*************************************************
#QUASISATIC POlARISABILITY
#INPUTS: epsilon, medium epsilon, depolarisation tensor, volume
#OUTPUT: polarisability
#*************************************************
function alpha_0(e,e_m,Ln,Vn)
    id=[1 0 0;0 1 0;0 0 1]
    Lni=inv(Ln)
    return (e*id-e_m*id)*inv((e*id-e_m*id)+Lni*e_m)*Lni*Vn
end
#*************************************************
#POlARISABILITY WITH RADIATIVE CORRECTION
#INPUTS: alpha_0 tensor, k
#OUTPUT: polarisability
#*************************************************
function alpha_radiative(a0,knorm)
    id=[1 0 0;0 1 0;0 0 1]
    return inv(inv(a0)-im*(knorm^3)/(6*pi)*id)
end