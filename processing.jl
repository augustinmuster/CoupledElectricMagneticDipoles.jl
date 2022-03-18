###########################
# IMPORTS
###########################
using Base
using LinearAlgebra
###########################
# FUNCTIONS
###########################
#*************************************************
#COMPUTE THE CROSS SECTIONS FOR PLANE_WAVE INPUT FIELD
#INPUTS:  norm of the wave vector, polarisations, incident fields, quasistatic polarisabilities,e0,whether to compute explicitely csca, verbose
#OUTPUT: array with lambda, Cabs, Csca, Cext
#*************************************************
function compute_cross_sections(knorm,p,e_inc,alpha0,r;e0=[1,0,0],explicit_scattering=true,verbose=true)

    #**********saving results*******
    if verbose
        println("computing cross sections")
    end
    #number of point dipoles
    n=length(p[:,1])
    #computation of the cross sections
    constant =(knorm)/dot(e0,e0)
    sumext=0.0
    sumabs=0.0
    sumsca=0.0
    for j=1:n
        #extinction
        sumext=sumext+imag(dot(e_inc[j,:],p[j,:]))
        #absorption
        sumabs=sumabs-imag(dot(p[j,:],inv(alpha0[j,:,:])*p[j,:]))
        #scattering
        if explicit_scattering
            for k=1:n
                if k!=j
                    sumsca=sumsca+dot(p[j,:],imag(green(r[j,:],r[k,:],knorm))*p[k,:])
                else
                    sumsca=sumsca+dot(p[j,:],(knorm/6/pi)*p[k,:])
                end
            end
        end
    end


    cext=constant*sumext
    cabs=constant*sumabs

    if explicit_scattering
        csca=constant*knorm^2*sumsca
    else
        csca=cext-cabs
    end

    return [2*pi/knorm real(cext) real(cabs) real(csca)]
end