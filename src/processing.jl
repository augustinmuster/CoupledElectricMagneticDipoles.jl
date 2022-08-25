###########################
# IMPORTS
###########################
using Base
using LinearAlgebra
include("green_tensors_e_m.jl")
include("input_fields.jl")
###########################
# FUNCTIONS
###########################
#*************************************************
#COMPUTE THE CROSS SECTIONS FOR PLANE_WAVE INPUT FIELD
#INPUTS:  norm of the wave vector, polarisations, incident fields, quasistatic polarisabilities,e0,whether to compute explicitely csca, verbose
#OUTPUT: array with lambda, Cabs, Csca, Cext
#*************************************************
function compute_cross_sections(knorm,r,p,e_inc,alpha0;e0=[1,0,0],explicit_scattering=true,verbose=true)

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
                    sumsca=sumsca+dot(p[j,:],imag(G_e(r[j,:],r[k,:],knorm))*p[k,:])
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

#*************************************************
#COMPUTE THE POWERS FOR PLANE_WAVE INPUT FIELD AND ELECTRIC AND MAGNETIC DIPOLES
#INPUTS:  norm of the wave vector, polarisations, incident fields, quasistatic polarisabilities,e0,whether to compute explicitely csca, verbose
#OUTPUT: array with lambda, Cabs, Csca, Cext
#*************************************************
function compute_cross_sections_e_m(knorm,r,p,m,e_inc,h_inc,e_inp,h_inp,alpha_e,alpha_m;e0=[1,0,0],explicit_scattering=true,verbose=true)

    #**********saving results*******
    if verbose
        println("computing cross sections")
    end
    #number of dipoles
    n=length(r[:,1])
    #define sum variables
    sum_sca=0.
    sum_ext=0.
    sum_abs=0.
    term_sca=0.

    id=[1 0 0;0 1 0;0 0 1]

    for i=1:n
        sum_ext=sum_ext+imag(alpha_e[i,1,1]*dot(e_inp[i,:],e_inc[i,:])+alpha_m[i,1,1]*dot(h_inp[i,:],h_inc[i,:]))
        sum_abs=sum_abs+ ( imag(alpha_e[i,1,1]) -2/3*abs2(alpha_e[i,1,1]))*dot(e_inc[i,:],e_inc[i,:])+(imag(alpha_m[i,1,1])-2/3*abs2(alpha_m[i,1,1]))*dot(h_inc[i,:],h_inc[i,:])
        if (explicit_scattering)
            sum_sca=sum_sca+1/3*dot(p[i,:],p[i,:])+1/3*dot(m[i,:],m[i,:])
            for j=(i+1):n
                sum_sca=sum_sca+real(transpose(p[j,:])*(imag(G_e_renorm(knorm*r[j,:],knorm*r[i,:]))*conj(p[i,:])) + transpose(m[j,:])*(imag(G_e_renorm(knorm*r[j,:],knorm*r[i,:]))*conj(m[i,:])))
                sum_sca=sum_sca+imag(transpose(conj(p[i,:]))*imag(G_m_renorm(knorm*r[i,:],knorm*r[j,:]))*m[j,:]    -   transpose(conj(p[j,:]))*imag(G_m_renorm(knorm*r[i,:],knorm*r[j,:]))*m[i,:])
            end
        end

    end
    cst=2*pi/knorm^2*sqrt(3.5)
    if (explicit_scattering)
        return [2*pi/knorm cst*sum_ext cst*sum_abs 2*cst*sum_sca]
    else
        return [2*pi/knorm cst*real(sum_ext) cst*real(sum_abs) real(sum_ext-sum_abs)]
    end

end