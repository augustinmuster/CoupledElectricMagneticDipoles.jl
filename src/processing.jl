module PostProcessing
###########################
# IMPORTS
###########################
using Base
using LinearAlgebra
include("green_tensors_e_m.jl")
include("input_fields.jl")
include("alpha.jl")
###########################
# FUNCTIONS
###########################

@doc raw"""
    compute_dipole_moment(alpha,phi_inc)

Computes the dipole moment (magnetic or electric) of a dipole with polarizability `alpha` under an incident field `phi_inc`. `alpha` can be:
- a complex scalar
- a 1D complex array of size ``N``
- a ``3\times 3`` or ``6\times 6`` complex matrix.
- a 3D complex array of size ``N\times 3\times 3`` or ``N\times 6\times 6``
"""
function compute_dipole_moment(alpha,phi_inc)
    n=length(phi_inc[:,1])
    p=zeros(ComplexF64,n,length(phi_inc[1,:]))
    if ndims(alpha)==2 
        for i=1:n
            p[i,:]=alpha*phi_inc[i,:]
        end
    elseif ndims(alpha)==3
        for i=1:n
            p[i,:]=alpha[i,:,:]*phi_inc[i,:]
        end
    elseif ndims(alpha)==0
        for i=1:n
            p[i,:]=alpha*phi_inc[i,:]
        end
    elseif ndims(alpha)==1
        for i=1:n
            p[i,:]=alpha[i]*phi_inc[i,:]
        end
    end
    return p
end


@doc raw"""
    compute_cross_sections_e(knorm,kr,e_inc,alpha,input_field;explicit_scattering=true,verbose=true)

Computes the extinction, absorbtion ans scattering cross section ``\sigma_{ext}``, ``\sigma_{abs}``, ``\sigma_{sca}`` of a system made out of electric dipoles.
Notes that it sould follow the optical theorem, i.e.
```math
\sigma_{ext}=\sigma_{abs}+\sigma_{sca}
```

Inputs 
- `knorm`: wavenumber
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the incident fields ``E_{inc}`` on every dipole.
- `alpha_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `input_field`: 2D complex array of size ``N\times 3`` containing the input field ``E_0`` at each of the dipoles positions.
- `explicit_scattering`: boolean that says wether to compute scttering cross section explicitely (formula) or to deduce it from the optical theorem. By default set to `true`.
- `verbose`: whether to output pieces of information to the standard output during running or not. By default set to `true`.
Outputs
- an array containing in order: extinction, absorption and scattering cross section.
"""
function compute_cross_sections_e(knorm,kr,e_inc,alpha_dl,input_field;explicit_scattering=true,verbose=true)

    #**********saving results*******
    if verbose
        println("computing cross sections...")
    end
    #number of point dipoles
    n=length(kr[:,1])
    #factor for dipole moment conversion
    factor_p=4*pi/knorm^3
    #computation of the cross sections
    constant =(knorm)/dot(input_field[1,:],input_field[1,:])
    sumext=0.0
    sumabs=0.0
    sumsca=0.0
    #dispatch alpha
    if ndims(alpha_dl)==1 || ndims(alpha_dl)==0
        id=1
    else
        id=Matrix{ComplexF64}(I,3,3)
    end
    alpha_dl=Alphas.dispatch_e(alpha_dl,n)
    #compute_cross sections
    for j=1:n
        #extinction
        sumext=sumext+imag(dot(input_field[j,:],alpha_dl[j]*e_inc[j,:]))
        #absorption
        sumabs=sumabs-imag(dot(alpha_dl[j]*e_inc[j,:],(inv(factor_p*alpha_dl[j])+im*knorm^3/6/pi*id)*alpha_dl[j]*e_inc[j,:]))
        #scattering
    end
    if explicit_scattering
        for j=1:n
            sumsca=sumsca+dot(alpha_dl[j]*e_inc[j,:],(knorm/6/pi)*alpha_dl[j]*e_inc[j,:])
            for k=1:j-1
                G=imag(knorm/4/pi*GreenTensors.G_e_renorm(kr[j,:],kr[k,:]))
                sumsca=sumsca+dot(alpha_dl[j]*e_inc[j,:],G*alpha_dl[k]*e_inc[k,:])
                sumsca=sumsca+dot(alpha_dl[k]*e_inc[k,:],G*alpha_dl[j]*e_inc[j,:])
            end
        end
    end

    cext=constant*factor_p*sumext
    cabs=constant*factor_p^2*sumabs

    if explicit_scattering
        csca=constant*knorm^2*factor_p^2*sumsca
    else
        csca=cext-cabs
    end

    return [real(cext) real(cabs) real(csca)]
end

@doc raw"""
    compute_cross_sections_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field;explicit_scattering=true,verbose=true)

Computes the extinction, absorbtion ans scattering cross section ``\sigma_{ext}``, ``\sigma_{abs}``, ``\sigma_{sca}`` of a system made out of electric dipoles.
Notes that it sould follow the optical theorem, i.e.
```math
\sigma_{ext}=\sigma_{abs}+\sigma_{sca}
```

Inputs 
- `knorm`: wavenumber
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `e_inc`: 2D complex array of size ``N\times 6`` containing the incident fields ``E_{inc}`` on every dipole.
- `alpha_e_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarisability of each dipoles. See the Alphas module for accepted formats.
- `input_field`: input field on each dipole (``N\times 6``) array.
- `explicit_scattering`: boolean that says wether to compute scttering cross section explicitely (formula) or to deduce it from the optical theorem. By default set to `true`.
- `verbose`: whether to output pieces of information to the standard output during running or not. By default set to `true`.
Outputs
- an array containing in order: extinction, absorption and scattering cross section.
"""
function compute_cross_sections_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field;explicit_scattering=true,verbose=true)
    #redefine things
    e_inc=phi_inc[:,1:3]
    h_inc=phi_inc[:,4:6]
    p=compute_dipole_moment(alpha_e_dl,e_inc)
    m=compute_dipole_moment(alpha_m_dl,h_inc)
    e_inp=input_field[:,1:3]
    h_inp=input_field[:,4:6]
    if verbose
        println("computing cross sections...")
    end
    #number of dipoles
    n=length(kr[:,1])
    #define sum variables
    sum_sca=0.
    sum_ext=0.
    sum_abs=0.
    #compute cross sections
    for i=1:n
        sum_ext=sum_ext+imag(dot(e_inp[i,:],p[i,:])+dot(h_inp[i,:],m[i,:]))
        sum_abs=sum_abs+ (imag(dot(e_inc[i,:],p[i,:])) -2/3*dot(p[i,:],p[i,:]))+imag(dot(h_inc[i,:],m[i,:]))-2/3*dot(m[i,:],m[i,:])
        if (explicit_scattering)
            sum_sca=sum_sca+1/3*dot(p[i,:],p[i,:])+1/3*dot(m[i,:],m[i,:])
            for j=(i+1):n
                sum_sca=sum_sca+real(transpose(p[j,:])*(imag(GreenTensors.G_e_renorm(kr[j,:],kr[i,:]))*conj(p[i,:])) + transpose(m[j,:])*(imag(GreenTensors.G_e_renorm(kr[j,:],kr[i,:]))*conj(m[i,:])))
                sum_sca=sum_sca+imag(-transpose(conj(p[i,:]))*imag(GreenTensors.G_m_renorm(kr[i,:],kr[j,:]))*m[j,:]    +   transpose(conj(p[j,:]))*imag(GreenTensors.G_m_renorm(kr[i,:],kr[j,:]))*m[i,:])
            end
        end

    end
    cst=2*pi/knorm^2/dot(e_inp[1,1:3],e_inp[1,1:3])
    if (explicit_scattering)
        return [2*pi/knorm cst*sum_ext cst*sum_abs 2*cst*sum_sca]
    else
        return [2*pi/knorm cst*real(sum_ext) cst*real(sum_abs) real(sum_ext-sum_abs)]
    end

end

@doc raw"""
    poynting_vector(phi)
Computes the poynting vector of a em field. Input is an electric and magnetic field `phi`(1D complex Array of length 6).
Outputs a 1D float array of length 3.
"""
function poynting_vector(phi)
    return 0.5*real(cross(phi[1:3],conj(phi[4:6])))
end

@doc raw"""
    diff_scattering_cross_section_e(knorm,kr,e_inc,alpha_e_dl,input_field,ur)

Computes the differential scattering cross section of a system made out of electric dipoles in directions `ur`.

Inputs 
- `knorm`: wavenumber
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the incident fields ``E_{inc}`` on every dipole.
- `alpha_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `input_field`: 2D complex array of size ``N\times 3`` containing the input field ``E_0`` at each of the dipoles positions.
- `krf`: 2D float array of size ``Nf\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the scattered field is calculated.
Outputs
- an array containing the differential cross section in every directions.
"""
function diff_scattering_cross_section_e(knorm,kr,e_inc,alpha_e_dl,input_field,ur)
    #if only one direction
    if ndims(ur)==1
        if norm(ur)!=1.
            ur=ur/norm(ur)
        end
        n=length(kr[:,1])
        max_norm=0
        for i=1:n
            if norm(kr[i,:])>max_norm
                max_norm=norm(kr[i,:])
            end
        end
        krf=knorm*ur*100*max_norm
        
        poynting=poynting_vector(far_field_sca_e(kr,e_inc,alpha_e_dl,krf))
        d_sigma=(100*max_norm)^2*dot(poynting,ur)/(dot(input_field[1,:],input_field[1,:])/2)
        return real(d_sigma)
    #if more than one, i.e. 2D array
    else
        nur=length(ur[:,1])
        for i=1:nur
            if norm(ur[i,:])!=1.
                ur[i,:]=ur[i,:]/norm(ur[i,:])
            end
        end
        n=length(kr[:,1])
        max_norm=0
        for i=1:n
            if norm(kr[i,:])>max_norm
                max_norm=norm(kr[i,:])
            end
        end

        d_sigma=zeros(nur)
        for i=1:nur
            krf=knorm*ur[i,:]*100*max_norm
            poynting=poynting_vector(far_field_sca_e(kr,e_inc,alpha_e_dl,krf))
            d_sigma[i]=(100*max_norm)^2*dot(poynting,ur[i,:])/(dot(input_field[1,:],input_field[1,:])/2)
        end
        return d_sigma
    end
end

@doc raw"""
    diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field,ur))

Computes the differential scattering cross section of a system made out of electric dipoles in directions `ur`.

Inputs 
- `knorm`: wavenumber
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the incident fields ``E_{inc}`` on every dipole.
- `alpha_e_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `alpha_m_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `input_field`: 2D complex array of size ``N\times 3`` containing the input field ``E_0`` at each of the dipoles positions.
- `krf`: 2D float array of size ``Nf\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the scattered field is calculated.
Outputs
- an array containing the differential cross section in every directions.
"""
function diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field,ur)
    #if only one direction
    if ndims(ur)==1
        if norm(ur)!=1.
            ur=ur/norm(ur)
        end
        n=length(kr[:,1])
        max_norm=0
        for i=1:n
            if norm(kr[i,:])>max_norm
                max_norm=norm(kr[i,:])
            end
        end
        krf=knorm*ur*100*max_norm
        
        poynting=poynting_vector(far_field_sca_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,krf))
        d_sigma=(100*max_norm)^2*dot(poynting,ur)/(dot(input_field[1,:],input_field[1,:])/2)
        return real(d_sigma)
    #if more than one, i.e. 2D array
    else
        nur=length(ur[:,1])
        for i=1:nur
            if norm(ur[i,:])!=1.
                ur[i,:]=ur[i,:]/norm(ur[i,:])
            end
        end
        n=length(kr[:,1])
        max_norm=0
        for i=1:n
            if norm(kr[i,:])>max_norm
                max_norm=norm(kr[i,:])
            end
        end

        d_sigma=zeros(nur)
        for i=1:nur
            krf=knorm*ur[i,:]*100*max_norm
            poynting=poynting_vector(far_field_sca_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,krf))
            d_sigma[i]=real((100*max_norm)^2*dot(poynting,ur[i,:])/(dot(input_field[1,:],input_field[1,:])/2))
        end
        return d_sigma
    end
end

@doc raw"""
    diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field,ur))

Computes the differential scattering cross section of a system made out of electric dipoles in directions `ur`.

Inputs 
- `knorm`: wavenumber
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the incident fields ``E_{inc}`` on every dipole.
- `alpha_e_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `alpha_m_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `input_field`: 2D complex array of size ``N\times 3`` containing the input field ``E_0`` at each of the dipoles positions.
- `krf`: 2D float array of size ``Nf\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the scattered field is calculated.
Outputs
- an array containing the differential cross section in every directions.
"""
function diff_total_emitted_power_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,krf,phi_krf)
    #if only one direction
    if ndims(krf)==1
        poynting=poynting_vector(far_field_sca_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,krf).+phi_krf)
        pow=real((norm(krf)/knorm)^2*dot(poynting,krf/norm(krf)))
        return real(pow)
    #if more than one, i.e. 2D array
    else
        nur=length(krf[:,1])
        pow=zeros(nur)
        for i=1:nur
            poynting=poynting_vector(far_field_sca_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,krf[i,:]).+phi_krf[i,:])
            pow[i]=real(norm((krf[i,:])/knorm)^2*dot(poynting,krf[i,:]/norm(krf[i,:])))
        end
        return real(pow)
    end
end

@doc raw"""
    field_sca_e_m(kr, alpha_e_dl, alpha_m_dl, e_inc, krf)
It computes the scattered field from the ensemble of dipoles.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarisability of each dipoles. See the Alphas module for accepted formats.
- `e_inc`: 2D complex array of size ``N\times 6`` with the incident field ``\phi_{inc}=(E_inc,H_inc)`` on the dipoles.
- `krf`: 2D float array of size ``Nf\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the scattered field is calculated.

#Outputs
- `field_r`: 2D complex array of size ``Nf\times 6`` with the field scattered by the dipoles at every ``k\vec{r_f}``.

Equation

```math
\mathbf{E}_{sca}(\mathbf{r}) = k^2G(\mathbf{r},\mathbf{\bar{r}}_N) \alpha(\mathbf{\bar{r}}_N) \mathbf{E}_{inc}(\mathbf{\bar{r}}_N) = k^2 G(\mathbf{r},\mathbf{\bar{r}}_N) \alpha(\mathbf{\bar{r}}_N) D(\mathbf{\bar{r}}_N) \E_{0}
```

\r = `r0`
\bar{r}}_N = `pos`
\alphagg(\mathbf{\bar{r}}_N) = `alpha`
\GG(\r,\mathbf{\bar{r}}_N) = `G_tensor`
\E_{inc}(\mathbf{\bar{r}}_N) = `E_inc`

\E_{sca}(\r) = `field_r`
"""
function field_sca_e_m(kr, alpha_e_dl, alpha_m_dl, e_inc, krf)

    n_particles = length(kr[:,1]) 
    n_r0 = length(krf[:,1]) 

    alp_e, alp_m = Alphas.dispatch_e_m(alpha_e_dl,alpha_m_dl,n_particles)

    G_tensor_fr = zeros(ComplexF64,n_r0*6,n_particles*6)
    for i = 1:n_particles
        for j = 1:n_r0
            Ge, Gm = GreenTensors.G_em_renorm(krf[j,:],kr[i,:])   
            G_tensor_fr[6 * (j-1) + 1:6 * (j-1) + 6 , 6 * (i-1) + 1:6 * (i-1) + 6] = [Ge*alp_e[i] im*Gm*alp_m[i]; -im*Gm*alp_e[i] Ge*alp_m[i]]
	    end
    end
    e_inc = reshape(transpose(e_inc),n_particles*6,)
    field_r = G_tensor_fr*e_inc
    return reshape(field_r,6,n_r0)   
end

@doc raw"""
    field_sca_e_m(kr, alpha_dl, e_inc, krf)
It computes the scattered field from the ensemble of dipoles.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_dl`: complex dimensionless electric and magnetic polarisability of each dipoles. See the Alphas module for accepted formats.
- `e_inc`: 2D complex array of size ``N\times 6`` with the incident field ``E_{inc}`` on the dipoles.
- `krf`: 2D float array of size ``Nf\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the scattered field is calculated.

#Outputs
- `field_r`: 2D complex array of size ``Nf\times 6`` with the field scattered by the dipoles at every ``k\vec{r_f}``.
"""
function field_sca_e_m(kr, alpha_dl, e_inc, krf)

    n_particles = length(kr[:,1]) 
    n_r0 = length(krf[:,1]) 

    alp = Alphas.dispatch_e_m(alpha_dl,n_particles)

    G_tensor_fr = zeros(ComplexF64,n_r0*6,n_particles*6)
    for i = 1:n_particles
        for j = 1:n_r0
            Ge, Gm = GreenTensors.G_em_renorm(krf[j,:],kr[i,:])   
            G_tensor_fr[6 * (j-1) + 1:6 * (j-1) + 6 , 6 * (i-1) + 1:6 * (i-1) + 6] = [Ge im*Gm; -im*Gm Ge]*alp[i]
	    end
    end
    e_inc = reshape(transpose(e_inc),n_particles*6,)
    field_r = G_tensor_fr*e_inc
    return reshape(field_r,6,n_r0)     
end

@doc raw"""
    function field_sca_e(kr, alpha_e_dl, e_inc, krf)
It computes the scattered field from the ensemble of dipoles.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `e_inc`: 2D complex array of size ``N\times 3`` with the incident field ``E_{inc}`` in the dipoles.
- `krf`: 2D float array of size ``Nf\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the scattered field is calculated.

#Outputs
- `field_r`: 2D complex array of size ``Nf\times 3`` with the field scattered by the dipoles at every ``k\vec{r_f}``.

Equation

```math
\mathbf{E}_{sca}(\mathbf{r}) = k^2G(\mathbf{r},\mathbf{\bar{r}}_N) \alpha(\mathbf{\bar{r}}_N) \mathbf{E}_{inc}(\mathbf{\bar{r}}_N) = k^2 G(\mathbf{r},\mathbf{\bar{r}}_N) \alpha(\mathbf{\bar{r}}_N) D(\mathbf{\bar{r}}_N) \E_{0}
```
"""
function field_sca_e(kr, alpha_e_dl, e_inc, krf)

    n_particles = length(kr[:,1]) 
    n_r0 = length(krf[:,1]) 

    alp_e = Alphas.dispatch_e(alpha_e_dl,n_particles)

    G_tensor_fr = zeros(ComplexF64,n_r0*3,n_particles*3)
    for i = 1:n_particles
        for j = 1:n_r0
            Ge = GreenTensors.G_e_renorm(krf[j,:],kr[i,:])   
            G_tensor_fr[3 * (j-1) + 1:3 * (j-1) + 3 , 3 * (i-1) + 1:3 * (i-1) + 3] = Ge*alp_e[i] 
	    end
    end
    e_inc = reshape(transpose(e_inc),n_particles*3,)
    field_r = G_tensor_fr*e_inc
    return reshape(field_r,n_r0,3)      
end

@doc raw"""
    function far_field_sca_e(kr,e_inc,alpha_e_dl,krf)
It computes the scattered electric and magnetic field from the ensemble of dipoles in the far field approximation.

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `e_inc`: 2D complex array of size ``N\times 3`` with the incident field ``E_{inc}`` in the dipoles.
- `krf`: 2D float array of size ``Nf\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the scattered field is calculated.

# Outputs
- `field_r`: 2D complex array of size ``Nf\times 6`` with the electric and magnetic field scattered by the dipoles at every ``k\vec{r_f}``.

"""
function far_field_sca_e(kr,e_inc,alpha_e_dl,krf)
    #number of dipoles
    n=length(kr[:,1])
    #dispatch alpha
    alpha_e_dl=Alphas.dispatch_e(alpha_e_dl,n)
    #if only one position
    if ndims(krf)==1
        res=zeros(ComplexF64,6)
        for i=1:n
            Ge,Gm=GreenTensors.G_em_far_field_renorm(krf,kr[i,:])
            res[1:3]=res[1:3]+Ge*alpha_e_dl[i]*e_inc[i,:]
            res[4:6]=res[4:6]-im*Gm*alpha_e_dl[i]*e_inc[i,:]
        end
        return res
    #if more than 1, i.e. 2D array
    else
        nf=length(krf[i,:])
        res=zeros(nf,6)
        for j=1:nf
            for i=1:n
                Ge,Gm=GreenTesors.G_em_far_field_renorm(krf[j,:],kr[i,:])
                res[j,1:3]=res[j,1:3]+Ge*alpha_e_dl[i]*e_inc[i,:]
                res[j,4:6]=res[j,4:6]-im*Gm*alpha_e_dl[i]*e_inc[i,:]
            end
        end
        return res
    end
end

@doc raw"""
    function far_field_sca_e_m(kr,e_inc,alpha_e_dl,alpha_m_dl,krf)
It computes the scattered electric and magnetic field from the ensemble of dipoles in the far field approximation.

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `alpha_m_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `e_inc`: 2D complex array of size ``N\times 3`` with the incident field ``E_{inc}`` in the dipoles.
- `krf`: 2D float array of size ``Nf\times 3`` containing the dimentionless positions ``k\vec{r_f}`` where the scattered field is calculated.

# Outputs
- `field_r`: 2D complex array of size ``Nf\times 6`` with the electric and magnetic field scattered by the dipoles at every ``k\vec{r_f}``.

"""
function far_field_sca_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,krf)
    #number of dipoles
    n=length(kr[:,1])
    #dispatch alpha
    alpha_e_dl=Alphas.dispatch_e(alpha_e_dl,n)
    #if only one position
    if ndims(krf)==1
        res=zeros(ComplexF64,6)
        for i=1:n
            Ge,Gm=GreenTensors.G_em_far_field_renorm(krf,kr[i,:])
            res[1:3]=res[1:3]+Ge*alpha_e_dl[i]*phi_inc[i,1:3]+im*Gm*alpha_m_dl[i]*phi_inc[i,4:6]
            res[4:6]=res[4:6]-im*Gm*alpha_e_dl[i]*phi_inc[i,1:3]+Ge*alpha_m_dl[i]*phi_inc[i,4:6]
        end
        return res
    #if more than 1, i.e. 2D array
    else
        nf=length(krf[i,:])
        res=zeros(nf,6)
        for j=1:nf
            for i=1:n
                Ge,Gm=GreenTesors.G_em_far_field_renorm(krf[j,:],kr[i,:])
                res[j,1:3]=res[j,1:3]+Ge*alpha_e_dl[i]*phi_inc[i,1:3]+im*Gm*alpha_m_dl[i]*phi_inc[i,4:6]
                res[j,4:6]=res[j,4:6]-im*Gm*alpha_e_dl[i]*e_inc[i,1:3]+Ge*alpha_m_dl[i]*phi_inc[i,4:6]
            end
        end
        return res
    end
end

@doc raw"""
    ldos_e_m(kr, alpha_e_dl, alpha_m_dl, Ainv, krd; dip=nothing)
It Computes local density of states (LDOS) by the imaginary part of the returning field.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarisability of each dipoles. See the Alphas module for accepted formats.
- `Ainv`: (inverse) DDA matrix ``[I - G*alpha]^(-1)``.
- `krd`: 2D float array of size ``Nd\times 3`` containing the dimentionless positions ``k\vec{r_d}`` where the LDOS is calculated.
- `dip`: integer defining the dipole moment (``dip = 1`` is an electric x-dipole, ``dip = 2`` an elctric y-dipole...) or float array of size 6 with the desired dipole moment of the dipole.  
#Outputs
- `LDOS`: float array with the LDOS.

Equation

```math
\mathrm{Copy from documentation}
```
"""
function ldos_e_m(kr, alpha_e_dl, alpha_m_dl, Ainv, krd; dip=nothing)

    n_particles = length(kr[:,1])
    n_dpos = length(krd[:,1])

    G_tensor = zeros(ComplexF64,n_particles*6,6) 
    G_tensor_fr = zeros(ComplexF64,6,n_particles*6)

    alp_e, alp_m = Alphas.dispatch_e_m(alpha_e_dl,alpha_m_dl,n_particles)

    if dip === nothing
        LDOS = zeros(n_dpos,2)
        for j=1:n_dpos
            for i=1:n_particles
                Ge, Gm = GreenTensors.G_em_renorm(kr[i,:],krd[j,:])        
                G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [Ge im*Gm; -im*Gm Ge]
                G_tensor_fr[:, 6 * (i-1) + 1:6 * (i-1) + 6] = [Ge*alp_e[i] -im*Gm*alp_m[i]; im*Gm*alp_e[i] Ge*alp_m[i]]
            end
            G_ldos = G_tensor_fr*Ainv*G_tensor
            LDOS[j,1] = 1 + 1/3*imag(tr(G_ldos[1:3,1:3]))/(2/3)
            LDOS[j,2] = 1 + 1/3*imag(tr(G_ldos[4:6,4:6]))/(2/3) 
        end
    else
        LDOS = zeros(n_dpos)
        if length(dip) == 1  && dip < 7 && dip > 0
            dip_o = dip
            dip = zeros(6)
            dip[dip_o] = 1
        elseif length(dip) == 6
            dip = dip/norm(dip) # Ensure that its modulus is equal to one
        else
            dip = zeros(6)
            println("dip should be an integer (between 1 and 6) or a vector of length 6")
        end
        for j=1:n_dpos
            for i=1:n_particles
                Ge, Gm = GreenTensors.G_em_renorm(kr[i,:],krd[j,:])        
                G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [Ge im*Gm; -im*Gm Ge]
                G_tensor_fr[:, 6 * (i-1) + 1:6 * (i-1) + 6] = [Ge*alp_e[i] -im*Gm*alp_m[i]; im*Gm*alp_e[i] Ge*alp_m[i]]
            end
            field_r = G_tensor_fr*Ainv*G_tensor*dip
            LDOS[j] = 1 + imag(transpose(dip)*field_r)/(2/3)
        end
    end
    if length(LDOS) == 1
        LDOS = LDOS[1]
    end
    return LDOS
end

@doc raw"""
    ldos_e_m(kr, alpha_dl, Ainv, krd; dip=nothing)
It Computes local density of states (LDOS) by the imaginary part of the returning field.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_dl`: complex dimensionless electric and magnetic polarisability of each dipoles. See the Alphas module for accepted formats.
- `Ainv`: (inverse) DDA matrix ``[I - G*alpha]^(-1)``.
- `krd`: 2D float array of size ``Nd\times 3`` containing the dimentionless positions ``k\vec{r_d}`` where the LDOS is calculated.
- `dip`: integer defining the dipole moment (``dip = 1`` is an electric x-dipole, ``dip = 2`` an elctric y-dipole...) or float array of size 6 with the desired dipole moment of the dipole.  
#Outputs
- `LDOS`: float array with the LDOS.

Equation

```math
\mathrm{Copy from documentation}
```
"""
function ldos_e_m(kr, alpha_dl, Ainv, krd; dip=nothing)

    n_particles = length(kr[:,1])
    n_dpos = length(krd[:,1])

    G_tensor = zeros(ComplexF64,n_particles*6,6) 
    G_tensor_fr = zeros(ComplexF64,6,n_particles*6)

    alp = Alphas.dispatch_e_m(alpha_dl,n_particles)

    if dip === nothing
        LDOS = zeros(n_dpos,2)
        for j=1:n_dpos
            for i=1:n_particles
                Ge, Gm = GreenTensors.G_em_renorm(kr[i,:],krd[j,:])        
                G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [Ge im*Gm; -im*Gm Ge]
                G_tensor_fr[:, 6 * (i-1) + 1:6 * (i-1) + 6] = [Ge -im*Gm; im*Gm Ge]*alp[i]
            end
            G_ldos = G_tensor_fr*Ainv*G_tensor
            LDOS[j,1] = 1 + 1/3*imag(tr(G_ldos[1:3,1:3]))/(2/3)
            LDOS[j,2] = 1 + 1/3*imag(tr(G_ldos[4:6,4:6]))/(2/3) 
        end
    else
        LDOS = zeros(n_dpos)
        if length(dip) == 1  && dip < 7 && dip > 0
            dip_o = dip
            dip = zeros(6)
            dip[dip_o] = 1
        elseif length(dip) == 6
            dip = dip/norm(dip) # Ensure that its modulus is equal to one
        else
            dip = zeros(6)
            println("dip should be an integer (between 1 and 6) or a vector of length 6")
        end
        for j=1:n_dpos
            for i=1:n_particles
                Ge, Gm = GreenTensors.G_em_renorm(kr[i,:],krd[j,:])        
                G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [Ge im*Gm; -im*Gm Ge]
                G_tensor_fr[:, 6 * (i-1) + 1:6 * (i-1) + 6] = [Ge -im*Gm; im*Gm Ge]*alp[i]
            end
            field_r = G_tensor_fr*Ainv*G_tensor*dip
            LDOS[j] = 1 + imag(transpose(dip)*field_r)/(2/3)
        end
    end
    if length(LDOS) == 1
        LDOS = LDOS[1]
    end
    return LDOS
end

@doc raw"""
    ldos_e(kr, alpha_e_dl, Ainv, krd; dip=nothing)
It Computes local density of states (LDOS) by the imaginary part of the returning field.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarisability of each dipoles. See the Alphas module for accepted formats.
- `Ainv`: (inverse) DDA matrix ``[I - G*alpha]^(-1)``.
- `krd`: 2D float array of size ``Nd\times 3`` containing the dimentionless positions ``k\vec{r_d}`` where the LDOS is calculated.
- `dip`: integer defining the dipole moment (``dip = 1`` is an electric x-dipole, ``dip = 2`` an elctric y-dipole...) or float array of size 3 with the desired dipole moment of the dipole.  
#Outputs
- `LDOS`: float array with the LDOS.

Equation

```math
\mathrm{Copy from documentation}
```
"""
function ldos_e(kr, alpha_e_dl, Ainv, krd; dip=nothing)

    n_particles = length(kr[:,1])
    n_dpos = length(krd[:,1])

    G_tensor = zeros(ComplexF64,n_particles*3,3) 
    G_tensor_fr = zeros(ComplexF64,3,n_particles*3)

    alp_e = Alphas.dispatch_e(alpha_e_dl,n_particles)

    LDOS = zeros(n_dpos)
    if dip === nothing
        for j=1:n_dpos
            for i=1:n_particles
                Ge = GreenTensors.G_e_renorm(kr[i,:],krd[j,:])        
                G_tensor[3 * (i-1) + 1:3 * (i-1) + 3 , :] = Ge 
                G_tensor_fr[:, 3 * (i-1) + 1:3 * (i-1) + 3] = Ge*alp_e[i] 
            end
            G_ldos = G_tensor_fr*Ainv*G_tensor
            LDOS[j] = 1 + 1/3*imag(tr(G_ldos))/(2/3)
        end
    else
        if length(dip) == 1 && dip < 4 && dip > 0
            dip_o = dip
            dip = zeros(3)
            dip[dip_o] = 1
        elseif length(dip) == 3
            dip = dip/norm(dip) # Ensure that its modulus is equal to one
        else
            dip = zeros(3)
            println("dip should be an integer (between 1 and 3) or a vector of length 3")
        end
        for j=1:n_dpos
            for i=1:n_particles
                Ge = GreenTensors.G_e_renorm(kr[i,:],krd[j,:])        
                G_tensor[3 * (i-1) + 1:3 * (i-1) + 3 , :] = Ge
                G_tensor_fr[:, 3 * (i-1) + 1:3 * (i-1) + 3] = Ge*alp_e[i] 
            end
            field_r = G_tensor_fr*Ainv*G_tensor*dip
            LDOS[j] = 1 + imag(transpose(dip)*field_r)/(2/3)
        end
    end
    if length(LDOS) == 1
        LDOS = LDOS[1]
    end
    return LDOS
end


@doc raw"""
    force_e_m(k,kr,alpha_e_dl, alpha_m_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
It computes the optical forces for deterministics imputs fields.

#Arguments
- `k`: float with the wavevector.
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex array containing the dimensionless electric polarisability.
- `alpha_m_dl`: complex array containing the dimensionless magnetic polarisability.
- `Ainv`: (inverse) DDA matrix ``[I - G*alpha]^(-1)``.
- `e_0`: 2D complex array of size ``N\times 6`` containing the external imput field.
- `dxe_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*x`` argument of the external imput field.
- `dye_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*y`` argument of the external imput field.
- `dze_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*z`` argument of the external imput field.
#Outputs
- `real(fx)`: float array of Size ``N`` with the value of the force along the ``x``-axis at each dipole.
- `real(fy)`: float array of Size ``N`` with the value of the force along the ``y``-axis at each dipole.
- `real(fz)`: float array of Size ``N`` with the value of the force along the ``z``-axis at each dipole.
"""
function force_e_m(k,kr,alpha_e_dl, alpha_m_dl, Ainv, e_0, dxe_0, dye_0, dze_0)

    eps0 = 1/2*k*8.8541878128e-12

    n_particles = length(kr[:,1])
    n_e_0 = length(e_0[:,1])
    n_dxe_0 = length(dxe_0[:,1])
    n_dye_0 = length(dye_0[:,1])
    n_dze_0 = length(dze_0[:,1])

    if n_particles != n_e_0 || n_particles != n_dxe_0 || n_e_0 != n_dxe_0 || n_e_0 != n_dye_0 || n_e_0 != n_dze_0
        println("the dimension of the filed must coincided with the number of particles")
        return 0
    end

    e_0 = reshape(transpose(e_0),n_particles*6,)
    dxe_0 = reshape(transpose(dxe_0),n_particles*6,)
    dye_0 = reshape(transpose(dye_0),n_particles*6,)
    dze_0 = reshape(transpose(dze_0),n_particles*6,)
    e_inc = Ainv*e_0
    
    dxG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    dyG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    dzG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    p = zeros(ComplexF64,n_particles*6,)

    alpha_e_dl,alpha_m_dl=Alphas.dispatch_e_m(alpha_e_dl,alpha_m_dl,n_particles)

    for i=1:n_particles
        p[6*(i-1)+1:6*(i-1)+6] = [alpha_e_dl[i]*e_inc[6*(i-1)+1:6*(i-1)+3];alpha_m_dl[i]*e_inc[6*(i-1)+4:6*(i-1)+6]]
        for j=1:i-1
            dxGe, dxGm = GreenTensors.dxG_em_renorm(kr[i,:],kr[j,:])
            dxG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dxGe*alpha_e_dl[j] im*dxGm*alpha_m_dl[j]; -im*dxGm*alpha_e_dl[j] dxGe*alpha_m_dl[j]]
            dxG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dxGe*alpha_e_dl[i] -im*dxGm*alpha_m_dl[i]; im*dxGm*alpha_e_dl[i] dxGe*alpha_m_dl[i]]
            dyGe, dyGm = GreenTensors.dyG_em_renorm(kr[i,:],kr[j,:])
            dyG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dyGe*alpha_e_dl[j] im*dyGm*alpha_m_dl[j]; -im*dyGm*alpha_e_dl[j] dyGe*alpha_m_dl[j]]
            dyG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dyGe*alpha_e_dl[i] -im*dyGm*alpha_m_dl[i]; im*dyGm*alpha_e_dl[i] dyGe*alpha_m_dl[i]]
            dzGe, dzGm = GreenTensors.dzG_em_renorm(kr[i,:],kr[j,:])
            dzG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dzGe*alpha_e_dl[j] im*dzGm*alpha_m_dl[j]; -im*dzGm*alpha_e_dl[j] dzGe*alpha_m_dl[j]]
            dzG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dzGe*alpha_e_dl[i] -im*dzGm*alpha_m_dl[i]; im*dzGm*alpha_e_dl[i] dzGe*alpha_m_dl[i]]
        end
    end

    dxe_inc = dxe_0 + dxG_alp*e_inc
    dye_inc = dye_0 + dyG_alp*e_inc
    dze_inc = dze_0 + dzG_alp*e_inc

    p = conj(transpose(reshape(p,6,n_particles)))
    dxe_inc = transpose(reshape(dxe_inc,6,n_particles))
    dye_inc = transpose(reshape(dye_inc,6,n_particles))
    dze_inc = transpose(reshape(dze_inc,6,n_particles))

    fx = eps0*sum(p.*dxe_inc,dims=2)
    fy = eps0*sum(p.*dye_inc,dims=2)
    fz = eps0*sum(p.*dze_inc,dims=2)
    if length(fz) == 1
        return real(fx[1]), real(fy[1]), real(fz[1])
    end
    return real(fx[:,1]), real(fy[:,1]), real(fz[:,1])
end

@doc raw"""
    force_e_m(k,kr,alpha_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
It computes the optical forces for deterministics imputs fields.

#Arguments
- `k`: float with the wavevector.
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_dl`: complex array containing the dimensionless polarisability.
- `Ainv`: (inverse) DDA matrix ``[I - G*alpha]^(-1)``.
- `e_0`: 2D complex array of size ``N\times 6`` containing the external imput field.
- `dxe_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*x`` argument of the external imput field.
- `dye_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*y`` argument of the external imput field.
- `dze_0`: 2D complex array of size ``N\times 6`` containing the derivative respect the ``k*z`` argument of the external imput field.
#Outputs
- `real(fx)`: float array of Size ``N`` with the value of the force along the ``x``-axis at each dipole.
- `real(fy)`: float array of Size ``N`` with the value of the force along the ``y``-axis at each dipole.
- `real(fz)`: float array of Size ``N`` with the value of the force along the ``z``-axis at each dipole.
"""
function force_e_m(k,kr,alpha_dl, Ainv, e_0, dxe_0, dye_0, dze_0)

    eps0 = 1/2*k*8.8541878128e-12

    n_particles = length(kr[:,1])
    n_e_0 = length(e_0[:,1])
    n_dxe_0 = length(dxe_0[:,1])
    n_dye_0 = length(dye_0[:,1])
    n_dze_0 = length(dze_0[:,1])

    if n_particles != n_e_0 || n_particles != n_dxe_0 || n_e_0 != n_dxe_0 || n_e_0 != n_dye_0 || n_e_0 != n_dze_0
        println("the dimension of the filed must coincided with the number of particles")
        return 0
    end

    e_0 = reshape(transpose(e_0),n_particles*6,)
    dxe_0 = reshape(transpose(dxe_0),n_particles*6,)
    dye_0 = reshape(transpose(dye_0),n_particles*6,)
    dze_0 = reshape(transpose(dze_0),n_particles*6,)
    e_inc = Ainv*e_0
    
    dxG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    dyG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    dzG_alp = zeros(ComplexF64,n_particles*6,n_particles*6)
    p = zeros(ComplexF64,n_particles*6,)

    alpha_dl=Alphas.dispatch_e_m(alpha_dl,n_particles)

    for i=1:n_particles
        p[6*(i-1)+1:6*(i-1)+6] = alpha_dl[i]*e_inc[6*(i-1)+1:6*(i-1)+6]
        for j=1:i-1
            dxGe, dxGm = GreenTensors.dxG_em_renorm(kr[i,:],kr[j,:])
            dxG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dxGe im*dxGm; -im*dxGm dxGe]*alpha_dl[j]
            dxG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dxGe -im*dxGm; im*dxGm dxGe]*alpha_dl[i]
            dyGe, dyGm = GreenTensors.dyG_em_renorm(kr[i,:],kr[j,:])
            dyG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dyGe im*dyGm; -im*dyGm dyGe]*alpha_dl[j]
            dyG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dyGe -im*dyGm; im*dyGm dyGe]*alpha_dl[i]
            dzGe, dzGm = GreenTensors.dzG_em_renorm(kr[i,:],kr[j,:])
            dzG_alp[6*(i-1)+1:6*(i-1)+6,6*(j-1)+1:6*(j-1)+6]=[dzGe im*dzGm; -im*dzGm dzGe]*alpha_dl[j]
            dzG_alp[6*(j-1)+1:6*(j-1)+6,6*(i-1)+1:6*(i-1)+6]=-[dzGe -im*dzGm; im*dzGm dzGe]*alpha_dl[i]
        end
    end

    dxe_inc = dxe_0 + dxG_alp*e_inc
    dye_inc = dye_0 + dyG_alp*e_inc
    dze_inc = dze_0 + dzG_alp*e_inc

    p = conj(transpose(reshape(p,6,n_particles)))
    dxe_inc = transpose(reshape(dxe_inc,6,n_particles))
    dye_inc = transpose(reshape(dye_inc,6,n_particles))
    dze_inc = transpose(reshape(dze_inc,6,n_particles))

    fx = eps0*sum(p.*dxe_inc,dims=2)
    fy = eps0*sum(p.*dye_inc,dims=2)
    fz = eps0*sum(p.*dze_inc,dims=2)
    if length(fz) == 1
        return real(fx[1]), real(fy[1]), real(fz[1])
    end
    return real(fx[:,1]), real(fy[:,1]), real(fz[:,1])
end

@doc raw"""
    force_e(k,kr,alpha_e_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
It computes the optical forces for deterministics imputs fields.

#Arguments
- `k`: float with the wavevector.
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex array containing the dimensionless electric polarisability.
- `Ainv`: (inverse) DDA matrix ``[I - G*alpha]^(-1)``.
- `e_0`: 2D complex array of size ``N\times 3`` containing the external imput field.
- `dxe_0`: 2D complex array of size ``N\times 3`` containing the derivative respect the ``k*x`` argument of the external imput field.
- `dye_0`: 2D complex array of size ``N\times 3`` containing the derivative respect the ``k*y`` argument of the external imput field.
- `dze_0`: 2D complex array of size ``N\times 3`` containing the derivative respect the ``k*z`` argument of the external imput field.
#Outputs
- `real(fx)`: float array of Size ``N`` with the value of the force along the ``x``-axis at each dipole.
- `real(fy)`: float array of Size ``N`` with the value of the force along the ``y``-axis at each dipole.
- `real(fz)`: float array of Size ``N`` with the value of the force along the ``z``-axis at each dipole.
"""
function force_e(k,kr,alpha_e_dl, Ainv, e_0, dxe_0, dye_0, dze_0)

    eps0 = 1/2*k*8.8541878128e-12

    n_particles = length(kr[:,1])
    n_e_0 = length(e_0[:,1])
    n_dxe_0 = length(dxe_0[:,1])
    n_dye_0 = length(dye_0[:,1])
    n_dze_0 = length(dze_0[:,1])

    if n_particles != n_e_0 || n_particles != n_dxe_0 || n_e_0 != n_dxe_0 || n_e_0 != n_dye_0 || n_e_0 != n_dze_0
        println("the dimension of the filed must coincided with the number of particles")
        return 0
    end

    e_0 = reshape(transpose(e_0),n_particles*3,)
    dxe_0 = reshape(transpose(dxe_0),n_particles*3,)
    dye_0 = reshape(transpose(dye_0),n_particles*3,)
    dze_0 = reshape(transpose(dze_0),n_particles*3,)
    e_inc = Ainv*e_0
    
    dxG_alp = zeros(ComplexF64,n_particles*3,n_particles*3)
    dyG_alp = zeros(ComplexF64,n_particles*3,n_particles*3)
    dzG_alp = zeros(ComplexF64,n_particles*3,n_particles*3)
    p = zeros(ComplexF64,n_particles*3,)

    alpha_e_dl = Alphas.dispatch_e(alpha_e_dl,n_particles)

    for i=1:n_particles
        p[3*(i-1)+1:3*(i-1)+3] = alpha_e_dl[i]*e_inc[3*(i-1)+1:3*(i-1)+3]
        for j=1:i-1
            dxGe = GreenTensors.dxG_e_renorm(kr[i,:],kr[j,:])
            dxG_alp[3*(i-1)+1:3*(i-1)+3,3*(j-1)+1:3*(j-1)+3]=dxGe*alpha_e_dl[j]
            dxG_alp[3*(j-1)+1:3*(j-1)+3,3*(i-1)+1:3*(i-1)+3]=-dxGe*alpha_e_dl[i]
            dyGe = GreenTensors.dyG_e_renorm(kr[i,:],kr[j,:])
            dyG_alp[3*(i-1)+1:3*(i-1)+3,3*(j-1)+1:3*(j-1)+3]=dyGe*alpha_e_dl[j]
            dyG_alp[3*(j-1)+1:3*(j-1)+3,3*(i-1)+1:3*(i-1)+3]=-dyGe*alpha_e_dl[i]
            dzGe = GreenTensors.dzG_e_renorm(kr[i,:],kr[j,:])
            dzG_alp[3*(i-1)+1:3*(i-1)+3,3*(j-1)+1:3*(j-1)+3]=dzGe*alpha_e_dl[j]
            dzG_alp[3*(j-1)+1:3*(j-1)+3,3*(i-1)+1:3*(i-1)+3]=-dzGe*alpha_e_dl[i]
        end
    end

    dxe_inc = dxe_0 + dxG_alp*e_inc
    dye_inc = dye_0 + dyG_alp*e_inc
    dze_inc = dze_0 + dzG_alp*e_inc

    p = conj(transpose(reshape(p,3,n_particles)))
    dxe_inc = transpose(reshape(dxe_inc,3,n_particles))
    dye_inc = transpose(reshape(dye_inc,3,n_particles))
    dze_inc = transpose(reshape(dze_inc,3,n_particles))

    fx = eps0*sum(p.*dxe_inc,dims=2)
    fy = eps0*sum(p.*dye_inc,dims=2)
    fz = eps0*sum(p.*dze_inc,dims=2)
    if length(fz) == 1
        return real(fx[1]), real(fy[1]), real(fz[1])
    end
    return real(fx[:,1]), real(fy[:,1]), real(fz[:,1])
end

@doc raw"""
    LDOS_sc(knorm, alpha, Ainv, pos, rd, dip_o)
It Computes partial local density of states (LDOS) by the scattering cross section (sc)

Imputs
- `knorm` = wavenumber
- `alpha` = polarizability of the particles (6N x 6N matrix, where -N- is the number of dipoles)
- `Ainv` = (inverse) DDA matrix (6N x 6N matrix, [I - k^2*G*alpha]^(-1))
- `pos` = position of the dipoles (N x 3 matrix, where -N- is the number of points)
- `rd` = emitting dipole position, i. e., position where the LDOS is calculated
- `dip_o` defined the nature of the dipole (see -point_dipole- function). Therefore, it defines the component of the LDOS that is calculated 

Outputs
- `LDOS` is a scalar with the value of the partial LDOS

Equation
```math
LDOS = \sigma^{sca}/\sigma^{sca}_{0}
```
"""
function LDOS_sc(knorm, alpha, Ainv, pos, rd, dip_o)

    r = [pos; rd]
    Np = length(r[:,1])
      
    E_0i = InputFields.point_dipole(knorm, 1, pos, rd, dip_o)
    
    if length(dip_o) == 1
        dip_oi = dip_o
        dip_o = zeros(6,1)
        dip_o[dip_oi] = 1
    else
        dip_o = dip_o/norm(dip_o) # Ensure that its modulus is equal to one
    end
    
    Pd = alpha*Ainv*E_0i
    Pd = [Pd; dip_o]
    Pd = reshape(Pd,6,Np)
    
    sum_sca = 0.
    
    for i=1:Np
        sum_sca = sum_sca + 1/3*dot(Pd[1:3,i],Pd[1:3,i]) + 1/3*dot(Pd[4:6,i],Pd[4:6,i])
        for j=(i+1):Np
            sum_sca=sum_sca+real(transpose(Pd[1:3,j])*(imag(GreenTensors.G_e_renorm(knorm*r[j,:],knorm*r[i,:]))*conj(Pd[1:3,i])) + transpose(Pd[4:6,j])*(imag(GreenTensors.G_e_renorm(knorm*r[j,:],knorm*r[i,:]))*conj(Pd[4:6,i])))
            sum_sca=sum_sca+imag(-transpose(conj(Pd[1:3,i]))*imag(GreenTensors.G_m_renorm(knorm*r[i,:],knorm*r[j,:]))*Pd[4:6,j]    +   transpose(conj(Pd[1:3,j]))*imag(GreenTensors.G_m_renorm(knorm*r[i,:],knorm*r[j,:]))*Pd[4:6,i])
        end
    end
    
    
    LDOS = real(sum_sca)*3#/(1/3*sum(abs.(dip_o).^2)) # Imaginary part of the dipole component (LDOS)

    return LDOS

end


end
