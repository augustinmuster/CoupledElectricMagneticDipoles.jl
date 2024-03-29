module PostProcessing

export compute_dipole_moment, compute_cross_sections_e, compute_cross_sections_e_m, poynting_vector, diff_scattering_cross_section_e, diff_scattering_cross_section_e_m, diff_emitted_power_e, diff_emitted_power_e,field_sca_e, field_sca_e_m,  far_field_sca_e, far_field_sca_e_m, ldos_e, ldos_e_m
###########################
# IMPORTS
###########################
using Base
using LinearAlgebra
using ..GreenTensors
using ..InputFields
using ..Alphas
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
    compute_cross_sections_e(knorm,kr,e_inc,alpha_e_dl,input_field;explicit_scattering=true,verbose=true)

Computes the extinction, absorbtion and scattering cross section ``\sigma_{ext}``, ``\sigma_{abs}``, ``\sigma_{sca}`` of a system made out of electric dipoles, illuminated by a plane wave.
Note that it should follow the optical theorem, i.e.
```math
\sigma_{ext}=\sigma_{abs}+\sigma_{sca}
```

# Arguments
- `knorm`: wave number in the medium.
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the total incident electric field ``\mathbf{E}_{i}`` on each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `input_field`: 2D complex array of size ``N\times 3`` containing the input field ``\mathbf{E}_0(\mathbf{r}_i)`` at the position of each dipole. Note that **it must be a plane wave.**
- `explicit_scattering`: boolean that says whether to compute scttering cross section explicitely (`true`) or to deduce it from the optical theorem (`false`). By default set to `true`.
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.

# Outputs
- a float array of length 3 containing in order: extinction, absorption and scattering cross section. Cross sections are given in units of `knorm``^{-2}``.
"""
function compute_cross_sections_e(knorm,kr,e_inc,alpha_e_dl,input_field;explicit_scattering=true,verbose=true)
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
    if ndims(alpha_e_dl)==1 || ndims(alpha_e_dl)==0
        id=1
    else
        id=Matrix{ComplexF64}(I,3,3)
    end
    alpha_e_dl=Alphas.dispatch_e(alpha_e_dl,n)
    #compute_cross sections
    for j=1:n
        #extinction
        sumext=sumext+imag(dot(input_field[j,:],alpha_e_dl[j]*e_inc[j,:]))
        #absorption
        sumabs=sumabs-imag(dot(alpha_e_dl[j]*e_inc[j,:],(inv(factor_p*alpha_e_dl[j])+im*knorm^3/6/pi*id)*alpha_e_dl[j]*e_inc[j,:]))
        #scattering
    end
    if explicit_scattering
        for j=1:n
            sumsca=sumsca+dot(alpha_e_dl[j]*e_inc[j,:],(knorm/6/pi)*alpha_e_dl[j]*e_inc[j,:])
            for k=1:j-1
                G=imag(knorm/4/pi*GreenTensors.G_e_renorm(kr[j,:],kr[k,:]))
                sumsca=sumsca+dot(alpha_e_dl[j]*e_inc[j,:],G*alpha_e_dl[k]*e_inc[k,:])
                sumsca=sumsca+dot(alpha_e_dl[k]*e_inc[k,:],G*alpha_e_dl[j]*e_inc[j,:])
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

Computes the extinction, absorbtion and scattering cross section ``\sigma_{ext}``, ``\sigma_{abs}``, ``\sigma_{sca}`` of a system made out of electric and magnetic dipoles, illuminated by a plane wave.
Note that it should follow the optical theorem, i.e.
```math
\sigma_{ext}=\sigma_{abs}+\sigma_{sca}
```

# Arguments 
- `knorm`: wave number in the medium.
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `phi_inc`: 2D complex array of size ``N\times 6`` containing the total incident electric and magnetic field ``\mathbf{\Phi}_i=(\mathbf{E}_i,\mathbf{H}_i)`` on each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `input_field`: 2D complex array of size ``N\times 6`` containing the electric and magnetic input field ``\mathbf{\Phi}_0=(\mathbf{E}_0(\mathbf{r}_i),\mathbf{H}_0(\mathbf{r}_i))`` at the position of each dipole. Note that **it should be a plane wave**.
- `explicit_scattering`: boolean that says whether to compute scttering cross section explicitely (`true`) or to deduce it from the optical theorem (`false`). By default set to `true`.
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.

# Outputs
- a float array of length 3 containing in order: extinction, absorption and scattering cross section. Cross sections are given in units of `knorm``^{-2}``.
"""
function compute_cross_sections_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field;explicit_scattering=true,verbose=true)
    #redefine things
    e_inc=phi_inc[:,1:3]
    h_inc=phi_inc[:,4:6]
    p=compute_dipole_moment(alpha_e_dl,e_inc)
    m=compute_dipole_moment(alpha_m_dl,h_inc)
    e_inp=input_field[:,1:3]
    h_inp=input_field[:,4:6]
    #logging
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
    cst=2*pi/knorm^2/real(dot(e_inp[1,1:3],e_inp[1,1:3]))
    if (explicit_scattering)
        return [cst*real(sum_ext) cst*real(sum_abs) 2*cst*real(sum_sca)]
    else
        return [cst*real(sum_ext) cst*real(sum_abs) real(sum_ext-sum_abs)]
    end

end

@doc raw"""
    compute_cross_sections_e_m(knorm,kr,phi_inc,alpha_dl,input_field;explicit_scattering=true,verbose=true)
Same as `compute_cross_sections_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field;explicit_scattering=true,verbose=true)`, but the electric and magnetic polarizabilities of each dipole are given by a single 6x6 complex matrix.  See the Alphas module documentation for accepted formats.
"""
function compute_cross_sections_e_m(knorm,kr,phi_inc,alpha_dl,input_field;explicit_scattering=true,verbose=true)
    #redefine things
    e_inc=phi_inc[:,1:3]
    h_inc=phi_inc[:,4:6]
    dip=compute_dipole_moment(alpha_dl,phi_inc)
    p=dip[:,1:3]
    m=dip[:,4:6]
    e_inp=input_field[:,1:3]
    h_inp=input_field[:,4:6]
    #logging
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
    cst=2*pi/knorm^2/real(dot(e_inp[1,1:3],e_inp[1,1:3]))
    if (explicit_scattering)
        return [cst*real(sum_ext) cst*real(sum_abs) 2*cst*real(sum_sca)]
    else
        return [cst*real(sum_ext) cst*real(sum_abs) real(sum_ext-sum_abs)]
    end

end


@doc raw"""
    poynting_vector(phi)
Computes the poynting vector of an electromagnetic field. Input is an electric and magnetic field `phi`(1D complex Array of length 6).
Outputs a 1D float array of length 3 in units of electriqc field squared.
"""
function poynting_vector(phi)
    return 0.5*real(cross(phi[1:3],conj(phi[4:6])))
end


@doc raw"""
    diff_scattering_cross_section_e(knorm,kr,e_inc,alpha_e_dl,input_field,ur;verbose=true)

Computes the differential scattering cross section ``d \sigma_{sca}/ d\Omega`` of a system made out of electric dipoles in direction(s) `ur`.

# Arguments
- `knorm`: wave number in the medium.
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the total incident electric field ``\mathbf{E}_{i}`` on each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `input_field`: 2D complex array of size ``N\times 3`` containing the electric input field ``\mathbf{E}_0(\mathbf{r}_i)`` at the position of each dipole. Note that **it should be a plane wave**
- `ur`: 1D float vector of length 3 (only one direction) or 2D float array of size ``Nu\times 3`` (more thant 1 directions) containing the directions ``\mathbf{u_r}`` where the diffrential scattering cross section is computed. Note that these directions vectors are normalized to 1 in the function.
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.

# Outputs
- an array containing the differential cross section in each direction. Cross sections are given in units of `knorm``^{-2}``.
"""
function diff_scattering_cross_section_e(knorm,kr,e_inc,alpha_e_dl,input_field,ur;verbose=true)
    #logging
    if verbose
        println("computing differential scattering cross section...")
    end
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

    diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field,ur;verbose=true)

Computes the differential scattering cross section ``d \sigma_{sca}/ d\Omega`` of a system made out of electric and magnetic dipoles in direction(s) `ur`.

# Arguments 
- `knorm`: wave number in the medium.
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `phi_inc`: 2D complex array of size ``N\times 6`` containing the total incident electric and magnetic field ``\mathbf{\Phi}_i=(\mathbf{E}_i,\mathbf{H}_i)`` on each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `input_field`: 2D complex array of size ``N\times 6`` containing the electric and magnetic input field ``\mathbf{\Phi}_0=(\mathbf{E}_0(\mathbf{r}_i),\mathbf{H}_0(\mathbf{r}_i))`` at the position of each dipole. Note that **it should be a plane wave.**
- `ur`: 1D float vector of length 3 (only one direction) or 2D float array of size ``Nu\times 3`` (more thant 1 directions) containing the dimensionless positions ``\mathbf{u_r}`` where the diffrential scattering cross section is computed. Note that these directions vectors are normalized to 1 in the function.
- `verbose`: whether to output pieces of information to the standard output during at runtime or not. By default set to `true`.

# Outputs
- an array containing the differential cross section in each direction. Cross sections are given in units of `knorm``^{-2}``.
"""
function diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field,ur;verbose=true)
    #logging
    if verbose
        println("computing differential scattering cross section...")
    end
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
    diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_dl,input_field,ur;verbose=true)
Same as `diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field,ur;verbose=true)`, but the electric and magnetic polarizabilities of each dipole are given by a single 6x6 complex matrix.  See the Alphas module documentation for accepted formats.
"""
function diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_dl,input_field,ur;verbose=true)
    #logging
    if verbose
        println("computing differential scattering cross section...")
    end
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
        
        poynting=poynting_vector(far_field_sca_e_m(kr,phi_inc,alpha_dl,krf))
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
            poynting=poynting_vector(far_field_sca_e_m(kr,phi_inc,alpha_dl,krf))
            d_sigma[i]=real((100*max_norm)^2*dot(poynting,ur[i,:])/(dot(input_field[1,:],input_field[1,:])/2))
        end
        return d_sigma
    end
end

@doc raw"""
    emission_pattern_e(kr,e_inc,alpha_e_dl,ur,krd,dip;verbose=true)
Computes the emission pattern ``d P/ d\Omega`` (differential power emitted in a given direction) of a system made out of electric dipoles in direction(s) of position(s) `krf`.

# Arguments 
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the total incident electric field ``\mathbf{E}_{i}`` on each dipole. The user is required to first solve the DDA problem for `e_inc` in the presence of a point dipole source placed at `krd`.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `ur`: 1D float vector of length 3 (only one direction) or 2D float array of size ``Nu\times 3`` (more thant 1 directions) containing the dimensionless positions ``\mathbf{u_r}`` where the diffrential scattering cross section is computed. Note that these directions vectors are normalized to 1 in the function.
- `krd` : float vector of length 3 containing the dimensionless position of the point electric dipole source.
- `dip`: integer defining the dipole moment (`dip = 1,2,3` is an electric dipole along x,y,z respectively) 
or complex array of size 3 with the components for the electric dipole moment.
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.

# Outputs
- an array containing the differential emitted power in directions `ur` in units of the power emitted by the emitter ``P_0`` without any scatterers.
"""
function emission_pattern_e(kr,e_inc,alpha_e_dl,ur,krd,dip;verbose=true)
    #logging
    if verbose
        println("computing emission pattern...")
    end
    #dipole moment definition
    if ndims(dip) == 0  && dip < 4 && dip > 0
        dipole = zeros(3)
        dipole[dip] = 1
    elseif length(dip) == 3
        dipole = dip/norm(dip) # Ensure that its modulus is equal to one
    else
        dipole = zeros(3)
        println("dip should be an integer (between 1 and 3) or a vector of length 3")
    end
    #computation of the emitted power bxy the dipole source
    P0=4*pi/3*norm(dipole)^2
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
        krf=transpose(ur*100*max_norm)
        #input field on the positions krf
        input_field_krf=InputFields.point_dipole_e(krf,krd,dipole)
        #poynting vector calculations
        poynting=poynting_vector(far_field_sca_e(kr,e_inc,alpha_e_dl,krf).+input_field_krf)
        pow=real((norm(krf))^2*dot(poynting,ur))
        return real(pow/P0)
    #if more than one, i.e. 2D array
    else
        #directions
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
        #compute emission pattern
        pow=zeros(nur)
        for i=1:nur
            krf=transpose(ur[i,:]*100*max_norm)
            #input field on the positions krf
            input_field_krf=InputFields.point_dipole_e(krf,krd,dipole)
            #
            poynting=poynting_vector(far_field_sca_e(kr,e_inc,alpha_e_dl,krf).+input_field_krf)
            pow[i]=real(norm((krf))^2*dot(poynting,ur[i,:]))
        end
        return real(pow/P0)
    end
end


@doc raw"""
    emission_pattern_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,ur,krd,dip;verbose=true)
Computes the emission pattern ``d \P/ d\Omega`` of a system made out of electric and magnetic dipoles in direction(s) of position(s) `krf`.

# Arguments 
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `phi_inc`: 2D complex array of size ``N\times 6`` containing the total incident electric and magnetic field ``\mathbf{\Phi}_i=(\mathbf{E}_i,\mathbf{H}_i)`` on each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `ur`: 1D float vector of length 3 (only one direction) or 2D float array of size ``Nu\times 3`` (more thant 1 directions) containing the dimensionless positions ``\mathbf{u_r}`` where the diffrential scattering cross section is computed. Note that these directions vectors are normalized to 1 in the function.
- `krd` : float vector of length 3 containing the dimensionless position of the point electric and/or magnetic dipole source.
- `dip`: integer defining the dipole moment (`dip = 1,2,3` is an electric dipole along x,y,z respectively and `dip = 4,5,6` is a magnetic dipole along x,y,z respectively) 
or complex array of size 6 with the first 3 components for the electric dipole moment and the last 3 components for the magnetic dipole moment.
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.

# Outputs
- an array containing the differential emitted power in directions `ur` in units of the power emitted by the emitter ``P_0`` without any scatterers.
"""
function emission_pattern_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,ur,krd,dip;verbose=true)
    #logging
    if verbose
        println("computing emission pattern...")
    end
    #dipole moment definition
    if ndims(dip) == 0  && dip < 7 && dip > 0
        dipole = zeros(6)
        dipole[dip] = 1
    elseif length(dip) == 6
        dipole = dip/norm(dip) # Ensure that its modulus is equal to one
    else
        dipole = zeros(6)
        println("dip should be an integer (between 1 and 6) or a vector of length 6")
    end
    #computation of the emitted power bxy the dipole source
    P0=4*pi/3*(norm(dipole[1:3])^2+norm(dipole[4:6])^2)
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
        krf=transpose(ur*100*max_norm)
        #input field on the positions krf
        input_field_krf=InputFields.point_dipole_e_m(krf,krd,dipole)
        #
        poynting=poynting_vector(far_field_sca_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,krf).+input_field_krf)
        pow=real((norm(krf))^2*dot(poynting,ur))
        return real(pow/P0)
    #if more than one, i.e. 2D array
    else
        #directions
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
        #compute differential cross section
        pow=zeros(nur)
        for i=1:nur
            krf=transpose(ur[i,:]*100*max_norm)
            #input field on the positions krf
            input_field_krf=InputFields.point_dipole_e_m(krf,krd,dipole)
            #
            poynting=poynting_vector(far_field_sca_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,krf).+input_field_krf)
            pow[i]=real(norm((krf))^2*dot(poynting,ur[i,:]))
        end
        return real(pow/P0)
    end
end

@doc raw"""
    emission_pattern_e_m(kr,phi_inc,alpha_dl,ur,krd,dip;verbose=true)
Same as `emission_pattern_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,ur,krd,dip;verbose=true)`, but the electric and magnetic polarizabilities of each dipole are given by a single 6x6 complex matrix.  See the Alphas module documentation for accepted formats.
"""
function emission_pattern_e_m(kr,phi_inc,alpha_dl,ur,krd,dip;verbose=true)
    #logging
    if verbose
        println("computing emission pattern...")
    end
    #computation of the emitted power bxy the dipole source
    if length(dip)==6
        P0=4*pi/3*(norm(dip[1:3])^2+norm(dip[4:6])^2)
    else
        P0=4*pi/3
    end
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
        krf=transpose(ur*100*max_norm)
        #input field on the positions krf
        input_field_krf=InputFields.point_dipole_e_m(krf,krd,dip)
        #
        poynting=poynting_vector(far_field_sca_e_m(kr,phi_inc,alpha_dl,krf).+input_field_krf)
        pow=real((norm(krf))^2*dot(poynting,ur))
        return real(pow/P0)
    #if more than one, i.e. 2D array
    else
        #input field on the positions krf
        input_field_krf=InputFields.point_dipole_e_m(krf,krd,dip)
        #directions
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
        #compute differential cross section
        pow=zeros(nur)
        for i=1:nur
            krf=transpose(ur[i,:]*100*max_norm)
            #input field on the positions krf
            input_field_krf=InputFields.point_dipole_e_m(krf,krd,dipole)
            #
            poynting=poynting_vector(far_field_sca_e_m(kr,phi_inc,alpha_dl,krf[i,:]).+input_field_krf[i,:])
            pow[i]=real(norm((krf))^2*dot(poynting,ur[i,:]))
        end
        return real(pow/P0)
    end
end

@doc raw"""
    field_sca_e_m(kr, alpha_e_dl, alpha_m_dl, phi_inc, krf; verbose=true)
Computes the scattered field from a system made out of electric and magnetic dipoles.

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `phi_inc`: 2D complex array of size ``N\times 6`` containing the total incident electric and magnetic field ``\mathbf{\Phi}_i=(\mathbf{E}_i,\mathbf{H}_i)`` on each dipole.
- `krf`: 1D float vector of length 3 (only one position) or 2D float array of size ``Nu\times 3`` (more thant 1 position) containing the dimensionless positions ``k\mathbf{r}_f`` at which the scattered field is computed.
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.

# Outputs
- `field_r`: 2D complex array of size ``Nf\times 6`` containing the scattered electric and magnetic fields by the dipoles at every ``k\mathbf{r_f}``.
"""
function field_sca_e_m(kr, alpha_e_dl, alpha_m_dl, phi_inc, krf; verbose=true)
    #logging
    if verbose
        println("computing scattered field...")
    end
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
    phi_inc = reshape(transpose(phi_inc),n_particles*6,)
    field_r = G_tensor_fr*phi_inc
    return transpose(reshape(field_r,6,n_r0))   
end

@doc raw"""
    field_sca_e_m(kr, alpha_dl, phi_inc, krf)
Same as `field_sca_e_m(kr, alpha_e_dl, alpha_m_dl, phi_inc, krf)`, but the electric and magnetic polarizabilities of each dipole are given by a single 6x6 complex matrix.  See the Alphas module documentation for accepted formats.
"""
function field_sca_e_m(kr, alpha_dl, phi_inc, krf; verbose=true)
    #logging
    if verbose
        println("computing scattered field...")
    end
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
    phi_inc = reshape(transpose(phi_inc),n_particles*6,)
    field_r = G_tensor_fr*phi_inc
    return transpose(reshape(field_r,6,n_r0))       
end

@doc raw"""
    function field_sca_e(kr, alpha_e_dl, e_inc, krf)
Computes the scattered field from a system made out of electric dipoles. Note that the term ``e^{ikr}/kr`` is retained, so `krf` has to be finite.

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the total incident electric field ``\mathbf{E}_{i}`` on each dipole.
- `krf`: 1D float vector of length 3 (only one position) or 2D float array of size ``Nu\times 3`` (more thant 1 position) containing the dimensionless positions ``k\mathbf{r}_f`` at which the scattered field is computed.
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.

# Outputs
- `field_r`: 2D complex array of size ``Nf\times 6`` containing the scattered electric and magnetic fields by the dipoles at every ``k\mathbf{r_f}``. Note that the term ``e^{ikr}/kr`` is retained, so `krf` has to be finite.
"""
function field_sca_e(kr, alpha_e_dl, e_inc, krf; verbose=true)
    #logging
    if verbose
        println("computing scattered field...")
    end
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
    return transpose(reshape(field_r,3,n_r0))       
end

@doc raw"""
    function far_field_sca_e(kr,e_inc,alpha_e_dl,krf)
Computes the scattered field from a system made out of electric dipoles in the far field approximation. Note that the term ``e^{ikr}/kr`` is retained, so `krf` has to be finite.

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole. Note that these positions have to be far away from the dipoles positions.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the incident electric field ``\mathbf{E}_{i}`` on each dipole.
- `krf`: 1D float vector of length 3 (only one position) or 2D float array of size ``Nu\times 3`` (more thant 1 position) containing the dimensionless positions ``k\mathbf{r}_f`` at which the scattered field is computed.

# Outputs
- `field_r`: 2D complex array of size ``Nf\times 6`` containing the scattered field by the dipoles at every ``k\mathbf{r_f}``. Note that the term ``e^{ikr}/kr`` is retained, so `krf` has to be finite.

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
        nf=length(ComplexF64,krf[:,1])
        res=zeros(nf,6)
        for j=1:nf
            for i=1:n
                Ge,Gm=GreenTensors.G_em_far_field_renorm(krf[j,:],kr[i,:])
                res[j,1:3]=res[j,1:3]+Ge*alpha_e_dl[i]*e_inc[i,:]
                res[j,4:6]=res[j,4:6]-im*Gm*alpha_e_dl[i]*e_inc[i,:]
            end
        end
        return res
    end
end

@doc raw"""
    function far_field_sca_e_m(kr,e_inc,alpha_e_dl,alpha_m_dl,krf)
Computes the scattered field from a system made out of electric and magnetic dipoles in the far field approximation. Note that the term ``e^{ikr}/kr`` is retained, so `krf` has to be finite.

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole. Note that these positions have to be far away from the dipoles positions.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `phi_inc`: 2D complex array of size ``N\times 6`` containing the incident electric and magnetic field ``\mathbf{\Phi}_i=(\mathbf{E}_i,\mathbf{H}_i)`` on each dipole.
- `krf`: 1D float vector of length 3 (only one position) or 2D float array of size ``Nu\times 3`` (more thant 1 position) containing the dimensionless positions ``k\mathbf{r}_f`` at which the scattered field is computed.

# Outputs
- `field_r`: 2D complex array of size ``Nf\times 6`` containing the scattered field by the dipoles at every ``k\mathbf{r_f}``. Note that the term ``e^{ikr}/kr`` is retained, so `krf` has to be finite.

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
        nf=length(krf[:,1])
        res=zeros(ComplexF64,nf,6)
        for j=1:nf
            for i=1:n
                Ge,Gm=GreenTensors.G_em_far_field_renorm(krf[j,:],kr[i,:])
                res[j,1:3]=res[j,1:3]+Ge*alpha_e_dl[i]*phi_inc[i,1:3]+im*Gm*alpha_m_dl[i]*phi_inc[i,4:6]
                res[j,4:6]=res[j,4:6]-im*Gm*alpha_e_dl[i]*phi_inc[i,1:3]+Ge*alpha_m_dl[i]*phi_inc[i,4:6]
            end
        end
        return res
    end
end

@doc raw"""
    function far_field_sca_e_m(kr,e_inc,alpha_dl,krf)
Same as `far_field_sca_e_m(kr, alpha_e_dl, alpha_m_dl, phi_inc, krf)`, but the electric and magnetic polarizabilities of each dipole are given by a single 6x6 complex matrix.  See the Alphas module documentation for accepted formats.
"""
function far_field_sca_e_m(kr,phi_inc,alpha_dl,krf)
    #number of dipoles
    n=length(kr[:,1])
    #dispatch alpha
    alpha_dl=Alphas.dispatch_e_m(alpha_dl,n)
    #if only one position
    if ndims(krf)==1
        res=zeros(ComplexF64,6)
        for i=1:n
            Ge,Gm=GreenTensors.G_em_far_field_renorm(krf,kr[i,:])
            big_g=zeros(6,6)
            big_g[1:3,1:3]=Ge
            big_g[1:3,4:6]=im*Gm
            big_g[4:6,1:3]=-im*Gm
            big_g[1:3,4:6]=Ge
            res=big_g*alpha_dl*phi_inc
        end
        return res
    #if more than 1, i.e. 2D array
    else
        nf=length(krf[:,1])
        res=zeros(ComplexF64,nf,6)
        for j=1:nf
            for i=1:n
                Ge,Gm=GreenTensors.G_em_far_field_renorm(krf[j,:],kr[i,:])
                big_g=zeros(6,6)
                big_g[1:3,1:3]=Ge
                big_g[1:3,4:6]=im*Gm
                big_g[4:6,1:3]=-im*Gm
                big_g[1:3,4:6]=Ge
                res[j,:]=big_g*alpha_dl*phi_inc
            end
        end
        return res
    end
end

@doc raw"""
    ldos_e_m(kr, alpha_e_dl, alpha_m_dl, Ainv, krd; dip=nothing, verbose=true)
Computes local density of states (LDOS) of a system made out of electric and magnetic dipoles normalized by the LDOS in the host medium without scatterers. 

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `alpha_m_dl`: complex dimensionless magnetic polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `Ainv`: (inverse) CEMD matrix.
- `krd`: 2D float array of size ``Nd\times 3`` containing the dimentionless positions ``k\mathbf{r_d}`` where the LDOS is calculated.
- `dip`: integer defining the dipole moment (`dip = 1,2,3` is an electric dipole along x,y,z respectively and `dip = 4,5,6` is a magnetic dipole along x,y,z respectively) 
or complex array of size 6 with the first 3 components for the electric dipole moment and the last 3 components for the magnetic dipole moment. 
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.

# Outputs
- `LDOS`: float array with the LDOS normalized by the LDOS in the host medium without scatterers.
"""
function ldos_e_m(kr, alpha_e_dl, alpha_m_dl, Ainv, krd; dip=nothing, verbose=true)
    #logging
    if verbose
        println("computing LDOS...")
    end
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
        if ndims(dip) == 0  && dip < 7 && dip > 0
            dipole = zeros(6)
            dipole[dip] = 1
        elseif length(dip) == 6
            dipole = dip/norm(dip) # Ensure that its modulus is equal to one
        else
            dipole = zeros(6)
            println("dip should be an integer (between 1 and 6) or a vector of length 6")
        end
        for j=1:n_dpos
            for i=1:n_particles
                Ge, Gm = GreenTensors.G_em_renorm(kr[i,:],krd[j,:])        
                G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [Ge im*Gm; -im*Gm Ge]
                G_tensor_fr[:, 6 * (i-1) + 1:6 * (i-1) + 6] = [Ge*alp_e[i] -im*Gm*alp_m[i]; im*Gm*alp_e[i] Ge*alp_m[i]]
            end
            field_r = G_tensor_fr*Ainv*G_tensor*dipole
            LDOS[j] = 1 + imag(transpose(dipole)*field_r)/(2/3)
        end
    end
    if length(LDOS) == 1
        LDOS = LDOS[1]
    end
    return LDOS
end

@doc raw"""
    ldos_e_m(kr, alpha_dl, Ainv, krd; dip=nothing, verbose=true)
Same as `ldos_e_m(kr, alpha_e_dl, alpha_m_dl, Ainv, krd; dip=nothing, verbose=true)`, but the electric and magnetic polarizabilities of each dipole are given by a single 6x6 complex matrix.  See the Alphas module documentation for accepted formats.
"""
function ldos_e_m(kr, alpha_dl, Ainv, krd; dip=nothing, verbose=true)
    #logging
    if verbose
        println("computing LDOS...")
    end
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
        if ndims(dip) == 0  && dip < 7 && dip > 0
            dipole = zeros(6)
            dipole[dip] = 1
        elseif length(dip) == 6
            dipole = dip/norm(dip) # Ensure that its modulus is equal to one
        else
            dipole = zeros(6)
            println("dip should be an integer (between 1 and 6) or a vector of length 6")
        end
        for j=1:n_dpos
            for i=1:n_particles
                Ge, Gm = GreenTensors.G_em_renorm(kr[i,:],krd[j,:])        
                G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , :] = [Ge im*Gm; -im*Gm Ge]
                G_tensor_fr[:, 6 * (i-1) + 1:6 * (i-1) + 6] = [Ge -im*Gm; im*Gm Ge]*alp[i]
            end
            field_r = G_tensor_fr*Ainv*G_tensor*dipole
            LDOS[j] = 1 + imag(transpose(dipole)*field_r)/(2/3)
        end
    end
    if length(LDOS) == 1
        LDOS = LDOS[1]
    end
    return LDOS
end

@doc raw"""
    ldos_e(kr, alpha_e_dl, Ainv, krd; dip=nothing, verbose=true)
It Computes local density of states (LDOS) of a system made out of electric dipoles normalized by the LDOS in the host medium without scatterers. 

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `alpha_e_dl`: complex dimensionless electric polarizability of each dipole. See the Alphas module documentation for accepted formats.
- `Ainv`: (inverse) DDA matrix.
- `krd`: 2D float array of size ``Nd\times 3`` containing the dimentionless positions ``k\mathbf{r_d}`` where the LDOS is calculated.
- `dip`: integer defining the dipole moment (`dip = 1,2,3` is an electric dipole along x,y,z respectively) 
or complex array of size 3 with the components for the electric dipole moment.  
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.

# Outputs
- `LDOS`: float array with the LDOS normalized by the LDOS in the host medium without scatterers.
"""
function ldos_e(kr, alpha_e_dl, Ainv, krd; dip=nothing, verbose=true)
    #logging
    if verbose
        println("computing LDOS...")
    end
    n_particles = length(kr[:,1])
    n_dpos = length(krd[:,1])

    G_tensor = zeros(ComplexF64,n_particles*3,3) 
    G_tensor_fr = zeros(ComplexF64,3,n_particles*3)

    alp_e = Alphas.dispatch_e(alpha_e_dl,n_particles)

    LDOS = zeros(n_dpos)
    if dip == nothing
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
        if ndims(dip) == 0 && dip < 4 && dip > 0
            dipole = zeros(3)
            dipole[dip] = 1
        elseif length(dip) == 3
            dipole = dip/norm(dip) # Ensure that its modulus is equal to one
        else
            dipole = zeros(3)
            println("dip should be an integer (between 1 and 3) or a vector of length 3")
        end
        for j=1:n_dpos
            for i=1:n_particles
                Ge = GreenTensors.G_e_renorm(kr[i,:],krd[j,:])        
                G_tensor[3 * (i-1) + 1:3 * (i-1) + 3 , :] = Ge
                G_tensor_fr[:, 3 * (i-1) + 1:3 * (i-1) + 3] = Ge*alp_e[i] 
            end
            field_r = G_tensor_fr*Ainv*G_tensor*dipole
            LDOS[j] = 1 + imag(transpose(dipole)*field_r)/(2/3)
        end
    end
    if length(LDOS) == 1
        LDOS = LDOS[1]
    end
    return LDOS
end


@doc raw"""
    rad_ldos_e(kr,krd,p,dip;verbose=true)
Computes the radiative part of the LDOS of a system of electric point dipoles scatterers normalized by the LDOS in the host medium without scatterers. 

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `krd`: float array of size 3 containing the dimentionless position ``k\mathbf{r_d}`` where the radiative part of the LDOS is calculated.
- `p`: 2D complex array of size``Nd\times 3`` containing the electric dipole moments of the dipoles.
- `dip`: integer defining the dipole moment (`dip = 1,2,3` is an electric dipole along x,y,z respectively) 
or complex array of size 3 with the components for the electric dipole moment.  
- `verbose`: whether to output pieces of information to the standard output during running or not. By default set to `true`.
# Outputs
- float containing the normalized (by the LDOS in the host medium without scatterers) radiative LDOS at the position `krd`.
"""
function rad_ldos_e(kr,krd,p,dip;verbose=true)
    #dipole moment
    if ndims(dip) == 0  && dip < 4 && dip > 0
        dipole = zeros(3)
        dipole[dip] = 1
    elseif length(dip) == 3
        dipole = dip/norm(dip) # Ensure that its modulus is equal to one
    else
        dipole = zeros(3)
        println("dip should be an integer (between 1 and 6) or a vector of length 6")
    end
    #adding the source to the list of dipoles
    p=vcat(p,transpose(dipole))
    kr=vcat(kr,krd)
    #logging
    if verbose
        println("computing radiative LDOS...")
    end
    #define sum variables
    sum_sca=0
    #compute rad LDOS
    n=length(p[:,1])
    for i=1:n
        sum_sca=sum_sca+dot(p[i,:],p[i,:])
        for j=(i+1):n
            sum_sca=sum_sca+3*real(transpose(p[j,:])*(imag(GreenTensors.G_e_renorm(kr[j,:],kr[i,:]))*conj(p[i,:])))
        end
    end
    return real(sum_sca)/norm(dipole)
end


@doc raw"""
    rad_ldos_e_m(kr,krd,p,m,dip;verbose=true)
Computes the radiative part of the LDOS of a system of electric and magnetic point dipoles scatterers normalized by the LDOS in the host medium without scatterers. 

# Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimensionless position ``k\mathbf{r}`` of each dipole.
- `krd`: float array of size 3 containing the dimentionless position ``k\mathbf{r_d}`` where the LDOS is calculated.
- `p`: 2D complex array of size``Nd\times 3`` containing the electric dipole moments of the dipoles.
- `m`: 2D complex array of size``Nd\times 3`` containing the magnetic dipole moments of the dipoles.
- `dip`: integer defining the dipole moment (`dip = 1,2,3` is an electric dipole along x,y,z respectively and `dip = 4,5,6` is a magnetic dipole along x,y,z respectively) 
or complex array of size 6 with the first 3 components for the electric dipole moment and the last 3 components for the magnetic dipole moment.  
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.
# Outputs
- float containing the normalized (by the LDOS in the host medium without scatterers) radiative LDOS at the position `krd`.
"""
function rad_ldos_e_m(kr,krd,p,m,dip;verbose=true)
    #dipole moment dispatch
    if ndims(dip) == 0  && dip < 7 && dip > 0
        dipole = zeros(6)
        dipole[dip] = 1
    elseif length(dip) == 6
        dipole = dip/norm(dip) # Ensure that its modulus is equal to one
    else
        dipole = zeros(6)
        println("dip should be an integer (between 1 and 6) or a vector of length 6")
    end
    #adding the source to the list of dipoles
    p=vcat(p,reshape(dipole[1:3],1,3))
    m=vcat(m,reshape(dipole[4:6],1,3))
    kr=vcat(kr,krd)
    #logging
    if verbose
        println("computing total radiated power...")
    end
    #define sum variables
    sum_sca=0
    #compute rad LDOS
    n=length(p[:,1])
    for i=1:n
        sum_sca=sum_sca+dot(p[i,:],p[i,:])+dot(m[i,:],m[i,:])
        for j=(i+1):n
            sum_sca=sum_sca+3*(real(transpose(p[j,:])*(imag(GreenTensors.G_e_renorm(kr[j,:],kr[i,:]))*conj(p[i,:])) + transpose(m[j,:])*(imag(GreenTensors.G_e_renorm(kr[j,:],kr[i,:]))*conj(m[i,:]))))
            sum_sca=sum_sca+3*(imag(-transpose(conj(p[i,:]))*imag(GreenTensors.G_m_renorm(kr[i,:],kr[j,:]))*m[j,:]    +   transpose(conj(p[j,:]))*imag(GreenTensors.G_m_renorm(kr[i,:],kr[j,:]))*m[i,:]))
        end
    end
    return real(sum_sca)/(norm(dipole[1:3])^2+norm(dipole[4:6])^2)
end

@doc raw"""
    nonrad_ldos_e(p,e_inc,dip;verbose=true)
Computes the non-radiative part of the LDOS of a system of electric point dipoles scatterers normalized by the LDOS in the host medium without scatterers. Note that the scattering problem has to be solved previously using a point dipole source as input field.

# Arguments
- `p`: 2D complex array of size``Nd\times 3`` containing the electric dipole moments of the dipoles. It has to be previously computed by one of the DDACore functions and `compute_dipole_moment`.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the total incident electric field ``\mathbf{E}_{i}`` on each dipole. It has to be previously computed by one of the DDACore functions, setting the input field to be a point dipole source.
- `dip`: integer defining the dipole moment (`dip = 1,2,3` is an electric dipole along x,y,z respectively) 
or complex array of size 3 with the components for the electric dipole moment.  
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.
# Outputs
- float array containing the normalized non-radiative LDOS at everey position of `krd`.
"""
function nonrad_ldos_e(p,e_inc,dip;verbose=true)
    #dipole moment
    if ndims(dip) == 0  && dip < 4 && dip > 0
        dipole = zeros(3)
        dipole[dip] = 1
    elseif length(dip) == 3
        dipole = dip/norm(dip) # Ensure that its modulus is equal to one
    else
        dipole = zeros(3)
        println("dip should be an integer (between 1 and 6) or a vector of length 6")
    end
    #logging
    if verbose
        println("computing non-radiative LDOS...")
    end
    #define sum variables
    sum_abs=0
    #compute power
    n=length(p[:,1])
    for i=1:n
        sum_abs=sum_abs+(imag(dot(e_inc[i,:],p[i,:])) -2/3*dot(p[i,:],p[i,:]))
    end
    return 1.5*real(sum_abs)/norm(dipole)^2
end

@doc raw"""
    nonrad_ldos_e_m(p,m,phi_inc,dip;verbose=true)
Computes the non-radiative part of the LDOS of a system of electric and magnetic point dipoles scatterers normalized by the LDOS in the host medium without scatterers. Note that the scattering problem has to be solved previously using a point dipole source as input field.

# Arguments
- `p`: 2D complex array of size``Nd\times 3`` containing the electric dipole moments of the dipoles. It has to be previously computed by one of the DDACore functions and `compute_dipole_moment`.
- `m`: 2D complex array of size``Nd\times 3`` containing the magnetic dipole moments of the dipoles. It has to be previously computed by one of the DDACore functions, setting the input field to be a point dipole source.
- `phi_inc`: 2D complex array of size ``N\times 6`` containing the total incident electric and magnetic field ``\mathbf{\Phi}_i=(\mathbf{E}_i,\mathbf{H}_i)`` on each dipole.
- `dip`: integer defining the dipole moment (`dip = 1,2,3` is an electric dipole along x,y,z respectively and `dip = 4,5,6` is a magnetic dipole along x,y,z respectively) 
or complex array of size 6 with the first 3 components for the electric dipole moment and the last 3 components for the magnetic dipole moment.  
- `verbose`: whether to output pieces of information to the standard output at runtime or not. By default set to `true`.
# Outputs
- 1D float array containing the normalized non-radiative LDOS at everey position of `krd`.
"""
function nonrad_ldos_e_m(p,m,phi_inc,dip;verbose=true)
    #dipole moment dispatch
    if ndims(dip) == 0  && dip < 7 && dip > 0
        dipole = zeros(6)
        dipole[dip] = 1
    elseif length(dip) == 6
        dipole = dip/norm(dip) # Ensure that its modulus is equal to one
    else
        dipole = zeros(6)
        println("dip should be an integer (between 1 and 6) or a vector of length 6")
    end
    #logging
    if verbose
        println("computing non-radiative LDOS...")
    end
    #redefine things
    e_inc=phi_inc[:,1:3]
    h_inc=phi_inc[:,4:6]
    #define sum variables
    sum_abs=0
    #compute power
    n=length(p[:,1])
    for i=1:n
        sum_abs=sum_abs+(imag(dot(e_inc[i,:],p[i,:])) -2/3*dot(p[i,:],p[i,:]))+imag(dot(h_inc[i,:],m[i,:]))-2/3*dot(m[i,:],m[i,:])
    end
    return 1.5*real(sum_abs)/(norm(dipole[1:3])^2+norm(dipole[4:6])^2)
end

end
