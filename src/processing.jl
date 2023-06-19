module PostProcessing
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

@doc raw"""
    compute_dipole_moment(alpha,phi_inc)

Computes the dipole moment (magnetic or electric) of a dipole with polarizability `alpha` under an incident field `phi_inc` 

"""
function compute_dipole_moment(alpha,phi_inc)
    n=length(phi_inc[:,1])
    p=zeros(ComplexF64,n,length(phi_inc[1,:]))
    if ndims(alpha)==2
        for i=1:n
            p[i,:]=alpha[i]*phi_inc[i,:]
        end
    elseif ndims(alpha)==3
        for i=1:n
            p[i,:]=alpha[i,:,:]*phi_inc[i,:]
        end
    end
    return p
end


@doc raw"""
    compute_cross_sections_e(knorm,kr,e_inc,alpha,input_field;explicit_scattering=true,verbose=true)

Computes the extinction, absorbtion ans scattering cross section ``\sigma_{ext}``, ``\sigma_{abs}``, ``\sigma_{sca}`` of a system made out of electric dipoles.
Notes that it sould follow the optical theorem (otherwise something is wrong), i.e.
```math
\sigma_{ext}=\sigma_{abs}+\sigma_{sca}
```

Inputs 
- `knorm`: wavenumber
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `e_inc`: 2D complex array of size ``N\times 3`` containing the incident fields ``E_inc`` on every dipole.
- `alpha`
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

    if ndims(alpha_dl)==1
        for j=1:n
            #extinction
            sumext=sumext+imag(dot(input_field[j,:],alpha_dl[j]*e_inc[j,:]))
            #absorption
            sumabs=sumabs-imag(dot(alpha_dl[j]*e_inc[j,:],(inv(factor_p*alpha_dl[j])+im*knorm^3/6/pi*Matrix{ComplexF64}(I,3,3))*alpha_dl[j]*e_inc[j,:]))
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
    elseif ndims(alpha_dl)==3
        for j=1:n
            #extinction
            sumext=sumext+imag(dot(input_field[j,:],alpha_dl[j,:,:]*e_inc[j,:]))
            #absorption
            sumabs=sumabs-imag(dot(alpha_dl[j,:,:]*e_inc[j,:],(inv(factor_p*alpha_dl[j,:,:])+im*knorm^3/6/pi*Matrix{ComplexF64}(I,3,3))*alpha_dl[j,:,:]*e_inc[j,:]))
            #scattering
        end
        if explicit_scattering
            for j=1:n
                sumsca=sumsca+dot(alpha_dl[j,:,:]*e_inc[j,:],(knorm/6/pi)*alpha_dl[j,:,:]*e_inc[j,:])
                for k=1:j-1
                    G=imag(knorm/4/pi*GreenTensors.G_e_renorm(kr[j,:],kr[k,:]))
                    sumsca=sumsca+dot(alpha_dl[j,:,:]*e_inc[j,:],G*alpha_dl[k,:,:]*e_inc[k,:])
                    sumsca=sumsca+dot(alpha_dl[k,:,:]*e_inc[k,:],G*alpha_dl[j,:,:]*e_inc[j,:])
                end
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

#*************************************************
#COMPUTE THE POWERS FOR PLANE_WAVE INPUT FIELD AND ELECTRIC AND MAGNETIC DIPOLES
#INPUTS:  norm of the wave vector, polarisations, incident fields, quasistatic polarisabilities,e0,whether to compute explicitely csca, verbose
#OUTPUT: array with lambda, Cabs, Csca, Cext
#*************************************************
function compute_cross_sections_e_m(knorm,kr,phi_inc,input_field,alpha_e,alpha_m;explicit_scattering=true,verbose=true)
    #redefine things
    e_inc=phi_inc[:,1:3]
    h_inc=phi_inc[:,4:6]
    p=compute_dipole_moment(alpha_e,e_inc)
    m=compute_dipole_moment(alpha_m,h_inc)
    println(size(p))
    e_inp=input_field[:,1:3]
    h_inp=input_field[:,4:6]
    if verbose
        println("computing cross sections")
    end
    #number of dipoles
    n=length(kr[:,1])
    #define sum variables
    sum_sca=0.
    sum_ext=0.
    sum_abs=0.

    for i=1:n
        sum_ext=sum_ext+imag(alpha_e[i,1,1]*dot(e_inp[i,:],e_inc[i,:])+alpha_m[i,1,1]*dot(h_inp[i,:],h_inc[i,:]))
        sum_abs=sum_abs+ ( imag(alpha_e[i,1,1]) -2/3*abs2(alpha_e[i,1,1]))*dot(e_inc[i,:],e_inc[i,:])+(imag(alpha_m[i,1,1])-2/3*abs2(alpha_m[i,1,1]))*dot(h_inc[i,:],h_inc[i,:])
        if (explicit_scattering)
            sum_sca=sum_sca+1/3*dot(p[i,:],p[i,:])+1/3*dot(m[i,:],m[i,:])
            for j=(i+1):n
                sum_sca=sum_sca+real(transpose(p[j,:])*(imag(GreenTensors.G_e_renorm(kr[j,:],kr[i,:]))*conj(p[i,:])) + transpose(m[j,:])*(imag(GreenTensors.G_e_renorm(kr[j,:],kr[i,:]))*conj(m[i,:])))
                sum_sca=sum_sca+imag(-transpose(conj(p[i,:]))*imag(GreenTensors.G_m_renorm(kr[i,:],kr[j,:]))*m[j,:]    +   transpose(conj(p[j,:]))*imag(GreenTensors.G_m_renorm(kr[i,:],kr[j,:]))*m[i,:])
            end
        end

    end
    cst=2*pi/knorm^2*2/dot(e_inp[1,1:3],e_inp[1,1:3])
    if (explicit_scattering)
        return [2*pi/knorm cst*sum_ext cst*sum_abs 2*cst*sum_sca]
    else
        return [2*pi/knorm cst*real(sum_ext) cst*real(sum_abs) real(sum_ext-sum_abs)]
    end

end


@doc raw"""
    field_sca(knorm, alpha, E_inc, r0, pos)
It computes the scattered Field from the ensamble of dipoles.

Imputs
- `knorm` = wavenumber
- `alpha` = polarizability of the particles (6N x 6N matrix, where -N- is the number of dipoles)
- `E_inc` = incoming field at every dipole (6N x 1 vector, where -N- is the number of dipoles). It is equal to the product of the (inverse) DDA matrix and the external field. 
- `r0` = position where the field is observed (Np x 3 matrix, where -Np- is the number of points where the field is calculated)
- `pos` = position of the dipoles (N x 3 matrix, where -N- is the number of points)

Outputs
- `field_r` is the field scattered by the dipoles (6N x 1 vector)


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
function field_sca(knorm, alpha, E_inc, r0, pos)

    N_particles = length(pos[:,1]) 
    N_r0 = length(r0[:,1]) 

    G_tensor = zeros(ComplexF64,N_r0*6,N_particles*6)
    G = zeros(ComplexF64,6,6)

    for i = 1:N_particles
        for j = 1:N_r0
            Ge, Gm = GreenTensors.G_em(r0[j,:],pos[i,:],knorm)   
            G[:,:] = [Ge im*Gm; -im*Gm Ge]
            G_tensor[6 * (j-1) + 1:6 * (j-1) + 6 , 6 * (i-1) + 1:6 * (i-1) + 6] = copy(G)
	end
    end

    p = alpha*E_inc 
    field_r = knorm^2*G_tensor*p

    return field_r
            
end


@doc raw"""
    LDOS_rf(knorm, alpha, Ainv, pos, rd, dip_o)
It Computes partial local density of states (LDOS) by the imaginary part of the returning field (rf)

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
\mathrm{LDOS}(\mathbf{\bar{r}}_N,\r_0) = 1 + \dfrac{1}{|\bm \mu|^2 }\dfrac{6\pi}{k^3} \Im\left[\bm \mu^{*} \cdot k^2\GG(\r_0,\mathbf{\bar{r}}_N) \alphagg(\mathbf{\bar{r}}_N) \DD(\mathbf{\bar{r}}_N) k^2 \GG(\mathbf{\bar{r}}_N, \r_0) \bm \mu  \right]
```

\alphagg(\mathbf{\bar{r}}_N) = `alpha`
\DD(\mathbf{\bar{r}}_N) = `Ainv` = [I - k^2*G*alpha]^(-1)
\bar{r}}_N = `pos`
\r_0 = `rd`
\dfrac{1}{\epsilon_0\epsilon} \bm \mu = `dip_o` (the pre-factor -\dfrac{1}{\epsilon_0\epsilon}- dessapears after normalization)


k^2\GG(\r_0,\mathbf{\bar{r}}_N) \alphagg(\mathbf{\bar{r}}_N) \DD(\mathbf{\bar{r}}_N) k^2 \GG(\mathbf{\bar{r}}_N, \r_0) \bm \mu = `field_r`
\DD(\mathbf{\bar{r}}_N) k^2 \GG(\mathbf{\bar{r}}_N, \r_0) \bm \mu = `E_inc`


\mathrm{LDOS}(\mathbf{\bar{r}}_N,\r_0) = `LDOS`
"""
function LDOS_rf(knorm, alpha, Ainv, pos, rd, dip_o)
 
    E_0i = InputFields.point_dipole(knorm, 1, pos, rd, dip_o)
    E_inc = Ainv*E_0i

    field_r = field_sca(knorm, alpha, E_inc, rd, pos) 
    
    if length(dip_o) == 1 
        LDOS = 1 + imag(field_r[dip_o])/( (knorm)^3/(6*pi) ) # Imaginary part of the dipole component (LDOS)
    else
        dip_o = dip_o/norm(dip_o)
        LDOS = 1 + dot(conj(dip_o), imag(field_r))/( (knorm)^3/(6*pi) )
    end

    return LDOS

end

@doc raw"""
    LDOS_rf(knorm, alpha, Ainv, pos, rd, dip_o)
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

@doc raw"""
    LDOS_EP(knorm, alpha, Ainv, pos, rd, dip_o)
It Computes partial local density of states (LDOS) by calculating the emiting power (EP). Due to the integration, the method is not very accurate.

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
LDOS = P/P_{0}
```

"""
function LDOS_EP(knorm, alpha, Ainv, pos, rd, dip_o)

    Pos = [pos; rd]
    
    center = sum(Pos,dims=1)/length(Pos[:,1])
    
    Pos_c = zeros(length(Pos[:,1]),3)
    Pos_c[:,1] = Pos[:,1] .- center[1,1]
    Pos_c[:,2] = Pos[:,2] .- center[1,2]
    Pos_c[:,3] = Pos[:,3] .- center[1,3]
    
    R = maximum(sqrt.(sum(Pos_c.^2,dims=2)))
    
    Rc = 1.5*R	# Radius of integration
    
    Nth = 100	# Discretitation points in theta
    Nph = 100	# Discretitation points in phi
    
    theta = LinRange(0,pi,Nth)
    phi = LinRange(0,2*pi,Nph)
        
    E_0i = InputFields.point_dipole(knorm, 1, pos, rd, dip_o)
    
    E_inc = Ainv*E_0i
    
    rf = zeros(Nth*Nph,3)
    Theta = zeros(Nth*Nph,1)

    for j=1:Nph
        
        rf[1 + (j-1)*Nth:Nth + (j-1)*Nth,1] = Rc*sin.(theta)*cos(phi[j]) .+ center[1,1]
        rf[1 + (j-1)*Nth:Nth + (j-1)*Nth,2] = Rc*sin.(theta)*sin(phi[j]) .+ center[1,2]
        rf[1 + (j-1)*Nth:Nth + (j-1)*Nth,3] = Rc*cos.(theta) .+ center[1,3]
        Theta[1 + (j-1)*Nth:Nth + (j-1)*Nth,1] = theta

    end
            
    field_d = InputFields.point_dipole(knorm, 1, rf, rd, dip_o) 
    field_t = field_d + field_sca(knorm, alpha, E_inc, rf, pos)
    
    field_d = reshape(field_d, 6, Nth*Nph)
    field_t = reshape(field_t, 6, Nth*Nph)
    
    S_0 = zeros(3,Nth*Nph)
    S_t = zeros(3,Nth*Nph)
    
    S_0[1,:] = real(field_d[2,:].*conj(field_d[6,:]) - field_d[3,:].*conj(field_d[5,:]))
    S_0[2,:] = real(field_d[3,:].*conj(field_d[4,:]) - field_d[1,:].*conj(field_d[6,:]))
    S_0[3,:] = real(field_d[1,:].*conj(field_d[5,:]) - field_d[2,:].*conj(field_d[4,:]))
    
    S_t[1,:] = real(field_t[2,:].*conj(field_t[6,:]) - field_t[3,:].*conj(field_t[5,:]))
    S_t[2,:] = real(field_t[3,:].*conj(field_t[4,:]) - field_t[1,:].*conj(field_t[6,:]))
    S_t[3,:] = real(field_t[1,:].*conj(field_t[5,:]) - field_t[2,:].*conj(field_t[4,:]))

    Sn_0 = sum(rf.*S_0',dims=2).*sin.(Theta)
    Sn_t = sum(rf.*S_t',dims=2).*sin.(Theta)
            
    P0 = sum(Sn_0)#/(4*pi)*Rc^2
    Pt = sum(Sn_t)#/(4*pi)*Rc^2

    LDOS = Pt/P0

    return LDOS

end
end
