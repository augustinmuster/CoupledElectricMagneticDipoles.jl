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
                    sumsca=sumsca+dot(p[j,:],imag(GreenTensors.G_e(r[j,:],r[k,:],knorm))*p[k,:])
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
                sum_sca=sum_sca+real(transpose(p[j,:])*(imag(GreenTensors.G_e_renorm(knorm*r[j,:],knorm*r[i,:]))*conj(p[i,:])) + transpose(m[j,:])*(imag(GreenTensors.G_e_renorm(knorm*r[j,:],knorm*r[i,:]))*conj(m[i,:])))
                sum_sca=sum_sca+imag(-transpose(conj(p[i,:]))*imag(GreenTensors.G_m_renorm(knorm*r[i,:],knorm*r[j,:]))*m[j,:]    +   transpose(conj(p[j,:]))*imag(GreenTensors.G_m_renorm(knorm*r[i,:],knorm*r[j,:]))*m[i,:])
            end
        end

    end
    cst=2*pi/knorm^2*2/dot(e0,e0)
    if (explicit_scattering)
        return [2*pi/knorm cst*sum_ext cst*sum_abs 2*cst*sum_sca]
    else
        return [2*pi/knorm cst*real(sum_ext) cst*real(sum_abs) real(sum_ext-sum_abs)]
    end

end


@doc raw"""
    field_sca_e_m(kr, alpha_e_dl, alpha_m_dl, E_inc, krf)
It computes the scattered field from the ensamble of dipoles.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex array containing the dimensionless electric polarisability.
- `alpha_m_dl`: complex array containing the dimesnionless magnetic polarisability.
- `E_inc`: 2D complex array of size ``N\times 6`` with the incident field in the dipoles.
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
function field_sca_e_m(kr, alpha_e_dl, alpha_m_dl, E_inc, krf)

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

    E_inc = reshape(E_inc,n_particles*6,)
    field_r = G_tensor_fr*E_inc

    return reshape(field_r,n_r0,6)
            
end

@doc raw"""
    field_sca_e(kr, alpha_e_dl, alpha_m_dl, E_inc, krf)
It computes the scattered field from the ensamble of dipoles.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex array containing the dimensionless electric polarisability.
- `E_inc`: 2D complex array of size ``N\times 6`` with the incident field in the dipoles.
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
function field_sca_e(kr, alpha_e_dl, E_inc, krf)

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

    E_inc = reshape(E_inc,n_particles*3,)
    field_r = G_tensor_fr*E_inc

    return reshape(field_r,n_r0,3)
            
end

@doc raw"""
    ldos_e_m(kr, alpha_e_dl, alpha_m_dl, Ainv, krd; dip=nothing)
It Computes local density of states (LDOS) by the imaginary part of the returning field.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex array containing the dimensionless electric polarisability.
- `alpha_m_dl`: complex array containing the dimesnionless magnetic polarisability.
- `Ainv`: (inverse) DDA matrix ``[I - G*alpha]^(-1)``.
- `krd`: 2D float array of size ``Nd\times 3`` containing the dimentionless positions ``k\vec{r_d}`` where the LDOS is calculated.
- `dip`: integer defining the dipole moment (``dip = 1`` is an electric x-dipole, ``dip = 2`` an elctric y-dipole...) or float array of size 6 with the desired dipole moment of the dipole.  
#Outputs
- `LDOS` float array with the LDOS.

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
function ldos_e_m(kr, alpha_e_dl, alpha_m_dl, Ainv, krd; dip=nothing)

    n_particles = length(kr[:,1])
    n_dpos = length(krd[:,1])

    G_tensor = zeros(ComplexF64,n_particles*6,6) 
    G_tensor_fr = zeros(ComplexF64,6,n_particles*6)

    alp_e, alp_m = Alphas.dispatch_e_m(alpha_e_dl,alpha_m_dl,n_particles)

    if dip == nothing
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

    return LDOS
end

@doc raw"""
    ldos_e(kr, alpha_e_dl, Ainv, krd; dip=nothing)
It Computes local density of states (LDOS) by the imaginary part of the returning field.

#Arguments
- `kr`: 2D float array of size ``N\times 3`` containing the dimentionless positions ``k\vec{r}`` of each dipole.
- `alpha_e_dl`: complex array containing the dimensionless electric polarisability.
- `Ainv`: (inverse) DDA matrix ``[I - G*alpha]^(-1)``.
- `krd`: 2D float array of size ``Nd\times 3`` containing the dimentionless positions ``k\vec{r_d}`` where the LDOS is calculated.
- `dip`: integer defining the dipole moment (``dip = 1`` is an electric x-dipole, ``dip = 2`` an elctric y-dipole...) or float array of size 3 with the desired dipole moment of the dipole.  
#Outputs
- `LDOS` float array with the LDOS.

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
function ldos_e(kr, alpha_e_dl, Ainv, krd; dip=nothing)

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

    return LDOS
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
