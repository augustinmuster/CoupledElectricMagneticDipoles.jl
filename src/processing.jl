module PostProcessing
###########################
# IMPORTS
###########################
using Base
using LinearAlgebra
include("green_tensors_e_m.jl")
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
    point_dipole(knorm, E0_const, positions, rd, dip_o)
Function that calculated the electromagnetic field emitted by a point dipole with dipole moment 
```math
\mathrm{dip}_o = \dfrac{1}{\epsilon_0\epsilon} \overrightarrow{\mu}, \quad \mathrm{(see \ equation \ below)}
``` 

Imputs
- `knorm0` is the medium wavevector (scalar)
- `E0_const` is the field intensity (scalar). The modulus of the dipole moment is set to -epsilon_0*epsilon_m-, where -epsilon_0- and -epsilon_m- are the vacuum and medium permittivity, respectively.
- `position` contains the position at which the field is calculated (N x 3 matrix, where -N- is the number of points)
- `rd` is the position of the emitting source/dipole (1 x 3 vector)
- `dip_o` defined the nature of the dipole. If `dip_o` is a scalar then:
    - dip_o = 1 -> elecric dipole along x-axis
    - dip_o = 2 -> elecric dipole along y-axis
    - dip_o = 3 -> elecric dipole along z-axis
    - dip_o = 4 -> magnetic dipole along x-axis
    - dip_o = 5 -> magnetic dipole along y-axis
    - dip_o = 6 -> magnetic dipole along z-axis
if `dip_o` is a 6 x 1 vector then it specifies the dipole moment orentation of the source. 

Outputs
- `E_0i` is the electromagnetic field vector of the field at the requiered positions (6N x 1 vector)

Equation

```math
\mathbf{E}_{\mathbf{\mu}}(\mathbf{r}) = \omega^2 \mu \mu_0 G(\mathbf{r}, \mathbf{r}_0) \overrightarrow{\mu} = k^2 G(\mathbf{r}, \mathbf{r_0}) \dfrac{1}{\epsilon_0\epsilon} \overrightarrow{\mu}
```
with
```math
\mathrm{positions} = \mathbf{r}, \\
\mathrm{rd} = \mathbf{r}_0, \\
\mathrm{E}_{\mathrm{0i}} = \mathbf{E}_{\mathbf{\mu}}(\mathbf{r}).
``` 
"""
function point_dipole(knorm, E0_const, positions, rd, dip_o)
    
    N_points = length(positions[:,1])
    
    G_tensor = zeros(ComplexF64,N_points*6,6)
    G = zeros(ComplexF64,6,6) 

    if length(dip_o) == 1
        dip_oi = dip_o
        dip_o = zeros(6,1)
        dip_o[dip_oi] = 1
    else
        dip_o = dip_o/norm(dip_o) # Ensure that its modulus is equal to one
    end

    for i = 1:N_points  
        Ge, Gm = GreenTensors.G_em(positions[i,:],rd[1,:],knorm)   
        G[:,:] = [Ge im*Gm; -im*Gm Ge]
        G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , 1:6] = copy(G)
    end
	    
    E_0i = knorm^2*G_tensor*dip_o*E0_const

    return E_0i
        
end


@doc raw"""
    point_dipole_dl(knorm, E0_const, kpositions, krd, dip_o)
Function that calculated the electromagnetic field emitted by a point dipole with dipole moment 
```math
\mathrm{dip}_o = \dfrac{1}{\epsilon_0\epsilon} \overrightarrow{\mu}, \quad \mathrm{(see \ equation \ below)}
``` 

Imputs
- `knorm0` is the medium wavevector (scalar)
- `E0_const` is the field intensity (scalar). The modulus of the dipole moment is set to -epsilon_0*epsilon_m-, where -epsilon_0- and -epsilon_m- are the vacuum and medium permittivity, respectively.
- `kposition` contains the position (multiplied by the wavevector) at which the field is calculated (N x 3 matrix, where -N- is the number of points)
- `krd` is the position of the emitting source/dipole (multiplied by the wavevector) (1 x 3 vector)
- `dip_o` defined the nature of the dipole. If `dip_o` is a scalar then:
    - dip_o = 1 -> elecric dipole along x-axis
    - dip_o = 2 -> elecric dipole along y-axis
    - dip_o = 3 -> elecric dipole along z-axis
    - dip_o = 4 -> magnetic dipole along x-axis
    - dip_o = 5 -> magnetic dipole along y-axis
    - dip_o = 6 -> magnetic dipole along z-axis
if `dip_o` is a 6 x 1 vector then it specifies the dipole moment orentation of the source. 

Outputs
- `E_0i` is the electromagnetic field vector of the field at the requiered positions (6N x 1 vector)

Equation

```math
\mathbf{E}_{\mathbf{\mu}}(\mathbf{r}) =  k^2 G(k\mathbf{r}, k\mathbf{r_0}) \dfrac{1}{\epsilon_0\epsilon} \overrightarrow{\mu}
```
with
```math
\mathrm{kpositions} = \mathbf{kr}, \\
\mathrm{krd} = \mathbf{kr}_0, \\
\mathrm{E}_{\mathrm{0i}} = \mathbf{E}_{\mathbf{\mu}}(\mathbf{r}).
``` 
"""
function point_dipole_dl(knorm, E0_const, kpositions, krd, dip_o)
    
    N_points = length(kpositions[:,1])
    
    G_tensor = zeros(ComplexF64,N_points*6,6)
    G = zeros(ComplexF64,6,6)  

    if length(dip_o) == 1
        dip_oi = dip_o
        dip_o = zeros(6,1)
        dip_o[dip_oi] = 1
    else
        dip_o = dip_o/norm(dip_o) # Ensure that its modulus is equal to one
    end

    for i = 1:N_points     
        Ge, Gm = GreenTensors.G_em_renorm(kpositions[i,:],krd[1,:],knorm)   
        G[:,:] = k^3/(6*pi)*[Ge im*Gm; -im*Gm Ge]
        G_tensor[6 * (i-1) + 1:6 * (i-1) + 6 , 1:6] = copy(G)
    end
	    
    E_0i = knorm^3/(6*pi)*G_tensor*dip_o*E0_const

    return E_0i
        
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
 
    E_0i = point_dipole(knorm, 1, pos, rd, dip_o)
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
      
    E_0i = point_dipole(knorm, 1, pos, rd, dip_o)
    
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
        
    E_0i = point_dipole(knorm, 1, pos, rd, dip_o)
    
    E_inc = Ainv*E_0i
    
    rf = zeros(Nth*Nph,3)
    Theta = zeros(Nth*Nph,1)

    for j=1:Nph
        
        rf[1 + (j-1)*Nth:Nth + (j-1)*Nth,1] = Rc*sin.(theta)*cos(phi[j]) .+ center[1,1]
        rf[1 + (j-1)*Nth:Nth + (j-1)*Nth,2] = Rc*sin.(theta)*sin(phi[j]) .+ center[1,2]
        rf[1 + (j-1)*Nth:Nth + (j-1)*Nth,3] = Rc*cos.(theta) .+ center[1,3]
        Theta[1 + (j-1)*Nth:Nth + (j-1)*Nth,1] = theta

    end
            
    field_d = point_dipole(knorm, 1, rf, rd, dip_o) 
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
