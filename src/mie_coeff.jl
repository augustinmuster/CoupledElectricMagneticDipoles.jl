module MieCoeff
export mie_an, mie_bn, mie_ab1,mie_scattering, mie_extinction, mie_absorption
#imports
using Base
using LinearAlgebra
using SpecialFunctions
#bessel functions
function spherical_jn(n,x,deriv)
    if deriv==0
        return sphericalbesselj(n, x)
    else
        return sphericalbesselj(n-1, x)-(n+1)/x*sphericalbesselj(n, x)
    end
end

function spherical_yn(n,x,deriv)
    if deriv==0
        return sphericalbessely(n, x)
    else
        return sphericalbessely(n-1, x)-(n+1)/x*sphericalbessely(n, x)
    end
end
# utlilities functions
function psi(n, x)
    return x * spherical_jn(n, x, 0)
end

function diff_psi(n, x)
    return spherical_jn(n, x, 0) + x * spherical_jn(n, x, 1)
end

function xi(n, x)
    return x * (spherical_jn(n, x, 0) + 1*im * spherical_yn(n, x, 0))
end

function diff_xi(n, x)
    return (spherical_jn(n, x, 0) + 1*im * spherical_yn(n, x, 0)) + x * (spherical_jn(n, x, 1) + 1*im * spherical_yn(n, x, 1))
end

@doc raw"""
    mie_an(ka, eps, eps_h; mu=1, mu_h=1, n=1)

Computes the `n`-th mie coefficient ``a_n`` of a sphere with size parameter `ka`,  dielectric permittivity `eps` and magnetic permeability `mu`, in a host medium with dielectric permittivity `eps_h` and magnetic permeability `mu_h`.
Returns a complex scalar.
"""
function mie_an(ka, eps, eps_h; mu=1, mu_h=1, n=1)
    mt = sqrt(eps*mu)/sqrt(eps_h*mu_h)
    alpha = ka 
    beta = ka *mt
    return (mu_h*mt * diff_psi(n, alpha) * psi(n, beta) - mu*psi(n, alpha) * diff_psi(n,beta)) / (mu_h*mt * diff_xi(n, alpha) *psi(n, beta) - mu*xi(n, alpha) * diff_psi(n, beta))
end

@doc raw"""
    mie_bn(ka, eps, eps_h; mu=1, mu_h=1, n=1)

Computes the `n`-th mie coefficient ``b_n`` of a sphere with size parameter `ka`,  dielectric permittivity `eps` and magnetic permeability `mu`, in a host medium with dielectric permittivity `eps_h` and magnetic permeability `mu_h`.
Returns a complex scalar. 
"""
function mie_bn(ka, eps, eps_h; mu=1, mu_h=1, n=1)
    mt = sqrt(eps*mu)/sqrt(eps_h*mu_h)
    alpha = ka 
    beta = ka * mt
    return (mu_h*mt * psi(n, alpha) * diff_psi(n, beta) - mu*diff_psi(n, alpha) * psi(n,beta)) / (mu_h*mt* xi(n, alpha) * diff_psi(n, beta) - mu*diff_xi(n, alpha) * psi(n, beta))
end

@doc raw"""
    mie_ab1(ka, eps, eps_h; mu=1, mu_h=1)

Computes the first mie coefficient ``a_1`` and ``b_1`` of a sphere with size parameter `ka`,  dielectric permittivity `eps` and magnetic permeability `mu`, in a host medium with dielectric permittivity `eps_h` and magnetic permeability `mu_h`.
Returns a tuple with two complex scalar, ``a_1`` and ``b_1``, respectively. 
"""
function mie_ab1(ka, eps, eps_h; mu=1, mu_h=1)
    mt = sqrt(eps*mu)/sqrt(eps_h*mu_h)
    alpha = ka 
    beta = ka *mt
    t1 = (sin(alpha) - sin(alpha) / alpha^2 + cos(alpha) / alpha) * (sin(beta) / beta - cos(beta))
    t2 = (sin(beta) - sin(beta) / beta^2 + cos(beta) / beta) * (sin(alpha) / alpha - cos(alpha))
    t3 = (sin(alpha) - sin(alpha) / alpha^2 + cos(alpha) / alpha + im*(-cos(alpha) + cos(alpha) / alpha^2 + sin(alpha) / alpha)) * (sin(beta) / beta - cos(beta))
    t4 = ((sin(alpha) / alpha - cos(alpha)) + im*(-cos(alpha) / alpha - sin(alpha))) * (sin(beta) - sin(beta) / beta^2 + cos(beta) / beta)
    a1 = (mu_h*mt * t1 - mu * t2) / (mu_h*mt * t3 - mu * t4)
    b1 = (mu_h*mt * t2 - mu * t1) / (mu_h*mt * t4 - mu * t3)
    return a1, b1
end

@doc raw"""
    mie_scattering(ka,eps,eps_h;mu=1, mu_h=1, cutoff=20)

Computes the scattering efficiency ``Q_{sca}`` of a sphere with size parameter `ka`, dielectric permittivity `eps` and magnetic permeability `mu`, in a host medium with dielectric permittivity `eps_h` and magnetic permeability `mu_h`. For this, we use the finite sum:
```math
Q_{sca} =\frac{2}{ka^2}\sum^{\text{cutoff}}_{n=1}\left(2n+1\right)\left(|a_n|^2+|b_n|^2\right)
```

where `cutoff` is set to 20 by default.

Returns a float.

"""
function mie_scattering(ka,eps,eps_h;mu=1,mu_h=1,cutoff=20)
    sum=0
    for i=1:cutoff
        sum=sum+(2*i+1)*(abs2(mie_an(ka, eps, eps_h; mu, mu_h, n=i))+abs2(mie_bn(ka, eps, eps_h; mu, mu_h, n=i)))
    end
    return 2/(ka)^2*sum
end

@doc raw"""
    mie_extinction(ka,eps,eps_h;cutoff=20)

Computes the extinction efficiency ``Q_{ext}`` of a sphere with size parameter `ka`, dielectric permittivity `eps` and magnetic permeability `mu`, in a host medium with dielectric permittivity `eps_h` and magnetic permeability `mu_h`. For this, we use the finite sum:

```math
Q_{ext} =\frac{2}{(ka)^2}\sum^{\text{cutoff}}_{n=1}\left(2n+1\right)Re\left(a_n+b_n\right)
```
where `cutoff` is set to 20 by default.

Returns a float.

"""
function mie_extinction(ka,eps,eps_h;mu=1,mu_h=1,cutoff=20)
    sum=0
    for i=1:cutoff
        sum=sum+(2*i+1)*real(mie_an(ka, eps, eps_h; mu, mu_h, n=i)+mie_bn(ka, eps, eps_h; mu, mu_h, n=i))
    end
    return 2/(ka)^2*sum
end


@doc raw"""
    mie_absorption(ka,eps,eps_h;mu=1,mu_h=1,cutoff=20)

Computes the extinction efficiency ``Q_{abs}`` of a sphere with size parameter `ka`, dielectric permittivity `eps` and magnetic permeability `mu`, in a host medium with dielectric permittivity `eps_h` and magnetic permeability `mu_h`. For this, we use:

```math
Q_{abs} =Q_{ext}-Q_{sca}
```
where corresponding sums are cut to `cutoff` which is set to 20 by default.

Returns a float.

"""
function mie_absorption(ka,eps,eps_h;mu=1,mu_h=1,cutoff=20)
    sum=0
    for i=1:cutoff
        an=mie_an(ka, eps, eps_h; mu, mu_h, n=i)
        bn=mie_bn(ka, eps, eps_h; mu, mu_h, n=i)
        sum=sum+(2*i+1)*(real(an+bn)-(abs2(an))+abs2(bn))
    end
    return 2/(ka)^2*sum
end

end
