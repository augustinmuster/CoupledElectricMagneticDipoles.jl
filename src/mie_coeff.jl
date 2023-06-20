module MieCoeff
#code to compute the mie coefficients author Augustin Muster (translate from python code of Diego Romero Abujetas)
#imports
using Base
using LinearAlgebra
using SpecialFunctions

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

#*************************************************
#MIE COEFFICIENT an
#INPUTS: k0 = wavevector in vacuum, R = particle radius, m_p = particle refractive index, m_bg = background refractive index, order = harmonic number order (integrer number)
#OUTPUT: mie coefficient
#*************************************************
function Mie_an(k0, R, m_p, m_bg, order)

    alpha = k0 * R * m_bg
    beta = k0 * R * m_p
    mt = m_p / m_bg

    return (mt * diff_psi(order, alpha) * psi(order, beta) - psi(order, alpha) * diff_psi(order,beta)) / (mt * diff_xi(order, alpha) * psi(order, beta) - xi(order, alpha) * diff_psi(order, beta))
end

#*************************************************
#MIE COEFFICIENT an
#INPUTS: k0 = wavevector in vacuum, R = particle radius, m_p = particle refractive index, m_bg = background refractive index, order = harmonic number order (integrer number)
#OUTPUT: mie coefficient
#*************************************************
function Mie_bn(k0, R, m_p, m_bg, order)

    """
    :param k0 = wavevector in vacuum
    :param R = particle radius
    :param m_p = particle refractive index
    :param m_bg = background refractive index
    :param order = harmonic number order (integrer number)
    """

    alpha = k0 * R * m_bg
    beta = k0 * R * m_p
    mt = m_p / m_bg

    return (mt * psi(order, alpha) * diff_psi(order, beta) - diff_psi(order, alpha) * psi(order,beta)) / (mt * xi(order, alpha) * diff_psi(order, beta) - diff_xi(order, alpha) * psi(order, beta))
end


function mie_scattering_cross_section(knorm0,a,eps,eps_h;cutoff=50)
    sum=0
    for i=1:cutoff
        sum=sum+(2*i+1)*(abs2(Mie_an(knorm0, a, sqrt(eps), sqrt(eps_h), i))+abs2(Mie_bn(knorm0, a, sqrt(eps), sqrt(eps_h), i)))
    end
    return 2*pi/(knorm0)^2/eps_h*sum
end
end