#code to compute the mie coefficients author Augustin Muster (translate from python code of Diego Romero Abujetas)
#imports
using Base
using LinearAlgebra
using SpecialFunctions
using PyCall
@pyimport scipy.special as sps

# utlilities functions
function psi(n, x)
    return x * sps.spherical_jn(n, x, 0)
end

function diff_psi(n, x)
    return sps.spherical_jn(n, x, 0) + x * sps.spherical_jn(n, x, 1)
end

function xi(n, x)
    return x * (sps.spherical_jn(n, x, 0) + 1*im * sps.spherical_yn(n, x, 0))
end

function diff_xi(n, x)
    return (sps.spherical_jn(n, x, 0) + 1*im * sps.spherical_yn(n, x, 0)) + x * (sps.spherical_jn(n, x, 1) + 1*im * sps.spherical_yn(n, x, 1))
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
