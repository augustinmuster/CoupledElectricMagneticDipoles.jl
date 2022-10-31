import matplotlib.pyplot as plt
import numpy as np
import csv
import numpy as np
import scipy.special as sps

def psi(n, x):
    return x * sps.spherical_jn(n, x, 0)

def diff_psi(n, x):
    return sps.spherical_jn(n, x, 0) + x * sps.spherical_jn(n, x, 1)

def xi(n, x):
    return x * (sps.spherical_jn(n, x, 0) + 1j * sps.spherical_yn(n, x, 0))

def diff_xi(n, x):
    return (sps.spherical_jn(n, x, 0) + 1j * sps.spherical_yn(n, x, 0)) + x * (sps.spherical_jn(n, x, 1) + 1j * sps.spherical_yn(n, x, 1))


def Mie_an(k0, R, m_p, m_bg, order):

    alpha = k0 * R * m_bg
    beta = k0 * R * m_p
    mt = m_p / m_bg

    return (mt * diff_psi(order, alpha) * psi(order, beta) - psi(order, alpha) * diff_psi(order,beta)) / (mt * diff_xi(order, alpha) * psi(order, beta) - xi(order, alpha) * diff_psi(order, beta))


def Mie_bn(k0, R, m_p, m_bg, order):

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




x=[]
yabs=[]
yext=[]
ysca=[]
yabs_mie=[]
yext_mie=[]
ysca_mie=[]
are=[]
aim=[]
a=230e-9
cst=np.pi*a*a
with open('../example/example_silicon_sphere/results.dat','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = '\t')
    for row in plots:
        x.append(float(row[0]))
        #x.append(a*2*np.pi/float(row[0]))
        k=2*np.pi/float(row[0])
        a1=Mie_an(k,230e-9,3.5+0.01j,1,1)
        b1=Mie_bn(k,230e-9,3.5+0.01j,1,1)
        const=2*np.pi/k/k

        yabs_mie.append(float((const*3*np.real(a1+b1))-float(const*3*(abs(a1)*abs(a1)+abs(b1)*abs(b1))))/cst)
        yext_mie.append(float(const*3*np.real(a1+b1))/cst)
        ysca_mie.append(float(const*3*(abs(a1)*abs(a1)+abs(b1)*abs(b1)))/cst)

        yext.append(float(row[1])/cst)
        yabs.append(float(row[2])/cst)
        ysca.append(float(row[3])/cst)

fig, ax=plt.subplots(2, sharex=True)
ax[0].scatter(x,yabs,marker='x',label="abs DDA")
ax[0].scatter(x,yext,marker='o',label="ext DDA")
ax[0].scatter(x,np.array(ysca),marker='^',label="sca DDA")
ax[0].plot(x,np.array(ysca_mie),marker='^',label="sca mie")
ax[0].plot(x,yabs_mie,marker='x',label="abs mie")
ax[0].plot(x,yext_mie,marker='o',label="ext mie")
ax[1].set_xlabel("wavelength")
ax[0].set_ylabel("normalized cross section")

ax[1].plot(x,np.array(yext)-np.array(yabs)-np.array(ysca))
fig.suptitle("Silicon sphere, radius="+str(a))
fig.legend()
plt.savefig("Silicon_cross_section")
plt.show()
