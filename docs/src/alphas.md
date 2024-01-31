# Polarizabilities

Alphas is a module to load polarizabilities. In this module, you will find functions to compute electric and magnetic polarizabilities of several objects. The modules also contains some functions to renormalize polarizabilities and to manage the multiple formats of the polarizabilities (dispatch). The list of functions (as well as how to use them) is given below. Note that ``N`` denotes the number of point dipoles in the problem.

## Format of the polarizabilities in the DDACore and PostProcessing functions
The functions of the DDACore and PostProcessing modules come with an automatic dispatch of the format of the polarizability. If we have ``N`` dipoles, it can be:
- a complex scalar
- a 1D complex array of size ``N``
- a ``3\times 3`` or ``6\times 6`` complex matrix.
- a 3D complex array of size ``N\times 3\times 3`` or ``N\times 6\times 6``



## Functions Documentation

```@docs
CoupledElectricMagneticDipoles.Alphas.alpha0_parallelepiped(lx,ly,lz,eps,eps_h)
```
```@docs
CoupledElectricMagneticDipoles.Alphas.alpha0_sphere(a,eps,eps_h)
```

```@docs
CoupledElectricMagneticDipoles.Alphas.alpha0_volume(V,eps,eps_h)
```

```@docs
CoupledElectricMagneticDipoles.Alphas.alpha_radiative(alpha0,knorm)
```

```@docs
CoupledElectricMagneticDipoles.Alphas.alpha_e_m_mie(ka,eps,eps_h)
```

```@docs
CoupledElectricMagneticDipoles.Alphas.dispatch_e(alpha_e_dl,n_particles)
```

```@docs
CoupledElectricMagneticDipoles.Alphas.dispatch_e_m(alpha_e_dl,alpha_m_dl,n_particles)
```

```@docs
CoupledElectricMagneticDipoles.Alphas.dispatch_e_m(alpha_dl,n_particles)
```

```@docs
CoupledElectricMagneticDipoles.Alphas.renorm_alpha(knorm,alpha)
```

```@docs
CoupledElectricMagneticDipoles.Alphas.denorm_alpha(knorm,alpha)
```