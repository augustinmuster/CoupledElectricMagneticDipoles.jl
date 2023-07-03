 # Mie Coefficients

 MieCoeff is a small modules to compute Mie Coefficients, mostly to use it to compute polarizabilities. It also contains some useful function to compute cross sections from the Mie theory. 
 
Expressions are taken in Craig F. Bohren, Donald R. Huffman, *Absorption and Scattering of Light by Small Particles* (1998).

 ## Functions Documentation


```@docs
CoupledElectricMagneticDipoles.MieCoeff.mie_an(vac_knorm, a, eps, eps_h,n)
```

```@docs
CoupledElectricMagneticDipoles.MieCoeff.mie_bn(vac_knorm, a, eps, eps_h, n)
```

```@docs
CoupledElectricMagneticDipoles.MieCoeff.mie_scattering_cross_section(knorm,a,eps,eps_h;cutoff=20)
```

```@docs
CoupledElectricMagneticDipoles.MieCoeff.mie_extinction_cross_section(knorm,a,eps,eps_h;cutoff=20)
```

```@docs
CoupledElectricMagneticDipoles.MieCoeff.mie_absorption_cross_section(knorm,a,eps,eps_h;cutoff=20)
```