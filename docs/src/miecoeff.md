 # Mie Coefficients

MieCoeff is a small module for computing Mie Coefficients. It also contains some useful function to compute cross sections from the Mie theory. 
 
Expressions are taken in Craig F. Bohren, Donald R. Huffman, *Absorption and Scattering of Light by Small Particles* (1998).

 ## Functions Documentation


```@docs
CoupledElectricMagneticDipoles.MieCoeff.mie_an(ka, eps, eps_h; mu=1, mu_h=1, n=1)
```

```@docs
CoupledElectricMagneticDipoles.MieCoeff.mie_bn(ka, eps, eps_h; mu=1, mu_h=1, n=1)
```

```@docs
CoupledElectricMagneticDipoles.MieCoeff.mie_ab1(ka, eps, eps_h; mu=1, mu_h=1)
```

```@docs
CoupledElectricMagneticDipoles.MieCoeff.mie_extinction(ka,eps,eps_h;mu=1,mu_h=1,cutoff=20)
```

```@docs
CoupledElectricMagneticDipoles.MieCoeff.mie_absorption(ka,eps,eps_h;mu=1,mu_h=1,cutoff=20)cutoff=20)
```