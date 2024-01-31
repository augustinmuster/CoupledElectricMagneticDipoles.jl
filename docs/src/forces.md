 # Optical Forces

The Forces module allows to computes optical forces on electric and magnetic dipoles with deterministic input fields. The list of functions is given below. Note that `N` is the number of dipoles in the system. 
 
 ## Functions Documentation

```@docs
CoupledElectricMagneticDipoles.Forces.force_e
```

```@docs
CoupledElectricMagneticDipoles.Forces.force_e_m(kr,alpha_e_dl, alpha_m_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
```

```@docs
CoupledElectricMagneticDipoles.Forces.force_e_m(kr,alpha_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
```

```@docs
CoupledElectricMagneticDipoles.Forces.force_factor_gaussianbeams
```
