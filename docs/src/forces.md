 # Optical Forces

The Forces module allows to computes optical forces on electric and magnetic dipoles with deterministic input fields. The list of functions is below. Note that `N` is the number of dipoles in the system. 
 
 ## Functions Documentation

```@docs
CoupledElectricMagneticDipoles.Forces.force_e(knorm,kr,alpha_e_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
```

```@docs
CoupledElectricMagneticDipoles.Forces.force_e_m(k,kr,alpha_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
```

```@docs
CoupledElectricMagneticDipoles.Forces.force_e_m(knorm,kr,alpha_e_dl, alpha_m_dl, Ainv, e_0, dxe_0, dye_0, dze_0)
```

```@docs
CoupledElectricMagneticDipoles.Forces.orce_factor_gaussianbeams
```
