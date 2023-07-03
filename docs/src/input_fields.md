# Input Fields

This module allows to compute input fields. It also implements derivatives of some of the beams. See the theory part to know more about the expression used tu compute the input fields. In general ``N`` represents the number of positions inputed in the function. 

## Functions Documentation

### Plane Waves
```@docs
CoupledElectricMagneticDipoles.InputFields.plane_wave_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.plane_wave_e_m
```

### Point Dipoles Sources
```@docs
CoupledElectricMagneticDipoles.InputFields.point_dipole_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.point_dipole_e_m
```
### Gaussian, Hermite-Gaussian and Laguerre-Gaussian Beams
```@docs
CoupledElectricMagneticDipoles.InputFields.gauss_beam_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.ghermite_beam_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.glaguerre_beam_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.gauss_beam_e_m
```
```@docs
CoupledElectricMagneticDipoles.InputFields.ghermite_beam_e_m
```
```@docs
CoupledElectricMagneticDipoles.InputFields.glaguerre_beam_e_m
```
### Derivatives of the Beams
```@docs
CoupledElectricMagneticDipoles.InputFields.d_plane_wave_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.d_plane_wave_e_m
```
```@docs
CoupledElectricMagneticDipoles.InputFields.d_point_dipole_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.d_point_dipole_e_m
```
```@docs
CoupledElectricMagneticDipoles.InputFields.d_gauss_beam_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.d_gauss_beam_e_m
```