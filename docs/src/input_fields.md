# Input Fields

This module allows to compute input fields. It also implements derivatives of some of them.

See the [Theory pdf](https://augustinmuster.github.io/assets/CoupledElectricMagneticDipoles_0_1_0.pdf) to know more about the expression used to compute the input fields. In general ``N`` represents the number of input positions. 

## Functions Documentation

### Plane Waves
```@docs
CoupledElectricMagneticDipoles.InputFields.plane_wave_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.plane_wave_e_m
```

### Point Dipole Sources
```@docs
CoupledElectricMagneticDipoles.InputFields.point_dipole_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.point_dipole_e_m
```
### Gaussian, Hermite-Gaussian and Laguerre-Gaussian Beams
```@docs
CoupledElectricMagneticDipoles.InputFields.gaussian_beam_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.gaussian_beam_e_m
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
CoupledElectricMagneticDipoles.InputFields.d_gaussian_beam_e
```
```@docs
CoupledElectricMagneticDipoles.InputFields.d_gaussian_beam_e_m
```