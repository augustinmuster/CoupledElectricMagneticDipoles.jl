# Green's Tensors

GreenTensors is a module aimed to compute the renormalized (dimensionless) electric and magnetic Green's tensors. To learn more about the renormalization and the expressions that are used, please see the [Theory pdf](https://augustinmuster.github.io/assets/CoupledElectricMagneticDipoles_0_1_0.pdf).

## Functions Documentation

### Renormalized Green's Tensors
```@docs
CoupledElectricMagneticDipoles.GreenTensors.G_e_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.G_m_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.G_em_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.G_em_far_field_renorm
```
### Derivatives of the Renormalized Green's Tensors
```@docs
CoupledElectricMagneticDipoles.GreenTensors.dxG_e_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.dyG_e_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.dzG_e_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.dxG_m_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.dyG_m_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.dzG_m_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.dxG_em_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.dyG_em_renorm
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.dzG_em_renorm
```
### Utility Functions
```@docs
CoupledElectricMagneticDipoles.GreenTensors.denormalize_G_e
```
```@docs
CoupledElectricMagneticDipoles.GreenTensors.denormalize_G_m
```