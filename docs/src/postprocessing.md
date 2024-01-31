# Post Processing

The PostProcessing module allows you to process the results of a CEMD calculation. Mostly for:

- Computing cross sections and emission pattern. 
- Computing scattered fields.
- Computing local density of states (LDOS).

The list of functions and a description of how to use them is given in the following section. In general ``N`` represents the number of dipoles.

## Functions List and Documentation

### Functions for Cross Sections and Differential Emitted Power

```@docs
CoupledElectricMagneticDipoles.PostProcessing.compute_cross_sections_e(knorm,kr,e_inc,alpha,input_field;explicit_scattering=true,verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.compute_cross_sections_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field;explicit_scattering=true,verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.compute_cross_sections_e_m(knorm,kr,phi_inc,alpha_dl,input_field;explicit_scattering=true,verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.PostProcessing.diff_scattering_cross_section_e(knorm,kr,e_inc,alpha_e_dl,input_field,ur;verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.PostProcessing.diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_e_dl,alpha_m_dl,input_field,ur;verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.PostProcessing.diff_scattering_cross_section_e_m(knorm,kr,phi_inc,alpha_dl,input_field,ur;verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.PostProcessing.emission_pattern_e(kr,e_inc,alpha_e_dl,krf,krd;verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.PostProcessing.emission_pattern_e_m(kr,phi_inc,alpha_e_dl,alpha_m_dl,krf,krd,dip;verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.PostProcessing.emission_pattern_e_m(kr,phi_inc,alpha_dl,krf,krd,dip;verbose=true)
```
### Functions for Scattered Fields

```@docs
CoupledElectricMagneticDipoles.PostProcessing.field_sca_e(kr, alpha_e_dl, e_inc, krf; verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.field_sca_e_m(kr, alpha_e_dl, alpha_m_dl, e_inc, krf; verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.field_sca_e_m(kr, alpha_dl, e_inc, krf; verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.far_field_sca_e(kr, alpha_e_dl, e_inc, krf)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.far_field_sca_e_m(kr, alpha_e_dl, alpha_m_dl, e_inc, krf)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.far_field_sca_e_m(kr, alpha_dl, e_inc, krf)
```
### Functions for LDOS

```@docs
CoupledElectricMagneticDipoles.PostProcessing.ldos_e(kr, alpha_e_dl, Ainv, krd; dip=nothing; verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.ldos_e_m(kr, alpha_e_dl, alpha_m_dl, Ainv, krd; dip=nothing;verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.ldos_e_m(kr, alpha_dl, Ainv, krd; dip=nothing; verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.rad_ldos_e(kr,krd,p,dip;verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.rad_ldos_e_m(kr,krd,p,m,dip;verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.nonrad_ldos_e(p,e_inc,dip;verbose=true)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.nonrad_ldos_e_m(p,m,phi_inc,dip;verbose=true)
```
### Utilities Functions
```@docs
CoupledElectricMagneticDipoles.PostProcessing.compute_dipole_moment(alpha,phi_inc)
```
```@docs
CoupledElectricMagneticDipoles.PostProcessing.poynting_vector(phi)
```