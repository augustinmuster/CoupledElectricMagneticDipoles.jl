# DDA Core Functionalities

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e(kr,alpha_dl;input_field=nothing,solver="CPU",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl;input_field=nothing,solver="CPU",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e_m(kr,alpha_dl;input_field=nothing,solver="CPU",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_system(A,b,solver,verbose)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.load_dda_matrix_e(kr,alpha_dl,verbose)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.load_dda_matrix_e_m(kr,alpha_e_dl,alpha_m_dl,verbose)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.load_dda_matrix_e_m(kr,alpha_tensor,verbose)
```