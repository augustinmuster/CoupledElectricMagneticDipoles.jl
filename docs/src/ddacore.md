# DDA Core Functionalities

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e(knorm,r,alpha,input_field::Function;solver="LAPACK",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e(knorm,r,alpha;solver="JULIA",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e(kr,alpha_dl,input_field::Function;solver="LAPACK",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e(kr,alpha_dl;solver="JULIA",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e_m(knorm,r,alpha_e,alpha_m,input_field::Function;solver="LAPACK",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e_m(knorm,r,alpha_e,alpha_m;solver="LAPACK",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl,input_field::Function;solver="AUTO",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl;solver="AUTO",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_system(matrix,vector,solver,verbose)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.invert_system(matrix,solver,verbose)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.load_dda_matrix_e(kr,alpha_dl,verbose)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.load_dda_matrix_e_m(kr,alpha_e_dl,alpha_m_dl,verbose)
```