# DDA Core Functionalities

DDACore is the modules allowing to solve CEMD problems (see supplemetary informations in the "Theory" tab for definition of the problem). The problems can be solved either on the CPU (in parrallel with LAPACK) or on the GPU (With CUDA). The list of functions and how to use them is in the next section. 

## Functions List and Documentation
```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e(kr,alpha_e_dl;input_field=nothing,solver="CPU",verbose=true)
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
CoupledElectricMagneticDipoles.DDACore.load_dda_matrix_e(kr,alpha_e_dl,verbose)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.load_dda_matrix_e_m(kr,alpha_e_dl,alpha_m_dl,verbose)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.load_dda_matrix_e_m(kr,alpha_tensor,verbose)
```