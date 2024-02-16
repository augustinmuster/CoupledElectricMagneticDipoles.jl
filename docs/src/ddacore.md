# DDA Core Functionalities

DDACore is the module allowing to solve DDA/CEMD problems (see the [Theory pdf](https://augustinmuster.github.io/assets/CoupledElectricMagneticDipoles_0_1_0.pdf) for definition of these problems). The problems can be solved either on the CPU (in parrallel with LAPACK) or on the GPU (With CUDA). The list of functions and how to use them is in the next section. Note that ``N`` denotes the number of point dipoles in the problem.

## Functions List and Documentation

### Main Solver Functions
```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e(kr,alpha_e_dl;input_field=nothing,solver="CPU",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e_m(kr,alpha_e_dl,alpha_m_dl;input_field=nothing,solver="CPU",verbose=true)
```

```@docs
CoupledElectricMagneticDipoles.DDACore.solve_DDA_e_m(kr,alpha_dl;input_field=nothing,solver="CPU",verbose=true)
```
### Utility Functions
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
CoupledElectricMagneticDipoles.DDACore.load_dda_matrix_e_m(kr,alpha_dl,verbose)
```