# CoupledElectricMagneticDipoles.jl

Welcome on the documentation website for the coupled electric and magnetic dipoles (CEMD) method implementation in Julia! On this website, you will find all the description of the functions implemented in this library. Please read carefully the next sections in order to be ready to code with CoupledElectricMagneticDipoles.jl!

## How to use this website
This website is divided in three parts. The first one is the theory tab, in which you will be able to find the derivations of all the formulas that are used in this software. The, 7 tabs are dedicated to the 7 submodules of the software:

- DDACore: Contains the functions to solve the coupled dipoles system of equations.
- GreenTensors: Contains functions to compute the electric and magnetic or both green tensors.
- Alphas: Small library to compute the polarisabilities.
- Geometries: Small library to discretize some 3d objects or to generate some 3D random structures.
- MieCoeff: Contains functions to compute the Mie coefficients.
- PostProcessing: Small library used to compute the scattering, absorption and extinction cross sections, as well as the scattered field.
- InputFields: Small library with functions for input fields (plane waves,...).

Finally, the example tab gives two examples for the use of this library.

## Installing the library

Since the library is not yet in the big registery of the julia package manager, you have to download the CoupledElectricMagneticDipoles in local in order to use it. In order to be able to run it, put in your code the following two lines:

```julia
using Pkg
Pkg.activate("path to the library")
```

or

```julia
julia
]add "path to the library"
```
With this, the package will be precompiled and the dependancies are going to be installed. Then, you will be able to import the library, by using

```julia
import CoupledElectricMagneticDipoles as CEMD
```
Note that it is recommended to use this alias in order to lighten your code. Since the library is installed and activated, we strongly recommand you to try to run one exemple to see if everything's works well.

## About the Units

Since the numericals methods used in the CEMD software are inputing and outputing only numbers (and not physical quantities), this is of the responsability of the user to use consistant system of units. That means that the same unit of length should be used for positions, wavenumber and polarisability.

On the other hand, this code is using two different system of units

- For the only electric coupled dipole method.

- For the electric and magnetic coupled dipoles method. In this case, a renormalized system of units desribed in the Theory part is used in order to simplify the computations. The functions with the `_renorm` suffix are then aimed to be used for the electric and magnetic coupled dipoles method.

Please then be aware of the type of

## CPU Parallel/GPU running

## Contact

Author: Augustin Muster, augustin@must-r.com

Don't hesitate to ask questions if needed!
