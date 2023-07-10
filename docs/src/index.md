# CoupledElectricMagneticDipoles.jl

Welcome on the documentation website for the coupled electric and magnetic dipoles (CEMD), or discrete dipoles approximation (DDA) method implementation in Julia! On this website, you will find all the description of the functions implemented in this library. Please, read carefully the next sections in order to be ready to code with CoupledElectricMagneticDipoles.jl!

## How to Use this Website and Learn the Library?
This website is divided in three parts. The first one is the home page, in which you will find all the informations to install, run and begin tu use the package. You will find as well a theory section containing the definitions of the CEMD (DDA) problems and all the mathematical expressions that are used to implement the functions. The second part is dedicated to the 7 submodules of the software:

- DDACore: Contains the functions to solve the coupled dipoles system of equations.
- Alphas: Small module to compute the polarizabilities.
- InputFields: Module with functions for input fields.
- PostProcessing: Module used to compute the scattering, absorption and extinction cross sections, as well as the scattered field and local density of states (LDOS).
- Forces: Module to compute optical forces between on dipoles.
- GreenTensors: Contains functions to compute the electric and magnetic or both green tensors.
- Geometries: Small library to discretize some 3D primitives.
- MieCoeff: Contains functions to compute the Mie coefficients.

Finally, the example tab gives three examples for the use of this library.

**Please have a look to the theory (next section) and the remaining of this page before starting**. When ready, have a look to the examples. Note that functions that ends in -`_e` are for systems made out of electric dipoles only and functions ending in -`_e_m` are for system with both electric and magnetic dipoles.

## Theory

Please read carefully the theory if you don't know about CEMD (DDA) method. 

```@raw html
<p>You can download the theory here: <a href="assets/CoupledElectricMagneticDipoles.pdf">Download PDF</a>.</p>
```

## Installing the Library

Since the library is not yet in the big registery of the julia package manager, you have to download the CoupledElectricMagneticDipoles in local in order to use it. In order to be able to run it, please run in your julia REPL:

```julia
]add "path to the library"
```
With this, the package will be precompiled and the dependancies are going to be installed. Then, you will be able to import the library, by using

```julia
using CoupledElectricMagneticDipoles
```
Since the library is installed and activated,we strongly recommand you to try to run one example to see if everything's works well.

## About the Units

Computers are dealing with number without units. Therefore, most of the inputs in the functions are dimensionless. For this, some renormalization may be needed. The most used are for:

- Positions ``\mathbf{r}`` (units of L) are multiplied by the wavenumber ``k`` (units of L⁻¹), in order to get dimensionless positions `kr`.
- Polarizabilities have units of volume (L³). We renormalize it by a factor ``k^3/4\pi`` (units of L⁻³) in oder to get dimensionless polarizabilities.
- Magnetic field is multiplied by the impedence ``Z`` in the medium. For this, we get a magnetic field that has the sames units as the electric field (E,H) becomes (E,ZH). 

Some functions are not using only dimensionless inputs. In all cases, **this is the user's responsability to send inputs that are coherents in terms of units**. 

## Running the Package in Parallel
Solving the DDA system of equations can be done with two different method that are parallel:

- With LAPACK LU decomposition on multiple CPU (the number of BLAS threads is set to be equal to the number of julia threads).
- With LU decomposition on the GPU (CUDA), by offloading the system on equations in the GPU memory.

To choose how to set the solver, see the DDACore module library.

On the other hand, some of the functions of the library are paralelized in the shared-memory scheme using the julia build-in parallelization. To use it, just set the number of threads to use when you run a julia script. Just like this:

```bash
julia --threads=8 foo.jl
```

## Importing the Package in Python

It is possible to run the package from python. See the [PyJulia package](https://pyjulia.readthedocs.io/en/latest/usage.html) and the examples in order to learn how to do this.

## Contact

To cite this software, please use: ....

Authors: Augustin Muster, Diego Romero Abujetas, Luis S. Froufe-Pérez.

Contact email: augustin@must-r.com

We are open to any comments, ideas or questions about this software. Don't hesitate to write us, but please be aware that we are not guaranteeing support.
