# CoupledElectricMagneticDipoles.jl

Welcome to the documentation website for the coupled electric and magnetic dipoles (CEMD), or discrete dipoles approximation (DDA) method implementation in Julia! On this website, you will find all the descriptions of the functions implemented in this library. Please, read carefully the next sections to be ready to code with CoupledElectricMagneticDipoles.jl!

## How to Use this Website and Learn the Library?
This website is divided into three parts. The first one is the home page, in which you will find all the information to install, run, and begin to use the package. You will find a theory section containing the definitions of the CEMD (DDA) problems and all the mathematical expressions that are used to implement the functions. The second part is dedicated to the 8 submodules of the software:

- DDACore: Contains the functions to solve the coupled dipoles system of equations.
- Alphas: Small module to compute the polarizabilities.
- InputFields: Module with functions for input fields.
- PostProcessing: Module used to compute the scattering, absorption, and extinction cross sections, as well as the scattered field and local density of states (LDOS).
- Forces: Module to compute optical forces between dipoles.
- GreenTensors: Contains functions to compute the electric and magnetic or both green tensors.
- Geometries: Small library to discretize some 3D primitives.
- MieCoeff: Contains functions to compute the Mie coefficients.

Finally, the example tab gives four examples of the use of this library.

**Please have a look at the theory (next section) and the remaining of this page before starting**. When ready, have a look at the examples. Note that functions that end in -`_e` are for systems made out of electric dipoles only and functions ending in -`_e_m` are for systems with both electric and magnetic dipoles.

## Theory

Please read carefully the theory if you don't know about the CEMD (DDA) method. This pdf also provides you all the detailed expressions that are use to implement the library.

```@raw html
<p>You can download the theory document here: <a href="assets/CoupledElectricMagneticDipoles_0_1_0.pdf">Download PDF</a>.</p>
```

## Installing the Library

To install the library, type the following command in the Julia REPL:

```bash
] add CoupledElectricMagneticDipoles
```
With this, the package will be precompiled and the dependencies are going to be installed. Then, you will be able to import the library using

```julia
using CoupledElectricMagneticDipoles
```
Since the library is installed and activated, we strongly recommend you try to run one example to see if everything works well. You can as well run unit tests (located in `CoupledElectricMagneticDipoles/test/unit_tests.jl`) to test if everything is well installed. 

## Physical Units

Computers deal with numbers without units. Therefore, most of the inputs in the functions are dimensionless. For this, some renormalization may be needed. You will find all the details about units renormalization in the theory pdf, but the most used are for:

- Positions ``\mathbf{r}`` (units of L) are multiplied by the wave number ``k`` (units of L⁻¹), to get dimensionless positions `kr`.
- Polarizabilities have units of volume (L³). We renormalize it by a factor ``k^3/4\pi`` (units of L⁻³) to get dimensionless polarizabilities.
- Magnetic field is multiplied by the impedance ``Z`` in the medium. We then get a magnetic field that has the same units as the electric field. (E, H) becomes (E, ZH). 

Some functions do not use only dimensionless inputs. In all cases, **this is the user's responsibility to send inputs that are coherent in terms of units**. For example, it is important to input physical magnitudes with units of length all in the same units (don't put meter and micrometers together). It is recommended to look at the examples where the good practices are applied.

## Running the Package in Parallel
Solving the DDA system of equations can be done with two different parallel methods:

- With LAPACK LU decomposition on multiple CPUs (the number of BLAS threads is set to be equal to the number of Julia threads).
- With LU decomposition on the GPU (CUDA), by offloading the system on equations in the GPU memory.

In order to know how to choose the solver, please have a look to the DDACore module documentation.

In case you use the parallel CPU solver, just set the number of julia threads to use when you run the julia script, i.e. in this way:

```bash
julia --threads=8 foo.jl
```

## Importing the Package in Python

It is possible to run the package from Python. See the [PyJulia package](https://pyjulia.readthedocs.io/en/latest/usage.html). Note that the example of the PS sphere has also been worked out in Python. The associated Python script can be found in the examples folder.

## Contact

To cite this software, please use: CoupledElectricMagneticDipoles.jl - Julia modules for coupled electric and magnetic dipoles method for light scattering, and optical forces in three dimensions

Authors: Augustin Muster, Diego Romero Abujetas, Frank Scheffold, Luis S. Froufe-Pérez.

We are open to any comments, ideas, or questions about this software. Don't hesitate to write us, but please be aware that we are not guaranteeing support.
