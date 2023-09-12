# Geometries library

Geometries is a small modules allowing to discretize primitives in small cubes that can be later used in the DDA. The two primitives that are proposed are a sphere and a cube. The description of the two functions are given in the next section. Note that the discretization of the sphere implements anti-aliasing (see [Wikipedia page for anti-aliasing](https://en.wikipedia.org/wiki/Spatial_anti-aliasing) or the PS sphere example).


## Functions Documentation
```@docs
CoupledElectricMagneticDipoles.Geometries.discretize_sphere
```

```@docs
CoupledElectricMagneticDipoles.Geometries.discretize_cube(L,N)
```
