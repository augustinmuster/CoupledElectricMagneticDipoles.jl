# Example: Yagi-Uda Antenna Made Out of Silicon particles.

In this example, we will simulate an Yagi-Uda antenna made out of small silicon spheres (like in Krasnok et al., Opt. Express *20*, 20599-20604 (2012), click [here](https://opg.optica.org/oe/fulltext.cfm?uri=oe-20-18-20599&id=240909) for more information). For this, we will create a linear structure of small as follow: 


```@raw html
<img src="../assets/YU_design.png">
```

At the origin, we place a silicon sphere of 230nm called the reflector. After a bigger spacing of 355nm + 1800nm in which we will place the emitter (an oscillating dipole source aligned along the y-axis), we align on the z axis 10 silicon spheres with radius of 200nm, (center to center spacing: 400nm). We will then compute the total scattering cross section of this structure, as well as the differential cross section, in order to investigate the directionality of this antenna.  

We suppose that you already know the example of the PS sphere. So if you don't have a look to it before. If you want to now about the electric and magnetic DDA problem, have a look to the theory as well.

If you want to run this example, copy it or download it on the github (`example_yagi_uda.jl`) and run it using 

```bash
julia example_yagi_uda.jl

```
If you can, it is recommanded to run it in parallel, using the `--threads` option. 

# Setting the Structure
# Modelling silicon particles
```@raw html
<img src="../assets/mie_dipole_qsca.svg">
```
# Computing Emitted Power
```@raw html
<img src="../assets/diff_P.svg">
```

