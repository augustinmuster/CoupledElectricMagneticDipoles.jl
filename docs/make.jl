# Inside make.jl

push!(LOAD_PATH,"../src/")

include("../src/CoupledElectricMagneticDipoles.jl")


using Documenter
using Base

mathengine = Documenter.MathJax3()
makedocs(
         sitename = "CoupledElectricMagneticDipoles.jl",
         modules  = [CoupledElectricMagneticDipoles],
         pages=[
                "Home" => "index.md"
                "Modules" => [
                "DDACore" =>"ddacore.md"
                "Alphas" => "alphas.md"
                "InputFields" => "input_fields.md"
                "PostProcessing" => "postprocessing.md"
                "Forces" => "forces.md"
                "GreenTensors" => "green.md"
                "MieCoeff" =>"miecoeff.md"
                "Geometries" =>"geometries.md"
                ]
                "Examples"=> [
                    "PS Sphere" => "example_PS_sphere.md"
                    "Yagi-Uda Antenna" => "example_yagi_uda.md"
                    "LDOS Silver Particle" => "example_ldos_silver_np.md"
                    "Optical Trap"=> "example_force_gaussbeam_PS_sphere.md"
                ]
               ])
