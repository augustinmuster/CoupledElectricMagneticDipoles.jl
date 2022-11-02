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
                "Theory" => "theory.md"
                "Modules" => [
                "GreenTensors" => "green.md"
                "DDACore" =>"ddacore.md"
                "Alphas" => "alphas.md"
                "PostProcessing" => "postprocessing.md"
                "InputFields" => "input_fields.md"
                "MieCoeff" =>"miecoeff.md"
                ]
                "Examples"=> "examples.md"
               ])

deploydocs(
    repo="https://github.com/augustinmuster/CoupledElectricMagneticDipoles.wiki.git",
)