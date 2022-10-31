# Inside make.jl

push!(LOAD_PATH,"../src/")

include("../src/green_tensors_e_m.jl")
include("../src/input_fields.jl")

using Documenter
using Base

makedocs(
         sitename = "CoupledElectricMagneticDipoles.jl",
         modules  = [green_tensor_e_m,input_fields],
         pages=[
                "Home" => "index.md"
                "Green tensors" => "green.md"
                "Input fields" => "input_fields.md"
               ])

deploydocs(;
    repo="https://github.com/augustinmuster/DDAjulia.wiki.git",
)