# Inside make.jl

push!(LOAD_PATH,"../src/")

include("../src/green_tensors_e_m.jl")

using Documenter
using Base

makedocs(
         sitename = "DDAjulia",
         modules  = [green_tensor_e_m],
         pages=[
                "Home" => "index.md"
                "Green tensors" => "green.md"
               ])

deploydocs(;
    repo="github.com/augustinmuster/DDAjulia.wiki.git",
)