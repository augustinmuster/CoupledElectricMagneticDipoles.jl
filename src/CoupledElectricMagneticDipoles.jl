module CoupledElectricMagneticDipoles
    include("alpha.jl")
    include("DDA.jl")
    include("green_tensors_e_m.jl")
    include("input_fields.jl")
    include("mie_coeff.jl")
    include("processing.jl")
end