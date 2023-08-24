module CoupledElectricMagneticDipoles
    export Alphas, InputFields, DDACore, PostProcessing, Force, MieCoeff, GreenTensors, Geometries
    include("mie_coeff.jl")
    include("alpha.jl")
    include("green_tensors_e_m.jl")
    include("DDA.jl")
    include("input_fields.jl")
    include("processing.jl")
    include("geometries.jl") 
    include("forces.jl") 
    include("alpha.jl")      
end
