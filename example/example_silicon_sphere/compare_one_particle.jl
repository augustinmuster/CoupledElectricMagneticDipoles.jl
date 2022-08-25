using Base
using LinearAlgebra
using Plots
include("../../mie_coeff.jl")

lambdas=LinRange(1000e-9,2000e-9,100)
print(lambdas[1])
cext=zeros(Float64,100)
csca=zeros(Float64,100)
cabs=zeros(Float64,100)
i=1
for lambda in lambdas
    knorm=2*pi/lambda
    ae=Mie_an(knorm,230e-9,3.5+0.01*im,1,1)
    am=Mie_bn(knorm,230e-9,3.5+0.01*im,1,1)
    cext[i]=2*pi/knorm^2*3*real(ae+am)
    csca[i]=2*pi/knorm^2*3*(abs2(ae)+abs2(am))
    cabs[i]=cext[i]-csca[i]
    global i=i+1
end

plot(lambdas,cext,label="extinction")
plot!(lambdas,csca,label="scattering")
plot!(lambdas,cabs,label="absorbtion")