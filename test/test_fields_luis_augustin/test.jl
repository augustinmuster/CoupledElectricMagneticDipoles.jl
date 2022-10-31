using DelimitedFiles
using LinearAlgebra

fields=open("fields.dat","r")
electric_field=open("electric_field.dat","r")

luis_fields=readdlm(fields,Float64)
augustin_fields=readdlm(electric_field,Float64)

diff=luis_fields-augustin_fields

for d in diff
    @assert abs(d)<1e-11
end