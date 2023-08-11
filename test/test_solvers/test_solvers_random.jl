using DelimitedFiles
using LinearSolve
using LinearSolveCUDA
include("../../src/DDA.jl")

#solving simple

solvers=["CPU", "GPU"]
res=zeros(14,length(solvers))
for i=1:14
    N=2^i
    A=rand(ComplexF64,N,N)
    b=rand(ComplexF64,N)
    res[i,:]=[@elapsed DDACore.solve_system(A,b,solvers[j],true) for j=1:length(solvers)]
    println(res[i,:])
end

writedlm("results_solving.dat",res)