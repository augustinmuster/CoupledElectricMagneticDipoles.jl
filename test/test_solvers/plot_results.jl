using Base
using LinearAlgebra
using PyCall
using LaTeXStrings
using DelimitedFiles
@pyimport matplotlib.pyplot as plt

data=readdlm("results_solving.dat")

plt.plot([2^i for i=2:14],data[2:end,1],label="CPU LAPACK")
plt.plot([2^i for i=2:14],data[2:end,2],label="GPU CUSOLVER")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(L"N")
plt.ylabel(L"t\ \ (s)")
plt.tight_layout()
plt.legend()
plt.show()
plt.savefig("times.svg")

