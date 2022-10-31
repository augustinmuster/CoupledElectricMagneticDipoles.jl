using Base
using LinearAlgebra
using PyCall
using LaTeXStrings
using DelimitedFiles
@pyimport matplotlib.pyplot as plt

data=readdlm("results.dat")

plt.plot(data[:,1],data[:,2],label="Auto CPU")
plt.plot(data[:,1],data[:,3],label="CPU1")
plt.plot(data[:,1],data[:,4],label="CPU2")
plt.plot(data[:,1],data[:,5],label="CPU3")
plt.plot(data[:,1],data[:,6],label="CPU4")
plt.plot(data[:,1],data[:,7],label="CPU5")
plt.plot(data[:,1],data[:,8],label="GPU")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Dipoles")
plt.ylabel("Solving Time (s)")
plt.tight_layout()
plt.legend()
plt.show()
plt.savefig("times.svg")

