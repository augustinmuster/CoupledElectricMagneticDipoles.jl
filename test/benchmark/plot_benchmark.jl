using LinearAlgebra
using DelimitedFiles
using PyCall
@pyimport matplotlib.pyplot as plt

#numbers of cores to plot
cores=[1,8,16]
Ncores=length(cores)

#other number parameters
Nfunctions=14
Ndip=6

#create array to store all data 
data=zeros(Ncores,Ndip,Nfunctions)
dips=zeros(Ndip)
labels=[]
for i=1:Ncores
    temp=readdlm("benchmark_results/res_"*string(cores[i])*"_cores.dat")
    println(temp)
    if i==1
        global dips=temp[2:end,1]
        global labels=temp[1,:]
    end
    global data[i,:,:]=temp[2:end,2:end]
end

##################### PLOTTING ####################
plt.rc("font", family="serif")
plt.rc("font", size=9)
linestyles=["-","--",":","-.",(0, (3, 1, 1, 1, 1, 1))]
#figure for the solving
# i.e. index 2-3
plt.rc("figure", figsize= [3.5, 2.5])
fig1, ax1=plt.subplots()
ax1.set_yscale("log")
ax1.set_xscale("log",base=2)
for i=1:Ncores
    ax1.plot(dips,data[i,:,9]*1e-9,color="black",linestyle=linestyles[i],label="CPU "*string(cores[i])*" cores")
end
ax1.plot(dips,data[1,:,10]*1e-9,color="red",linestyle="-",label="GPU")
ax1.set_xlabel("Number of dipoles")
ax1.set_ylabel("Time (s)")
plt.tight_layout()
plt.legend(frameon=false)
plt.savefig("inversion_time_e_m.svg")
plt.show()


#figure for the solving
# i.e. index 2-3
plt.rc("figure", figsize= [3.5, 2.5])
fig2, ax2=plt.subplots()
ax2.set_yscale("log")
ax2.set_xscale("log",base=2)
for i=1:Ncores
    ax2.plot(dips,data[i,:,1]*1e-9,color="black",linestyle=linestyles[i])
    ax2.plot(dips,data[i,:,2]*1e-9,color="red",linestyle=linestyles[i])
end
ax2.set_xlabel("Number of dipoles")
ax2.set_ylabel("Time (s)")
plt.tight_layout()
plt.legend(frameon=false)
plt.savefig("matrix_loading.svg")
plt.show()

