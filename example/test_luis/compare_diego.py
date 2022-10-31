import matplotlib.pyplot as plt
import numpy as np
import csv

x=[]
norm=[]
norm_diego=[]

a=1
cst=1
with open('electric_field.dat','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = '\t')
    i=0
    for row in plots:
        x.append(i)
        i=i+1
        norm.append(float(row[0]))

with open('nomrE_l0.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    for row in plots:
        norm_diego.append(float(row[0]))


fig, ax=plt.subplots(2, sharex=True)
ax[0].plot(x,norm,marker="x",label="norm Augustin")
ax[0].plot(x,norm_diego,label="norm Diego")

ax[1].set_xlabel("wavelength")
ax[0].set_ylabel("field")

ax[1].plot(x,np.zeros(len(x)))
fig.suptitle("Silicon sphere, radius="+str(a))
fig.legend()
plt.savefig("Silicon_cross_section")
plt.show()
