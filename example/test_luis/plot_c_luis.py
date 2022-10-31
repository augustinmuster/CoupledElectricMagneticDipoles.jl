import matplotlib.pyplot as plt
import numpy as np
import csv

x=[]
yabs=[]
yext=[]
ysca=[]
are=[]
aim=[]
cst=1
a=1
with open('refractive_index.dat','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = '\t')
    for row in plots:
        x.append(float(row[0]))


with open('results_luis.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = '\t')
    for row in plots:
        #x.append(a*2*np.pi/float(row[0]))
        yext.append(float(row[1])/cst)
        yabs.append(float(row[2])/cst)
        ysca.append(float(row[0])/cst)


fig, ax=plt.subplots(2, sharex=True)
ax[0].scatter(x,yabs,marker='x',label="abs DDA")
ax[0].scatter(x,yext,marker='o',label="ext DDA")
ax[0].scatter(x,np.array(ysca),marker='^',label="sca DDA")

ax[1].set_xlabel("wavelength")
ax[0].set_ylabel("normalized cross section")

ax[1].plot(x,np.array(yext)-np.array(yabs)-np.array(ysca))
fig.suptitle("Silicon sphere, radius="+str(a))
fig.legend()
plt.savefig("Silicon_cross_section")
plt.show()
