import matplotlib.pyplot as plt
import numpy as np
import csv

x=[]
yabs=[]
yext=[]
ysca=[]
are=[]
aim=[]

a=230e-9
cst=np.pi*a*a
with open('../example/results.dat','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = '\t')
    for row in plots:
        x.append(float(row[0]))
        #x.append(a*2*np.pi/float(row[0]))
        yext.append(float(row[1])/cst)
        yabs.append(float(row[2])/cst)
        ysca.append(float(row[3])/cst)

plt.scatter(x,yabs,marker='x',label="abs DDA")
plt.scatter(x,yext,marker='o',label="ext DDA")
plt.scatter(x,np.array(ysca),marker='^',label="sca DDA")
#plt.scatter(x,np.array(ysca)/np.array(yext))
'''
plt.scatter(x,are,label="real")
plt.scatter(x,aim,label="imaginary")
'''
plt.xlabel("wavelength")
plt.ylabel("normalized cross section")
plt.ylabel("polarisbilities")
plt.title("Silicon sphere, radius="+str(a))
plt.legend()
plt.savefig("Silicon_cross_section")
plt.show()
