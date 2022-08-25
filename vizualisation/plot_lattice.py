import matplotlib.pyplot as plt
import numpy as np
import csv

x=[]
y=[]
z=[]

with open('../example/example_silicon_sphere/sphere_lattice.dat','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = '\t')
    i=0
    for row in plots:
        row2=row
        if(i%2==0):
            x.append(float(row2[0]))
            y.append(float(row2[1]))
            z.append(float(row2[2]))
        i=i+1

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(x,y,z)
plt.show()
