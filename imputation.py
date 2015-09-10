__author__ = 'Andrada'
import numpy as np
from numpy import genfromtxt
import math

V_init = genfromtxt("init-nofact.txt", delimiter="\t")

V_est = np.loadtxt("after-fact.txt",dtype="float",delimiter="\t")

#print V_init

#print V_est

new_array = np.zeros((5000,20))
print new_array.shape

for i in range (0,V_init.shape[0]):
    for j in range (0,V_init.shape[1]):
        if math.isnan(V_init[i,j]):
            new_array[i,j] = V_est[i,j]
        else:
            new_array[i,j] = V_init[i,j]

print new_array


file = open("after-replace.txt", "w")
for i in range (0, new_array.shape[0]):
    for j in range (0, new_array.shape[1]):
        #n = np.around(new_array[i,j], decimals=2)
        if j != 20:
            file.write(str(new_array[i,j]) + "\t")
        else:
            file.write(str(new_array[i,j]))

    file.write("\n")





