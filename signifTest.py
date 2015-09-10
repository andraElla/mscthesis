__author__ = 'Andrada'
import numpy as np
import math

arr = []
abs_dif = []
for testid in range (0,10):
    abs_dif.append([])
    with open("trec_eval.9.0/test/logScikit/results/res" + str(testid) + ".txt","r") as f:
        for line in f:
            line_array = line.split()
            if(line_array[0] == 'ndcg_rel'):
                x = line_array[2]
                #print x
                break

    with open("trec_eval.9.0/test/MART/results/res" + str(testid) + ".txt","r") as f:
        for line in f:
            line_array = line.split()
            if(line_array[0] == 'ndcg_rel'):
                y = line_array[2]
                print y
                break

    dif = float(x) - float(y)
    dif = round(dif,4)

    abs_dif[testid].append(abs(dif))
    if(dif>=0):
        abs_dif[testid].append(1)
    else:
        abs_dif[testid].append(-1)

    arr.append(dif)

print arr
mean = np.mean(arr)
print mean
sd = np.std(arr)
print sd

SE = sd/math.sqrt(10)
print SE

#t-test
t = mean/SE

print t, "ttest"

#wilcoxon test

print abs_dif
abs_dif = np.asarray(abs_dif)
newarray = abs_dif[np.argsort(abs_dif[:,0])]
print newarray

s=0
for rank,item in enumerate(newarray):
    s += item[1] * (rank+1)
print abs(s)
