__author__ = 'Andrada'
import numpy as np
import pylab as plt

m = np.loadtxt("checkOutliers.txt",dtype="float",delimiter="\t")
print m.shape

cols = m.shape[1]

feats = ['sex','age',	'hco3',	'ph', 'bpdia', 'bpsys',	'lactate', 'bili', 'creatinine', 'sodium', 'hrate','hsinus', 'platelets','rrate','spo2', 'temperature', 'urea', 'wcc', 'uvol1h','fio2_std', 'pf_ratio', 'sf_ratio', 'bpmap','sofa_score','news_score','icnarc_score']

it = range(m.shape[0])

for i in range (0,cols):
    col = m[:,i]
    fig = plt.gcf()
    fig.canvas.set_window_title(feats[i])
    plt.plot(it,col,'ro')
    plt.show()


