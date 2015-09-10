__author__ = 'Andrada'
import numpy as np

for testid in range (0,10):

    #scikit input

    m = np.loadtxt("scikitlogistic/results/resultsScikit" + str(testid) + ".txt",dtype="float",delimiter="\t")
    out = open("trec_eval.9.0/test/logScikit/input/in" + str(testid) + ".txt", "w")

    m = m[np.argsort(m[:,1])][::-1]


    for i in range (0, m.shape[0]):
            line = "1" + "\t" + "Q0" + "\t" + str(int(m[i,0])) + "\t" + str(i+1) + "\t" + str(m[i,1]) + "\t" + "STANDARD" + "\n"
            out.write(line)


    #ranklib input

    mart_scores = np.loadtxt("RankLib/results/MART/myscorefile" + str(testid) + ".txt",dtype="float",delimiter="\t")
    out_mart = open("trec_eval.9.0/test/MART/input/in"+ str(testid) + ".txt", "w")

    scores = mart_scores[:,2]
    patientIDs = []

    with open("RankLib/data/folds/test/dataset-test" + str(testid) + ".txt",'r') as f:
        for line in f:
            line_array = line.split()
            #patientid
            patientIDs.append(line_array[len(line_array)-1])

    newarray = np.column_stack((patientIDs,scores))

    newarray = newarray[np.argsort(newarray[:,1])][::-1]

    for i in range (0, mart_scores.shape[0]):
            line = "1" + "\t" + "Q0" + "\t" + str(newarray[i,0]) + "\t" + str(i+1) + "\t" + str(newarray[i,1]) + "\t" + "STANDARD" + "\n"
            out_mart.write(line)




