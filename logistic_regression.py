import numpy as np
import math

#load normalized data
m = np.loadtxt("patientdata_icu_norm.data",dtype="float",delimiter=",")
print m.shape

cols = m.shape[1]
W = np.zeros(cols)
lr = 0.001
C = []
it = []
repeat = True
Sum = []
s = 0


#initialise list
folds = []

#create folds
for x in range (0,10):
    if x == 0:
        arr = m[::10,:]
    else:
        arr = m[x::10,:]
    np.random.shuffle(arr)
    folds.append(arr)


def update(W,X,lr,y,type,alg,feat):

    global s
    #x0 must be 1 - for the dot product
    X = np.insert(X,0,1)
    h = np.dot(X,W)

    if(alg == "logistic"):
        h = 1/(1 + pow(math.e,(-h)))
    err = h - y

    if type == "stochastic" :
        for j in range (0,cols-1):
            W[j] = W[j] - lr * err * X[j]
    else:
        s = s + err * X[feat]

def costFct(err,alg,y,h):

    if alg == "logistic":
        if(y == 1):
            return -(math.log(h,math.e))
        else:
            return -(math.log((1-h),math.e))

    return (pow(err,2))/2

def MSE(m,f,W,alg):
    mseSum = 0
    for i in range (0,m):
        X = np.insert(f[i][0:cols-2],0,1)
        h = np.dot(W,X)
        if alg == "logistic":
            h  = 1/(1 + pow(math.e,(-h)))
        err = h-f[i,cols-2]
        costC = costFct(err,alg,f[i,cols-2],h)

        mseSum = mseSum + costC
    return mseSum/m

def train(f,type,alg):

    global repeat,C,it,W,s,line,mse_vals

    #ignore patient ID column
    f = f[:,1:cols]

    #print f.shape
    m = f.shape[0]
    W = np.zeros(cols-1)
    #print cols
    C = []
    it = []
    repeat = True
    iterator = 0
    s = 0
    #mse_thresh = 0.456
    mse_thresh = 0.475

    # STOCHASTIC GRADIENT DESCENT
    while(repeat):
        for i in range (0,m):
            #update weights
            update(W, f[i][0:cols-2], lr, f[i,cols-2],type,alg,"")

       #calculate cost
        mse = MSE(m,f,W,alg)
        it.append(iterator)
        C.append(mse)
        print mse

        iterator = iterator + 1
        if(mse < mse_thresh):
            repeat = False
            mse_vals.append(mse)


def test(testset,alg,testid):

    global max_accuracy,best_thresh,besttest
    scores = []
    fname = "test" +  str(testid) + ".txt"
    file = open(fname, "w")
    fname_results = "mainlogistic/results" + str(testid) + ".txt"
    fout = open(fname_results, "w")

    testdata_ids =  testset[:,0]

    #ignore patient ID column
    testdata = testset[:,1:cols]

    #compute scores
    for index,item in enumerate(testdata):

        #x0 must be 1 - for the dot product
        X = item[0:cols-2]
        X = np.insert(X,0,1)

        score = np.dot(X,W)
        if alg == "logistic":
            score  = 1/(1 + pow(math.e,(-score)))
        scores.append(score)
        fout.write(str(int(testdata_ids[index]))+ "\t" + str(score) + "\n" )

    for thresh in scores:

        truePositive = 0
        falsePositive = 0
        trueNegative = 0
        falseNegative = 0
        k = 0
        icu_rec = 0
        for index,item in enumerate(scores):

            actual = testdata[index][cols-2]
            predicted = item

            if actual == 1:
                icu_rec = icu_rec + 1

            #take prev calculated score as a threshold
            if(predicted >= thresh):
                predicted = 1
            else:
                predicted = 0

            #if correctly predicted
            if predicted == actual:
                k = k + 1
                if(predicted == 1):
                    truePositive += 1
                else:
                    trueNegative += 1
            else:
                if actual == 0:
                    falsePositive += 1
                else:
                    falseNegative += 1


        #calculate 1-specificity
        spec = float(falsePositive)/(falsePositive + trueNegative)

        #calculate sensitivity
        sens = float(truePositive)/(truePositive + falseNegative)

        accuracy = float((k * 100)/testdata.shape[0])
        #print accuracy
        if (accuracy > max_accuracy) :
            max_accuracy = accuracy
            best_thresh = thresh
            besttest = testid



        file.write(str(spec)+ "\t" + str(sens)+ "\t" + str(thresh) + "\t" + str(accuracy)+ "%" + "\n" )

    file.close()


def foldstofile(traindata, testdata):

    fname_scikit_train = "scikitlogistic/dataset-train4Scikit" + str(i) + ".txt"
    fname_scikit_test = "scikitlogistic/dataset-test4Scikit" + str(i) + ".txt"

    fname_log_train = "mainlogistic/folds/train/dataset-train" + str(i) + ".txt"
    fname_log_test = "mainlogistic/folds/test/dataset-test" + str(i) + ".txt"
    fname_log_val = "mainlogistic/folds/validate/dataset-validate" + str(i) + ".txt"

    file_train = open(fname_log_train, "w")
    file_train_noformat = open(fname_scikit_train, "w")

    file_test = open(fname_log_test, "w")
    file_test_noformat = open(fname_scikit_test, "w")

    file_validate = open(fname_log_val, "w")

    col = testdata.shape[1]-1

    #features
    feats_train= traindata[:,1:col]
    feats_test= testdata[:,1:col]

    #label
    Y_train = traindata[:,col]
    Y_test = testdata[:,col]

    #patient ids
    id_train = traindata[:,0]
    id_test = testdata[:,0]

    for index,feat in enumerate(feats_train):
        line = ""
        line2 = ""
        line = str(int(Y_train[index])) + " " + "qid:1" + " "
        line2 += str(int(id_train[index])) + "\t"
        for j in range(0,feats_train.shape[1]):
            line += str(j+1) + ":" + str(feats_train[index,j]) + " "
            line2 += str(feats_train[index,j]) + "\t"

        line2 += str(int(Y_train[index])) + "\n"
        line += "#patientid = " + str(int(id_train[index])) + "\n"

        file_train_noformat.write(line2)

        if index < 3500:
            file_train.write(line)
        else:
            file_validate.write(line)



    for index,feat in enumerate(feats_test):
        line = ""
        line2 = ""
        line = str(int(Y_test[index])) + " " + "qid:1" + " "
        line2 += str(int(id_test[index])) + "\t"
        for j in range(0,feats_test.shape[1]):
            line += str(j+1) + ":" + str(feats_test[index,j]) + " "
            line2 += str(feats_test[index,j]) + "\t"

        line2 += str(int(Y_test[index])) + "\n"
        line += "#patientid = " + str(int(id_test[index])) + "\n"

        file_test_noformat.write(line2)
        file_test.write(line)




# MAIN loop
mse_vals = []
max_accuracy = 0
besttest = 0
for i in range (0,10):
    # i is for testing, train on the rest
    f = []
    testid = i
    for index,fold in enumerate(folds):
        if(index != i):
            if len(f):
                f = np.concatenate((f,fold),axis=0)
            else:
                f = fold

    traindata = f
    testdata = folds[i]

    train(traindata,"stochastic","logistic")
    print "testing on set ",i


    test(testdata,"logistic",testid)


    foldstofile(traindata,testdata)

print max_accuracy






