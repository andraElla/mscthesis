__author__ = 'Andrada'
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import feature_selection as fs


#load train and test from file
testid = 0
data_train = np.loadtxt("scikitlogistic/dataset-train4Scikit" + str(testid) + ".txt",dtype="float",delimiter="\t")
data_test = np.loadtxt("scikitlogistic/dataset-test4Scikit" + str(testid) + ".txt",dtype="float",delimiter="\t")

print data_train.shape
print data_test.shape

cols = data_train.shape[1]-1

X_train = data_train[:,1:cols]
y_train = data_train[:,cols]

X_test = data_test[:,1:cols]
y_test = data_test[:,cols]

patientIDs = data_test[:,0]
predict = True
feature_selection = False

model = LogisticRegression()
model.fit(X_train, y_train)

# predict class labels for the test set
predicted = model.predict(X_test)
#print predicted

# generate class probabilities
probs = model.predict_proba(X_test)

print probs

file_res = open("scikitlogistic/results/resultsScikit" + str(testid) + ".txt", "w")
file_groundtruth = open("trec_eval.9.0/test/groundtruth/gt" + str(testid) + ".txt", "w")

for i in range (0,500):
   line = ""
   l = ""
   line += str(int(patientIDs[i])) + "\t" + str(probs[i,1]) + "\n"
   l +=  "1" + "\t" + "0" + "\t" + str(int(patientIDs[i])) + "\t" + str(int(y_test[i])) + "\n"
   #print line
   file_groundtruth.write(l)
   file_res.write(line)

# generate evaluation metrics
print "Acurracy ", metrics.accuracy_score(y_test, predicted)
print "AUC ",metrics.roc_auc_score(y_test, probs[:, 1])

if feature_selection == True:

   selection = fs.SelectKBest(k=20)
   X_features = selection.fit(X_train, y_train).transform(X_train)
   selected = selection.get_support()

   feats = ['sex','age','hco3','ph<7.3','ph>=7.3<7.4','ph>=7.4','bpdia','bpsys','lactate','bili','creatinine','sodium','hrate','hsinus','platelets','rrate','spo2','temperature','urea','wcc','uvol1h','fio2_std<=40','fio2_std>40<=80','fio2_std>80','pf_ratio','sf_ratio<=120','sf_ratio>120<=210','sf_ratio>210<=290','sf_ratio>290<=410','sf_ratio>410','bpmap','sofa_score','news_score','icnarc_score','rxlimits','v_ccmds_0.0','v_ccmds_1.0','v_ccmds_2.0','v_ccmds_3.0','v_ccmds_NA','sepsis_1.0','sepsis_2.0','sepsis_3.0','sepsis_4.0','sepsis_NA','sepsis_site_1.0','sepsis_site_2.0','sepsis_site_3.0','sepsis_site_4.0','sepsis_site_5.0','sepsis_site_6.0','sepsis_site_7.0','sepsis_site_8.0','sepsis_site_9.0','sepsis_site_10.0','sepsis_site_NA','avpu_0.0','avpu_1.0','avpu_2.0','avpu_3.0','avpu_4.0','avpu_5.0','avpu_NA','v_abx_0.0','v_abx_1.0','v_abx_2.0','v_abx_3.0','v_abx_5.0','v_abx_NA']

   scores = selection.scores_

   print "Top features"
   for index,item in enumerate(selected):
       if item:
           print feats[index],"\t" ,scores[index]
