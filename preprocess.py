__author__ = 'Andrada'
import pandas as pd
import numpy as np
import nimfa

#function definitions

#calculate zscore for each item with the specified feature
def compute_zscore(arr,feature):
    list = []
    for item in arr:
        zscore = (item-mean[feature])/stdev[feature]
        list.append(zscore)
    return list


#options - what type of preprocessing
normalisation = 1
feature_scaling = 0
convert_cat_data = 0
imputation = 0


if normalisation == 1:
    #load original data
    m = np.loadtxt("patientdata-icu.data",dtype="float",delimiter="\t")
    file = open("patientdata_icu_norm.data", "w")

    if feature_scaling == 1:
        cols = m.shape[1] - 1
        min = m.min(axis=0)
        max = m.max(axis=0)
        for i in range(0,m.shape[0]):
            line = ""
            for j in range(0,m.shape[1]):
                if j == 0:
                    line = line + str(m[i][j]) + ","
                else:
                    new_item = (m[i][j] - min[j])/ (max[j] - min[j])
                    if j != cols:
                        line = line + str(new_item) + ","
                    else:
                        line = line + str(new_item)

            line = line + "\n"
            file.write(line)

    #do z-score normalisation
    else:
        print m.shape
        cols = m.shape[1] - 1
        feats = m[:,:cols]
        severity_scores = m[:,cols]

        #compute standard deviation and mean of features
        mean = []
        stdev = []
        for j in range (0,cols):
            mean.append(round(np.mean(m[:,j]),5))
            stdev.append(round(np.std(m[:,j]),5))


        #normalize - compute z-score using mean and standard dev
        for j in range (0,cols):
            #ignore patient ID column
            if j != 0:
                x = compute_zscore(m[:,j],j)
            else:
                x = m[:,0]
            m[:,j] = x

        print m.shape

        for i in range(0,m.shape[0]):
            line = ""
            for j in range(0,m.shape[1]):
                if j != cols:
                    line = line + str(m[i][j]) + ","
                else:
                    line = line + str(m[i][j])
            line = line + "\n"
            file.write(line)

    file.close()

else:

    #convert categorial data
    if convert_cat_data == 1:

        # read the data in
        df = pd.read_csv("data.csv")


        # take a look at the dataset
        #print df.head(100)

        # dummify
        vccmds_data = df['v_ccmds'].fillna( 'NA' )
        dummy_ranks_vccmds = pd.get_dummies(vccmds_data, prefix='v_ccmds')

        sepsis_data = df['sepsis'].fillna( 'NA' )
        dummy_ranks_sepsis = pd.get_dummies(sepsis_data, prefix='sepsis')

        sepsis_site_data = df['sepsis_site'].fillna( 'NA' )
        dummy_ranks_sepsis_site = pd.get_dummies(sepsis_site_data, prefix='sepsis_site')

        avpu_data = df['avpu'].fillna( 'NA' )
        dummy_ranks_avpu = pd.get_dummies(avpu_data, prefix='avpu')

        vabx_data = df['v_abx'].fillna( 'NA' )
        dummy_ranks_vabx = pd.get_dummies(vabx_data, prefix='v_abx')

        #print dummy_ranks_sepsis_site.head()

        # create a clean data frame
        cols_to_keep = ['id','sex','age','hco3','ph','bpdia','bpsys','lactate','bili','creatinine','sodium','hrate','platelets','rrate','spo2','temperature','urea','wcc','uvol1h','fio2_std','pf_ratio','sf_ratio','bpmap','sofa_score','news_score','icnarc_score','rxlimits','hsinus','icu_recommend']

        data = df[cols_to_keep].join(dummy_ranks_vccmds.ix[:, 'v_ccmds_0':])

        #print list(data.columns.values)
        datacols = list(data.columns.values)
        data = data[datacols].join(dummy_ranks_sepsis.ix[:, 'sepsis_1':])
        datacols = list(data.columns.values)
        print dummy_ranks_sepsis_site.keys()
        data = data[datacols].join(dummy_ranks_sepsis_site.ix[:, 'sepsis_site_1.0':])
        datacols = list(data.columns.values)
        data = data[datacols].join(dummy_ranks_avpu.ix[:, 'avpu_0':])
        datacols = list(data.columns.values)
        data = data[datacols].join(dummy_ranks_vabx.ix[:, 'v_abx_0':])

        print data.head()

        data.to_csv("dataout.csv", sep=',')

    #do imputation
    else:
        if imputation == 1:


            V = np.loadtxt("input-fact.txt",dtype="float",delimiter="\t")

            '''
            pmf = nimfa.Pmf(V, seed="random_vcol", rank=10, max_iter=3000)
            method_fit = pmf()

            nmf = nimfa.Nmf(V, seed="nndsvd", rank=10, max_iter=3000, update='euclidean',objective='fro')
            method_fit = nmf()

            bd = nimfa.Bd(V, seed="random_c", rank=10, max_iter=12, alpha=np.zeros((V.shape[0], 10)),
              beta=np.zeros((10, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
              n_w=np.zeros((10, 1)), n_h=np.zeros((10, 1)), n_sigma=False)
            method_fit = bd()

            '''
            #nmf = nimfa.Nsnmf(V, seed="nndsvd", rank=10, max_iter=3000, update='euclidean',objective='fro')
            #method_fit = nmf()

            bd = nimfa.Bd(V, seed="random_c", rank=10, max_iter=12, alpha=np.zeros((V.shape[0], 10)),
              beta=np.zeros((10, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
              n_w=np.zeros((10, 1)), n_h=np.zeros((10, 1)), n_sigma=False)
            method_fit = bd()

            W = method_fit.basis()
            #print('Basis matrix:\n%s' % W)

            H = method_fit.coef()
            #print('Mixture matrix:\n%s' % H)

            print('K-L divergence: %5.3f' % method_fit.distance(metric='kl'))

            print('Rss: %5.3f' % method_fit.fit.rss())
            print('Evar: %5.3f' % method_fit.fit.evar())
            print('Iterations: %d' % method_fit.n_iter)
            est = np.dot(W, H)
            #print('Target estimate:\n%s' % est)
            file = open("after-fact.txt", "w")
            print est.shape
            print est.shape[0]
            print est.shape[1]

            for i in range (0, est.shape[0]):

                for j in range (0, est.shape[1]):
                    n = np.around(est[i,j], decimals=2)
                    if j != 20:
                        file.write(str(n) + "\t")
                    else:
                        file.write(str(n))

                file.write("\n")



