import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

trainData = np.loadtxt('wine.train',dtype = float, delimiter=',')
testData = np.loadtxt('wine.test',dtype = float, delimiter=',')

# SVM, MLP, or RF ?

classifierMLP = MLPClassifier(hidden_layer_sizes= (20,10), max_iter=10)



classifierSVM = svm.SVC(C=2.0, cache_size=400, class_weight=None, coef0=0.0,
                       decision_function_shape='ovo', gamma=1.0, kernel='linear',
                       max_iter=500, probability=False, random_state=0, shrinking=True,
                       tol=0.001, verbose=False)


classifierRF = RandomForestClassifier(n_estimators=50, bootstrap=True, max_features = 'log2',
                                    max_depth = 5,min_samples_split =3, n_jobs = -1,
                                    random_state = 0, verbose = 0,oob_score = True)



X_train = trainData[:, 1:13]
Y_train = trainData[:, 0]
X_test = testData[:, 1:13]


cvMLP =np.array((np.average(cross_val_score(classifierMLP, X_train, Y_train, cv =3)),
     np.average(cross_val_score(classifierMLP, X_train, Y_train, cv =5)),
     np.average(cross_val_score(classifierMLP, X_train, Y_train, cv =10)),
     np.average(cross_val_score(classifierMLP, X_train, Y_train, cv =20))), dtype = float)

cvSVM =np.array((np.average(cross_val_score(classifierSVM, X_train, Y_train, cv =3)),
     np.average(cross_val_score(classifierSVM, X_train, Y_train, cv =5)),
     np.average(cross_val_score(classifierSVM, X_train, Y_train, cv =10)),
     np.average(cross_val_score(classifierSVM, X_train, Y_train, cv =20))), dtype = float)

cvRF =np.array((np.average(cross_val_score(classifierRF, X_train, Y_train, cv =3)),
     np.average(cross_val_score(classifierRF, X_train, Y_train, cv =5)),
     np.average(cross_val_score(classifierRF, X_train, Y_train, cv =10)),
     np.average(cross_val_score(classifierRF, X_train, Y_train, cv =20))), dtype = float)
