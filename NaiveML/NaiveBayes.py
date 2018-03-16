from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

train_20news = fetch_20newsgroups(subset='train', shuffle = True, random_state =21)
test_20news = fetch_20newsgroups(subset='test', shuffle = True, random_state =21 )

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_20news.data)
X_test = vectorizer.transform(test_20news.data)

#dataTrain = csr_matrix(1*(X_train>0))
dataTrain = X_train.toarray()
targetTrain = train_20news.target

#dataTest = csr_matrix(1*(X_test>0))
dataTest = X_test.toarray()
targetTest = test_20news.target


def step1A(x,y):
    separated = {}
    for i, j in zip(dataTrain, targetTrain):
        if (j not in separated):
            separated[j] = []
        separated[j].append(i)
    return separated

def step1B(dataset):
    summaries = [((np.sum(attribute)+1.0)/(len(attribute)+2.0)) for attribute in zip(*dataset)]
    return summaries

def step1(x,y):
    separated = step1A(x,y)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = step1B(instances)
    return summaries


def step2(dictClasses, classSize, totalSize):
    sizes = np.ndarray((classSize,1), dtype = float )
    for i, v in dictClasses.items():
        sizes[i] = len(v)/totalSize
    return sizes

def step3(dictClasses, testData, testtarget, length, totalClasses):
    arr = np.ndarray((len(testData),len(dictClasses.keys())) , dtype = float) 
    for i in range(length):
        for j in range(totalClasses):
            arr[i][j] = np.product(np.trim_zeros(np.asarray((dataDic[targetTest[j]])*sizes[targetTest[j]]) * testData[i]))
    return arr

def step4(testOutputClasses, lengthTest):
    arr = np.ndarray((lengthTest,1) , dtype = float) 
    for i,row in zip(range(lengthTest), testOutputClasses):
        arr[i] = np.argmax(row)
    return arr

dataDic = step1(dataTrain, targetTrain)
lengthTrain = len(dataTrain)

lengthClasses = len(dataDic.keys())
sizes = step2(dataDic,lengthClasses ,lengthTrain)

lengthTest = len(dataTest)

arrFinal = step3(dataDic, dataTest, targetTest, lengthTest, lengthClasses)
predictions = step4(arrFinal, lengthTest)


print(accuracy_score(targetTest, predictions))
