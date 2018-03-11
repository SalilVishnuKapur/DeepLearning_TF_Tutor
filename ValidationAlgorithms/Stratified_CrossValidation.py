from sklearn.model_selection import StratifiedKFold
def cvStratified(trainDataset, trainLabelset,typeOfClassification):
    if(typeOfClassification == 'RandomForestHierarchy'):
        cvRfHierarchyPerClass = []
        cvRfHierarchy = []
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(trainData, trainLabels)
        for train_index, test_index in skf.split(trainData, trainLabels):
                trainIndex = train_index.tolist()
                testIndex = test_index.tolist()
                result = fitHierarchyRFC(trainData.iloc[trainIndex], trainLabels.iloc[trainIndex], modelDic)
                predLabels = predictHierarchy(trainData.iloc[testIndex], result)
                cvRfHierarchy.append(accuracy_score(trainLabels.iloc[testIndex], predLabels))
                cvRfHierarchyPerClass.append(classwiseAccuracy(trainLabels.iloc[testIndex], predLabels))
        return (cvRfHierarchyPerClass,cvRfHierarchy)
    if(typeOfClassification == 'DecisionTreeHierarchy'):
        cvDtHierarchyPerClass = []
        cvDtHierarchy = []
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(trainData, trainLabels)
        for train_index, test_index in skf.split(trainData, trainLabels):
                trainIndex = train_index.tolist()
                testIndex = test_index.tolist()
                result = fitHierarchyDTC(trainData.iloc[trainIndex], trainLabels.iloc[trainIndex], modelDic)
                predLabels = predictHierarchy(trainData.iloc[testIndex], result)
                cvDtHierarchy.append(accuracy_score(trainLabels.iloc[testIndex], predLabels))
                cvDtHierarchyPerClass.append(classwiseAccuracy(trainLabels.iloc[testIndex], predLabels))
        return (cvDtHierarchyPerClass, cvDtHierarchy)        
    if(typeOfClassification == 'RandomForestFlat'):
        cvRfFlatPerClass = []
        cvRfFlat = []
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(trainData, trainLabels)
        for train_index, test_index in skf.split(trainData, trainLabels):
                trainIndex = train_index.tolist()
                testIndex = test_index.tolist()
                rfc = RandomForestClassifier()
                rfc.fit(trainData.iloc[trainIndex], trainLabels.iloc[trainIndex])
                predFlatRFC = rfc.predict(trainData.iloc[testIndex])
                cvRfFlat.append(accuracy_score(trainLabels.iloc[testIndex], predFlatRFC))
                cvRfFlatPerClass.append(classwiseAccuracy(trainLabels.iloc[testIndex], predFlatRFC))    
        return (cvRfFlatPerClass, cvRfFlat)
    if(typeOfClassification == 'DecisionTreeFlat'):
        cvDtFlatPerClass = [] 
        cvDtFlat = [] 
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(trainData, trainLabels)
        for train_index, test_index in skf.split(trainData, trainLabels):
                trainIndex = train_index.tolist()
                testIndex = test_index.tolist()
                dtc = DecisionTreeClassifier()
                dtc.fit(trainData.iloc[trainIndex], trainLabels.iloc[trainIndex])
                predFlatDTC = dtc.predict(trainData.iloc[testIndex])
                cvDtFlat.append(accuracy_score(trainLabels.iloc[testIndex], predFlatDTC))
                cvDtFlatPerClass.append(classwiseAccuracy(trainLabels.iloc[testIndex], predFlatDTC))
        return (cvDtFlatPerClass, cvDtFlat)        
