#Cleaning data by removing unimportant columns and creating a DataFrame for classification process.
dataSubTrajectories = pd.DataFrame(A2FiltTraj, columns = ['t_user_id', 'transportation_mode', 'date_Start', 'flag' 
                               , 'minDis' ,'maxDis', 'meanDis', 'medianDis', 'stdDis'
                               , 'minSpeed' ,'maxSpeed', 'meanSpeed', 'medianSpeed', 'stdSpeed'
                               , 'minAcc' ,'maxAcc', 'meanAcc', 'medianAcc', 'stdAcc'
                              , 'minBrng' ,'maxBrng', 'meanBrng', 'medianBrng', 'stdBrng']  )

#dataSubTrajectories = pd.read_csv('dataFinal_A1.txt', delimiter = '\t')
dataSubTrajectories = dataSubTrajectories.drop('t_user_id', axis =1)
dataSubTrajectories = dataSubTrajectories.drop('date_Start', axis =1)
dataSubTrajectories = dataSubTrajectories.drop('flag', axis =1)

# This is the relabling method which is used in the hierarchical structure
# Coded on the bases of Q1 of this section
def relabel(node, labels):
    lb = []
    if(node == 1):
        for value in labels:
            if(value=='train'):
                lb.append(100)
            else:
                lb.append(-100)    
    elif(node == 2):
        for value in labels:
            if(value=='subway'):
                lb.append(-80)
            else:
                lb.append(80)
    elif(node == 3):
        for value in labels:
            if(value=='walk'):
                lb.append(-60)
            else:
                lb.append(60)
    elif(node == 4):
        for value in labels:
            if(value=='car'):
                lb.append(-40)
            else:
                lb.append(40)
    elif(node == 5):
        for value in labels:
            if(value=='taxi'):
                lb.append(-20)
            else:
                lb.append(20)
    return lb                

# This is the implementation of the proposed hierarchy above using Random Forest Classifier
# This hierarchy learns on the bases of the relabling method above  
def fitHierarchyRFC(trainData,trainLabels, modelDic):
    trainData1 = trainData.copy()
    label = relabel(1,trainLabels)   
    C1 = RandomForestClassifier().fit(trainData1, label)
    modelDic['C1'] = C1
    trainData1['oldLabels'] = trainLabels
    trainData1['newLabels'] = label
    grp1 = trainData1.groupby('newLabels')
    for grp in grp1: 
        if(grp[0] == -100):
            trainData2 = grp[1].iloc[:,0:20]
            trainLabels2 = grp[1]['oldLabels']
            labels2 = relabel(2,trainLabels2)
            C2 = RandomForestClassifier().fit(trainData2, labels2)
            modelDic['C2'] = C2
            trainData2['oldLabels'] = trainLabels2
            trainData2['newLabels'] = labels2
            grp2 = trainData2.groupby('newLabels')
            for grp in grp2:
                if(grp[0] == 80):
                    trainData3 = grp[1].iloc[:,0:20]
                    trainLabels3 = grp[1]['oldLabels']
                    labels3 = relabel(3,trainLabels3)
                    C3 = RandomForestClassifier().fit(trainData3, labels3)
                    modelDic['C3'] = C3
                    trainData3['oldLabels'] = trainLabels3
                    trainData3['newLabels'] = labels3
                    grp3 = trainData3.groupby('newLabels')
                    for grp in grp3:
                        if(grp[0] == 60):
                            trainData4 = grp[1].iloc[:,0:20]
                            trainLabels4 = grp[1]['oldLabels']
                            labels4 = relabel(4,trainLabels4)
                            C4 = RandomForestClassifier().fit(trainData4, labels4)
                            modelDic['C4'] = C4
                            trainData4['oldLabels'] = trainLabels4
                            trainData4['newLabels'] = labels4
                            grp4 = trainData4.groupby('newLabels')
                            for grp in grp4:
                                if(grp[0] == 40):
                                    trainData5 = grp[1].iloc[:,0:20]
                                    trainLabels5 = grp[1]['oldLabels']
                                    labels5 = relabel(5,trainLabels5)
                                    C5 = RandomForestClassifier().fit(trainData5, labels5)
                                    modelDic['C5'] = C5
    return modelDic

# This is the implementation of the predict method where you pass your learnt model and it gives you the predicted labels.
def predictHierarchy(testData, modelDic):
    testData1 = testData.copy()
    indexList = []
    predList = []
    frames = []
    predLabels = []
    pred = modelDic['C1'].predict(testData1)
    testData1['newLabels'] = pred
    grp1 = testData1.groupby('newLabels')
    for grp in grp1:
        if(grp[0] == -100):
            testData2 = grp[1].iloc[:,0:20]
            pred2 = modelDic['C2'].predict(testData2)
            testData2['newLabels'] = pred2
            grp2 = testData2.groupby('newLabels')
            for grp in grp2:
                #print('grp2 ->'+ str(grp[0]))
                if(grp[0] == 80):
                    testData3 = grp[1].iloc[:,0:20]
                    pred3 = modelDic['C3'].predict(testData3)
                    testData3['newLabels'] = pred3
                    grp3 = testData3.groupby('newLabels')
                    for grp in grp3:
                        #print('grp3 ->'+ str(grp[0]))
                        if(grp[0] == 60):
                            testData4 = grp[1].iloc[:,0:20]
                            pred4 = modelDic['C4'].predict(testData4)
                            testData4['newLabels'] = pred4
                            grp4 = testData4.groupby('newLabels')
                            for grp in grp4:
                                #print('grp4 ->'+ str(grp[0]))
                                if(grp[0] == 40):
                                    testData5 = grp[1].iloc[:,0:20]
                                    pred5 = modelDic['C5'].predict(testData5)
                                    testData5['newLabels'] = pred5
                                    grp5 = testData5.groupby('newLabels')
                                    for grp in grp5:
                                        #print('grp5 ->'+ str(grp[0]))
                                        if(grp[0] == 20):
                                            predList.append(grp[1].iloc[:,20])
                                        if(grp[0] == -20):
                                            predList.append(grp[1].iloc[:,20])    
                                if(grp[0] == -40):
                                    predList.append(grp[1].iloc[:,20])    
                        if(grp[0] == -60):
                            predList.append(grp[1].iloc[:,20])    
                if(grp[0] == -80):
                    predList.append(grp[1].iloc[:,20])            
        if(grp[0] == 100):  
            predList.append(grp[1].iloc[:,20]) 
    # Converting the predictions numberical value to corresponding class value i.e {100 -> 'train', -80 -> 'subway',
    # -60 -> 'walk', -40 -> 'car', -20 -> 'taxi', 20 -> 'bus'}if output of hierarchy was 100 then get 'train'. Similarliy
    # for other numerical values to their respective classes.
    for i in range(len(predList)):
        frames.append(pd.DataFrame(predList[i]))
    result = pd.concat(frames)
    predictions = result.sort_index(axis=0, ascending=True)
    for i in predictions['newLabels']:
        if(i==100):
            predLabels.append('train')
        if(i==-80):
            predLabels.append('subway')
        elif(i==-60):
            predLabels.append('walk')
        elif(i==-40):
            predLabels.append('car')
        elif(i==-20):
            predLabels.append('taxi')
        elif(i==20):
            predLabels.append('bus')
    return (predLabels)
