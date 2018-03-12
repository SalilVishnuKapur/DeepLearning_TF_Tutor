import itertools
import numpy as np

def customizedCrossValidation(model,data,target,cv, ratio):
    CCV = np.ndarray(shape= (1,cv), dtype=float)
    numRows = int(len(target))
    data = np.hstack((data,target.reshape(150,1)))
    for i in range(cv):
        # shuffle
        np.random.shuffle(data)
        limit = int(ratio*len(target))
        # train test split  
        X_train = data[0:limit,[0,1,2,3]]
        Y_train = data[0:limit,[4]]
        X_test = data[limit:numRows,[0,1,2,3]]
        Y_test = data[limit:numRows,[4]]
        # train model 
        trainModel = model.fit(X_train, Y_train)
        y_predicted = trainModel.predict(X_test)
        count = len(y_predicted)
        success = 0
        for j in range(count):
            if(Y_test[j] == y_predicted[j]):
                success +=1
        CCV[0][i] = success/count        
    return CCV
