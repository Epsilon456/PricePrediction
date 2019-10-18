import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RF
import os

#Folder path to where the Train and test folders are saved
filePath = r'D:\Python Codes\PricingModel\Data'

#Get the paths for the inputs (U) and outputs (Z) for training and testing set
path_UTrain = os.path.join(filePath,"Train","Inputs.npy")
path_UTest = os.path.join(filePath,"Test","Inputs.npy")
path_ZTrain = os.path.join(filePath,"Train","Outputs.npy")
path_ZTest = os.path.join(filePath,"Test","Outputs.npy")

#Load the data from the files
UTrain = np.load(path_UTrain)
UTest = np.load(path_UTest)
ZTrain = np.load(path_ZTrain)
ZTest = np.load(path_ZTest)


#Instantiate the random forest to have 20 trees and to run on the full number of processors 
    #on the system.  
rf = RF(n_estimators=20,verbose=2,n_jobs=-1)
#Train the Random forest with training data
rf.fit(UTrain,ZTrain)
#Make predictions
guesses = rf.predict(UTest)
#Score results using the test set. 
    #Score with R score
score = rf.score(UTest,ZTest)


print("R Score: ",score)
    


        