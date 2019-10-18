import pandas as pd
import numpy as np
import os

#Data From https://www.kaggle.com/austinreese/craigslist-carstrucks-data

#Pathway to CSV file
file = r"D:\Kaggle Datasets\Car Data\craigslistVehicles.csv"
#Pathway to the file outputs
outputFolder = r'D:\Python Codes\PricingModel\Data'

def MakeCategories(myList):
    """Creates a list of unique categories for a given column"""
    return list(set(myList))

def EncodeColumn(df,parameter):
    """Encodes a label parameter into a size (3,) array. This array carries the stats of mean,sd,and length
    of price entries for that column.
    Inputs:
        df - The pandas dataframe
        parameter - (str) The label for the desired parameter representing a label.
    Outputs:
        A dictionary where each key is a different category of the colun and each value is a (3,) array.
    """
    #Get a list of unique categories for the column
    categories = MakeCategories(df[parameter])
    catPrices = {}
    #Create a dictionary that will hold a list for each category
    for cat in categories:
        catPrices[cat] = []
    
    #Create a list of all the prices for a given category
    for i in range(len(df[parameter])):
        cat = df[parameter][i]
        price = df['price'][i]
        catPrices[cat].append(price)
    #Obtain the mean,std, and length of each list of prices for each category for a given column.
    catBox = {}
    for cat in categories:
        array = np.array(catPrices[cat])
        
        mean = np.average(array)
        sd = np.std(array)
        N = len(array)
        catBox[cat] = np.array([mean,sd,N])
    print("finished",parameter)
    return catBox

def BuildArrays(data,filePath):
    """Takes the dataframe and saves it to a standardized array. The descrete variables are 
    converted to a (3,) array by assigning the statistics taken from colbox to the labels.  
    The arrays are then saved to two files using np.save. The first file is the inputs "U" and the
    second file is the outputs "Z"
    Inputs:
        data- the dataframe
        filePath - The filepath to save the outputs
    """
    global continuousParams,labelParams,colBox
    U = []
    Z = []
    #Iterate through data in dataframe
    for i in range(len(data)):
        #Lowercase u represents a single row of the dataframe
        u = []
        #Obtain each column in the row and append it to u.
        for label in continuousParams:
            u.append(data[label][i])
        #Convert the labels to the (3,) array for each column
        #First, iterate through each column
        for label in labelParams:
            #Get the possible categories for that column
            dictionary = colBox[label]
            #Get the category for the given row and column
            key = data[label][i]
            #Obtrain the (3,) array for that category by referencing the dictionary
            value = dictionary[key].tolist()
            #Append each element of the dictionary to u.
            for v in value:
                u.append(v)
            
        #Store each row u into a big list U.
        U.append(u)
    
    #Save U and Z to arrays
    U = np.array(U)
    Z = data['price'].values
    
    #Set nan values to zero
    U = np.nan_to_num(U)
    U = np.where(U>100000,0,U)
    
    #Get the pathway for the Inputs. 
        #Save the array to the path.
    UPath = os.path.join(filePath,"Inputs")
    print("Built Arrays")
    np.save(UPath,U)
    print("Saved inputs")

    #Get the pathway for the Outputs. Create one if it does not exist.
        #Save the array to the path.      
    ZPath = os.path.join(filePath,"Outputs")

    np.save(ZPath,Z)
    print("Saved Prices")

#The parameters that are labels
labelParams = ['manufacturer','condition','cylinders','fuel','size','type','make','transmission','drive']
#The parameters that are continuous
continuousParams = ['year','odometer','lat','long']
usedParams = continuousParams+labelParams+['price']

############################Load and Clean data#############################################

#Use pandas to read csv file and load o dataframe.
df = pd.read_csv(file,usecols=usedParams)
print("Data Loaded")

#Remove rows that have a price of zero
df = df.drop(df[df.price==0].index)
#Remove rows that have an erroniously high price
df = df.drop(df[df.price>100000].index)
#Remove rows that have a nan value for price
df = df.drop(df[np.isnan(df.price)].index)
#Reset the indices so that the dropped rows are removed
df = df.reset_index()


##################################ShuffleData##################################################
import random
#Get length of dataFrame and the length of the desired training set (90% of total data)
length = len(df['year'])
trainLength = int(.9*length)

#Arange inegers from 0 to the length of the dataset. Shuffle the arangement.
indices = np.arange(length)
random.shuffle(indices)
#Break this set of shuffled indices to train and test indices
trainIndices = indices[0:trainLength]
testIndices = indices[trainLength:]
#Split the original dataframe into two copies by these random indices
    #These dataframes will be the training and testing datasets
trainData = pd.DataFrame(df,index=trainIndices,copy=True)
testData = pd.DataFrame(df,index=testIndices,copy=True)
#Reset the indices relative to their own datasets
trainData = trainData.reset_index()
testData = testData.reset_index()

#####################################Encode labeled data to be continuous####################
colBox = {}
for lp in labelParams:
    catBox = EncodeColumn(trainData,lp)
    colBox[lp] = catBox

del df

##################################Build Arrays#################################################
trainPath = os.path.join(outputFolder,"Train")
#Create the pathway if it does not exist.
if not os.path.exists(trainPath):
    os.makedirs(trainPath)
BuildArrays(trainData,trainPath)


testPath = os.path.join(outputFolder,"Test")
#Create the pathway if it does not exist.
if not os.path.exists(testPath):
    os.makedirs(testPath)
BuildArrays(trainData,testPath)
    

    
        
        
        




    
    
