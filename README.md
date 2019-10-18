# PricePrediction
Predicts the Price of a car listing given information on a car using a random forest model.

# Installs
This code makes use of the following libraries.  You may install these libraries using pip install [library name]:

pandas,numpy,sklearn

# Overview
Given a dataset of used car listings scrapped from Craigslist, this program will predict the price of a given car by analyzing the characteristics of that car (such as milage, make, model, etc.).  The code is split into two parts, the first part, "Price Cleaning" will preprocess the data while "Pricing Model" will run train the model with a training data set and make predictions using a separate test dataset.

# Dataset
The data used for this program was taken from Kaggle at the following link:
https://www.kaggle.com/austinreese/craigslist-carstrucks-data

## Preprocessing
The data will comein the form of a csv file The code will convert the file to a dataframe and filter out erroneous values (such as abnormally high values and nan values).  Once filtered, it will shuffle each row of the file.  It will then split the dataframe into two separate ones - one for training and one for testing - with a 90/10 split.  

After splitting, each dataframe will undergo a process (described in the next section) that will convert string data to an array that can be read by the algorithm.  Once preprocessing is complete, both dataframes will be converted to numpy arrays called "Output" (containing the pricing data) and "Input" (containing all of the other useful parameters).  These four arrays will then be saved to a folder using the np.save() function.

The program will create two subfolders labeled "Train" and "Test" respectively.  Each subfolder will contain a .npy file labeled "Inputs" and "Outputs".

### Labeled data
This dataset contains two types of data: continueous and labeled data.  The continuous data can be represented as a float (such as milage) while labeled data is represented by a string (such as manufacturer).  When converting the label data to numeric data, special care must be taken to ensure that similar labels are closer to each other numerically.  For example, if we had four manufacturers (Toyota, Honda, Ford, and Audi), we could assign the numbers 1-4 for each brand.  However, since the order of the labels is arbitrary, Toyota could just as easily be assigned the number "3" as it well as the number "1".  

To avoid this problem, a spacial model is used.  Using the training dataset, the program will look at the statisitcs of each brand.  The average and standard deviation of the prices of each brand and the number of samples of that brand are used to build an array which represents the brand.  For example, the average price of a used honda is $8827, the standard deviation is 6861, and there are 22746 Hondas in the Training dataset.  As a result, the label "Honda" is replaced with the array (8827,6861,22746).  This ensures that labels that are similar to each other appear next to each other.

This process is repeated for each column that can be represented by a continuous label (such as drivetrain, condition, etc.

## Continuous Data
The city that the car was being sold in is also relevant to predicting the price. Since the number of possible cities is high relative to the number of samples, the cities can be represented instead as lat/long coordinates.  

## Random Forest
The "Pricing Model" will load in the data saved by the Preprocessing script and run a random forest consisting of 20 trees. Afterwards, it will create predictions for the test dataset and give error statisitcs for the predictions.  Note that the Labeled data is converted to an array baced off of the prices from the training dataset. The test dataset is kept entirely separate.
