###########################################################################################
#######################  I> IMPORT THE LIBRARIES  ######################################### 
###########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm

############################################################################################
#######################  II> IMPORT THE DATASET & DATA PREPROCESSING   #####################
############################################################################################

dataset = pd.read_csv('E:\\DESK PROJECTS\\MACHINE LEARNING SUMMARY\\ML DATASETS\\Salary_Data.csv') 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

############################################################################################
###################### III> SPLIT DATASET INTO TRAIN AND TEST SETS  ########################
############################################################################################

from sklearn.model_selection import train_test_split                           
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

############################################################################################
####################### IV> FIT ML MODEL TO TRAINING SET ###################################
############################################################################################

from sklearn.linear_model import LinearRegression                 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

############################################################################################
####################### V> PREDICT TEST SET RESULTS ########################################
############################################################################################
    
y_pred = regressor.predict(X_test)                                

############################################################################################
####################### VI> VISUALIZE TRAINING SET RESULTS #################################
############################################################################################

plt.scatter(X_train, y_train, color = 'red')                       # training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')                         # test set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

