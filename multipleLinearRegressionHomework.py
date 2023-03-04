import numpy as np
import pandas as pd

#loading variable
data = pd.read_csv('./multipleLrHomeworkData.csv')

#test
print(data)
print('--------------------')

#data pre-processing
#encoder:  Categoric -> Numeric
from sklearn.preprocessing import LabelEncoder
#this one line encodes whole columns
dataEncoded = data.apply(LabelEncoder().fit_transform)
print('Verinin tamamının numerice çevrilmiş hali')
print(dataEncoded)
print('--------------------')

#now separate the first column to 3 columns with oneHotEncoding
#Bring the first column
outlook = dataEncoded[['outlook']]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
#Fitting and transforming process
outlook=ohe.fit_transform(outlook).toarray()
print('outlookun Ohe ile işlenmiş hali')
print(outlook)
print('--------------------')

outlookDF = pd.DataFrame(data = outlook, index = range(14), columns=['overcast','rainy','sunny'])
print("Outlook'un dataFrame hali")
print(outlookDF)
print('--------------------')

#Then concatenate dataFrames
lastData = pd.concat([outlookDF,data.iloc[:,1:3]],axis = 1)
lastData = pd.concat([dataEncoded.iloc[:,-2:],lastData], axis = 1)
print("Verinin son hali")
print(lastData)
print('--------------------')
#Now we can code the learning algorithm that predicts humidity with Multiple Linear Regression

#Seperating data as x_train,x_test,y_train,y_test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(lastData.iloc[:,:-1],lastData.iloc[:,-1:],test_size=0.33, random_state=0)

print("x_train şu şekildedir")
print(x_train)
print('--------------------')

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#Let's train the model
regressor.fit(x_train,y_train)

#Make predictions with x_test, then we will compare with y_test
y_pred = regressor.predict(x_test)

print("y_pred şu şekildedir")
print(y_pred)
print('--------------------')
print("y_test şu şekildedir")
print(y_test)
print('--------------------')

#Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values=lastData.iloc[:,:-1], axis=1 )
X_list = lastData.iloc[:,[0,1,2,3,4,5]].values
model = sm.OLS(lastData.iloc[:,-1:],X_list).fit()
print(model.summary())
print('--------------------')

#As a result of summary, we have to delete column x1 (first column)
lastData = lastData.iloc[:,1:]

#Take the summary again
X = np.append(arr = np.ones((14,1)).astype(int), values=lastData.iloc[:,:-1], axis=1)
X_list = lastData.iloc[:,[0,1,2,3,4]].values
model = sm.OLS(lastData.iloc[:,-1:],X_list).fit()
print("Backward elimination sonrası özeye tekrar bakalım")
print(model.summary())
print('--------------------')
# Now there is no column that's P value greater than 0.05

#Train the model again
x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

#Make predictions with new x_train and y_train
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)


#Comparing results
print("y_pred şu şekildedir")
print(y_pred)
print('--------------------')
print(y_test)
#As you can see accuracy of the model increased