import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Data loading
data = pd.read_csv('./churnModelling.csv')


#Data slicing
x = data.iloc[:,3:13].values
y = data.iloc[:,-1].values


#Encoding
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
x[:,1] = labelEncoder.fit_transform(x[:,1])     #Encoding geography column
labelEncoder2 = LabelEncoder()
x[:,2] = labelEncoder2.fit_transform(x[:,2])    #Encoding gender column


#OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#ColumnsTransformer birden fazla kolonun aynı anda encode edilmesini sağlar
ohe = ColumnTransformer([('ohe',OneHotEncoder(dtype=float),[1])],remainder='passthrough')
x = ohe.fit_transform(x)
x = x[:,1:]


#Seperating the data as x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)


#Module importing
from xgboost import XGBClassifier
xgBoost = XGBClassifier()
#Train the model
xgBoost.fit(x_train,y_train)


#Predictions
y_pred = xgBoost.predict(x_test)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)