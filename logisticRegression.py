import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#boy,kilo ve yaş bilgilerini kullanarak cinsiyet tahmini algoritması yapalım

#Data loading
data = pd.read_csv('./data.csv')


#Data slicing
hwa = data[['boy','kilo','yas']]
gender = data[['cinsiyet']]

#Separating the data as x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(hwa,gender,test_size=0.33,random_state=0)


#Scaling variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#Importing library
from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression(random_state=0)
#Train the model
logisticRegression.fit(x_train,y_train)


#Predictions
y_pred = logisticRegression.predict(x_test)
print('y_pred şu şekildedir')
print(y_pred)
print('------------------------')
print('y_test şu şekildedir')
print(y_test)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test,y_pred)
print(confusionMatrix)

