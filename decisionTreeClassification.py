import pandas as pd
import numpy as np


#Data loading
data = pd.read_csv('./data.csv')


#Data slicing
hwa = data[['boy','kilo','yas']]
gender = data[['cinsiyet']]


#Seperating the data as x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(hwa,gender,test_size=0.33,random_state=0)


#Module importing
from sklearn.tree import DecisionTreeClassifier
#criterion, gini veya entropy parametrelerinden birini alır
#Dökümantasyonu inceleyerek hangi durumda hangisini kullanacağınızı öğrenebilirsiniz
decisionTreeClassifier = DecisionTreeClassifier(criterion='entropy')
#Train the model
decisionTreeClassifier.fit(x_train, y_train)


#Predictions
y_pred = decisionTreeClassifier.predict(x_test)
print('y_pred şu şekildedir')
print(y_pred)
print('-------------------')
print('y_test şu şekildedir')
print(y_test)
print('-------------------')


#Confusion matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)