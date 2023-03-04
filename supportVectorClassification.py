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


#Importing the library
from sklearn.svm import SVC
#Buradaki kernel tipi probleme ve veri setine göre değişiklik gösterecektir
#dökümantasyondan bunun tiplerini araştırmalısınız
svc = SVC(kernel='linear')
#Train the model
svc.fit(x_train, y_train)


#Predictions
y_pred = svc.predict(x_test)
print('y_pred şu şekildedir')
print(y_pred)
print('---------------')
print('y_test şu şekildedir')
print(y_test)


#Confusion matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)

