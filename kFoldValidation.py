#K-Fold işlemine kadar olan kodlar daha önceden gördüğümüz SVC kodlarıdır.

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
svc = SVC(kernel='linear',random_state=0)
svc.fit(x_train, y_train)


#Predictions
y_pred = svc.predict(x_test)


#Confusion matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)
print('---------------')


#K-Fold Validation
from sklearn.model_selection import cross_val_score
#estimator => kullanılan algoritma verilir (bu örneğe göre adı svc olan değişken)
#x ve y verilerini de veriri cvs'ye
#cv'ye ise fold sayısını yazarız
score = cross_val_score(estimator=svc, X=x_train, y=y_train, cv=4)
#Score'ların ortalamasını alarak bastıralım
print(score.mean())

#score.std() yazarsanız da standart sapmayı öğrenebilirsiniz
#Standart sapma ne kadar düşük olursa o kadar iyi