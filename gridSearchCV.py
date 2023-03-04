#Optimum parametreyi bulan bu algoritmayı istediğiniz herhangi bir modelde deneyebilirsiniz
#Şimdi biz KNN'de deneyelim. Aşağıdaki kodların çoğu daha önce gördüğünüz kodlar olacak


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
from sklearn.neighbors import KNeighborsClassifier
#GridSearchCV ile n_neighbors ve metric özelliklerini optimize etmeye çalışalım
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(x_train,y_train)


#Predictions
y_pred = knn.predict(x_test)


#Confusion matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)
print('----------------')


#GridSearch
#GridSearchCV'nin sonundaki cv cross validation dediğimiz k-fold dahil işlemdir
from sklearn.model_selection import GridSearchCV
#Grid search'un aramasını istediğimiz parametreleri bir array olarak yazalım
#Ardından teker teker o parametreleri modele vererek GridSearchCV denemeler yapar
#Arrayin içine de dictionary açarak optimize edilmesini istediğimiz parametreleri veririz
parameters = [{'n_neighbors':[1,2,3,4,5],'metric':['minkowski','euclidean','manhattan']}]
#Bu listenin her bir kombinasyonu arka planda denenerek en optimum parametreler bulunacak

#estimator => optimize edilecek model
#param_grid => parametreler/denenecekler
#scoring => neye göre score belirlenecek (örn: accuracy)
#cv => kaç fold olacağı
#n_jobs => aynı anca çalışılacak iş sayısı
gs = GridSearchCV(estimator=knn,
                          param_grid=parameters,
                          scoring='accuracy',
                          cv=4,
                          n_jobs=-1)

#Artık gridSearch'u eğitelim/çalıştıralım
gridSearch = gs.fit(x_train,y_train)

#Optimum parametreleri ve en iyi sonucu alalım (accuracy'e göre)
bestResult = gridSearch.best_score_
bestParams = gridSearch.best_params_

print('Best accuracy')
print(bestResult)
print('----------------')
print('Best params')
print(bestParams)