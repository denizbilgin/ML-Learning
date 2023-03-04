import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Veri yükleme
data = pd.read_csv('./polynomialRegressionData.csv')


#Data slicing
x_train = data[['Egitim Seviyesi']]
y_train = data[['maas']]


#Kütüphaneyi import edelim ve regressor oluşturalım
from sklearn.ensemble import RandomForestRegressor
#n_estimators kaç adet decision tree oluşturularak tahmin yapılacağını belirler
randomForestRegressor = RandomForestRegressor(n_estimators=10, random_state=0)

#Modelimizi eğitelim
randomForestRegressor.fit(x_train,y_train)


#Predictions
print(randomForestRegressor.predict([[6.6]]))
#Birden fazla decision tree'nin tahminlerinin ortalaması alındığı için
#random forest algoritmaları veri setinde olmayan sonuçlar üretebilir
#Decision treeler sadece veri setindeki sonuçlardan tahminler yapar


#Görselleştirme
plt.scatter(x_train,y_train,c='r')
plt.plot(x_train, randomForestRegressor.predict(x_train))
plt.show()


#R square ile modeli puanlayalım
from sklearn.metrics import r2_score
print('Random Forest Regression R2 score')
print(r2_score(y_train,randomForestRegressor.predict(x_train)))