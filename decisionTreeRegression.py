import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Veri yükleme
data = pd.read_csv('./polynomialRegressionData.csv')


#Data slicing
x_train = data[['Egitim Seviyesi']]
y_train = data[['maas']]

print(x_train)

#Kütüphaneyi import edelim ve regressor oluşturalım
from sklearn.tree import DecisionTreeRegressor
decisionTreeRegressor = DecisionTreeRegressor(random_state=0)
#Modeli eğitelim
decisionTreeRegressor.fit(x_train, y_train)


#Görselleştirme
plt.scatter(x_train, y_train,c='r')
plt.plot(x_train, decisionTreeRegressor.predict(x_train))
plt.show()


#Predictions
print('Decision Tree Regression Predictions')
print(decisionTreeRegressor.predict([[11]]))
print(decisionTreeRegressor.predict([[6.6]]))

#Decision Tree Regression'un kötü özelliklerinden birisi, her zaman
#verilen y değerlerinden sonuçlar gelir. 0.678 ile 1'in çıktısını
#aynı olarak ve örneğin 2250 olarak tahmin eder