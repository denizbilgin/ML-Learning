import matplotlib.pyplot as plt
import pandas as pd


# veri yukleme
data = pd.read_csv('./polynomialRegressionData.csv')


#Data slicing
educationDegree = data[['Egitim Seviyesi']]
salary = data[['maas']]


#Verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_scaled = sc1.fit_transform(educationDegree)
sc2=StandardScaler()
y_scaled = sc2.fit_transform(salary)


#Support vector regression işlemleri
from sklearn.svm import SVR
svrRegression = SVR(kernel='rbf') # rbf => değişkenler arasındaki ilişki durumuna göre bir çekirdek seçilmeli.
#Modeli eğitelim
svrRegression.fit(x_scaled, y_scaled.ravel())


#Görselleştirme
plt.scatter(x_scaled, y_scaled,c='r')
plt.plot(x_scaled,svrRegression.predict(x_scaled),color='blue')
plt.show()


#Prediction
print(svrRegression.predict([[11]]))
print(svrRegression.predict([[6.6]]))

