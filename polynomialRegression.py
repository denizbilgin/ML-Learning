# Deniz Bilgin - 16.02.2023
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#Veri yükleme
data = pd.read_csv("./polynomialRegressionData.csv")


#Data Frame dilimleme (slice)
educationDegree = data[['Egitim Seviyesi']]
salary = data[['maas']]


#Polynomial Regression
polyReg = PolynomialFeatures(degree=2)
x_poly = polyReg.fit_transform(educationDegree.values)
lr = LinearRegression()
lr.fit(x_poly,salary)


#Farklı Dereceden Polynomial Regression
lr2 = LinearRegression()
polyReg2 = PolynomialFeatures(degree=4)
x_poly2 = polyReg2.fit_transform(educationDegree.values)
lr2.fit(x_poly2,salary)


#Görselleştirme
plt.scatter(educationDegree,salary,c='r',marker='x')
plt.plot(educationDegree,lr.predict(x_poly))
plt.show()

plt.scatter(educationDegree,salary,c='r',marker='x')
plt.plot(educationDegree,lr2.predict(x_poly2))
plt.show()


#Predictions
print('2. dereceden model ile tahminler')
print(lr.predict(polyReg.fit_transform([[11]])))
print(lr.predict(polyReg.fit_transform([[6.6]])))
print('------------------------------')

print('2. dereceden model ile tahminler')
print(lr2.predict(polyReg2.fit_transform([[11]])))
print(lr2.predict(polyReg2.fit_transform([[6.6]])))
print('------------------------------')