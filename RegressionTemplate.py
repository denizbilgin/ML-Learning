#Deniz Bilgin - 2023 February
#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


#Data loading
data = pd.read_csv('./polynomialRegressionData.csv')


#Data slicing
x_train = data[['Egitim Seviyesi']]
y_train = data[['maas']]


#Separating the data as x_train, x_test, y_train, y_test
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)


#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,lin_reg.predict(x_train))
plt.title('Linear Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaşlar')
plt.show()
print('Linear Regression R2 değeri')
print(r2_score(y_train, lin_reg.predict(x_train)))


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,lin_reg2.predict(poly_reg.fit_transform(x_train)))
plt.title('2nd Degree Polynomial Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaşlar')
plt.show()
print('2nd Degree Polynomial Regression R2 değeri')
print(r2_score(y_train, lin_reg2.predict(poly_reg.fit_transform(x_train))))

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,lin_reg2.predict(poly_reg.fit_transform(x_train)))
plt.title('4th Degree Polynomial Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaşlar')
plt.show()
print('4th Degree Polynomial Regression R2 değeri')
print(r2_score(y_train, lin_reg2.predict(poly_reg.fit_transform(x_train))))


#Data scaling
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_scaled = sc1.fit_transform(x_train)
sc2=StandardScaler()
y_scaled = sc2.fit_transform(y_train)


#Support Vector Regression
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaled,y_scaled.ravel())
plt.scatter(x_scaled,y_scaled,color='red')
plt.plot(x_scaled,svr_reg.predict(x_scaled))
plt.title('Support Vector Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaşlar')
plt.show()
print('Support Vector Regression R2 değeri')
print(r2_score(y_scaled, svr_reg.predict(x_scaled)))


#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x_train,y_train)
plt.scatter(x_train,y_train, color='red')
plt.title('Decision Tree Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaşlar')
plt.plot(x_train,r_dt.predict(x_train))
plt.show()
print('Decision Tree R2 değeri')
print(r2_score(y_train, r_dt.predict(x_train)))


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(x_train,y_train)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,rf_reg.predict(x_train))
plt.title('Random Forest Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaşlar')
plt.show()
print('Random Forest R2 değeri')
print(r2_score(y_train, rf_reg.predict(x_train)))


#Summary Of R2 Scores
print('-----------------------')
print('Linear Regression R2 değeri')
print(r2_score(y_train, lin_reg.predict(x_train)))

print('Polynomial Regression R2 değeri')
print(r2_score(y_train, lin_reg2.predict(poly_reg.fit_transform(x_train))))

print('Support Vector Regression R2 değeri')
print(r2_score(y_scaled, svr_reg.predict(x_scaled)))

print('Decision Tree Regression R2 değeri')
print(r2_score(y_train, r_dt.predict(x_train)))

print('Random Forest Regression R2 değeri')
print(r2_score(y_train, rf_reg.predict(x_train)))