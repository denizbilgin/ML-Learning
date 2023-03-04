# - HOMEWORK -
# 1- Download the data
# 2- Find Necessary/Unnecessary variables
# 3- Train the model with 5 different algorithms (MLR, PR, SVR, DT, RF)
# 4- Compare the accuracys of the models
# 5- Make predictions that includes (10 years experience, has 100 point, CEO), (10 years experience, has 100 point, Manager)
# with 5 different models

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Load the data
data = pd.read_csv('./salaries.csv')
print("RAW DATA")
print(data)
print("------------------------")


#Checking corelation
print(data.corr())
print("------------------------")
#Görülen bu korelasyon matrisinde neyin ne ile ne kadar ilişkili ve baskın olduğunu
#görebilirsiniz.


#Data Slicing
# id ve ünvan kolonları gereksiz kolonlardır
# x_train = data[['UnvanSeviyesi','Kidem','Puan']] => that line deleted by column 39
x_train = data[['UnvanSeviyesi']]
y_train = data[['maas']]


#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(x_train,y_train)


#Column's Importance
import statsmodels.api as sm
model = sm.OLS(linearRegressor.predict(x_train),x_train)
print(model.fit().summary())
#As you can see P-Value of Kidem and Puan is so high, so we must delete that columns


#Visualization
plt.scatter(x_train,y_train,c='r')
plt.plot(x_train,linearRegressor.predict(x_train))
plt.title('Linear Regression')
plt.xlabel('Ünvan Seviyesi')
plt.ylabel('Maaş')
plt.show()


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
polyRegressor = PolynomialFeatures(degree=5)
x_poly = polyRegressor.fit_transform(x_train)
lr = LinearRegression()
lr.fit(x_poly,y_train)
plt.scatter(x_train,y_train,c='r')
plt.plot(x_train,lr.predict(x_poly))
plt.title('Polynomial 5th Degree Regression')
plt.xlabel('Ünvan Seviyesi')
plt.ylabel('Maaş')
plt.show()


#Data Scaling
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
x_scaled = scaler1.fit_transform(x_train)
scaler2 = StandardScaler()
y_scaled = scaler2.fit_transform(y_train)


#Support Vector Regression
from sklearn.svm import SVR
svrRegressor = SVR(kernel='rbf')
svrRegressor.fit(x_scaled,y_scaled)
plt.scatter(x_scaled, y_scaled, c='r')
plt.plot(x_scaled, svrRegressor.predict(x_scaled))
plt.title('Support Vector Regression')
plt.xlabel('Ünvan Seviyesi')
plt.ylabel('Maaş')
plt.show()


#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
decisionTreeRegressor = DecisionTreeRegressor(random_state=0)
decisionTreeRegressor.fit(x_train, y_train)
plt.scatter(x_train, y_train, c='r')
plt.plot(x_train, decisionTreeRegressor.predict(x_train))
plt.title('Decision Tree Regression')
plt.xlabel('Ünvan Seviyesi')
plt.ylabel('Maaş')
plt.show()


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
randomForestRegressor = RandomForestRegressor(random_state=0, n_estimators=10)
randomForestRegressor.fit(x_train, y_train)
plt.scatter(x_train, y_train, c='r')
plt.plot(x_train, randomForestRegressor.predict(x_train))
plt.title('Decision Tree Regression')
plt.xlabel('Ünvan Seviyesi')
plt.ylabel('Maaş')
plt.show()

print("------------------------")
print('R2 Scores of models')
from sklearn.metrics import r2_score
print("------------------------")
print('Multiple Linear Regression R2 Score')
print(r2_score(y_train, linearRegressor.predict(x_train)))
print("------------------------")
print('Polynomial 5th Degree Regression R2 Score')
print(r2_score(y_train, lr.predict(polyRegressor.fit_transform(x_train))))
print("------------------------")
print('Support Vector Regression R2 Score')
print(r2_score(y_scaled, svrRegressor.predict(x_scaled)))
print("------------------------")
print('Decision Tree Regression R2 Score')
print(r2_score(y_train, decisionTreeRegressor.predict(x_train)))
print("------------------------")
print('Random Forest Regression R2 Score')
print(r2_score(y_train, randomForestRegressor.predict(x_train)))
print("------------------------")


#Predictions
prediction1 = [[10]]
prediction2 = [[7]]
print('PREDICTIONS')
print('Multiple Linear Regression Predictions')
print(linearRegressor.predict(prediction1))
print(linearRegressor.predict(prediction2))
print("------------------------")
print('Polynomial 5th Degree Regression Predictions')
print(lr.predict(polyRegressor.fit_transform(prediction1)))
print(lr.predict(polyRegressor.fit_transform(prediction2)))
print("------------------------")
print('Support Vector Regression Predictions')
print(svrRegressor.predict(prediction1))
print(svrRegressor.predict(prediction2))
print("------------------------")
print('Decision Tree Regression Predictions')
print(decisionTreeRegressor.predict(prediction1))
print(decisionTreeRegressor.predict(prediction2))
print("------------------------")
print('Random Forest Regression Predictions')
print(randomForestRegressor.predict(prediction1))
print(randomForestRegressor.predict(prediction2))
print("------------------------")


#Beign sure about R2 value
model = sm.OLS(decisionTreeRegressor.predict(x_train),x_train)
print(model.fit().summary())
