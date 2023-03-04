#Deniz Bilgin - 2023 February
#Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

#Data loading
data = pd.read_csv('data.csv')


#Data slicing
x = data[['boy','kilo','yas']]
y = data[['cinsiyet']]


#Seperating the data as x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)


#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logisticRegressor = LogisticRegression(random_state=0)
logisticRegressor.fit(x_train,y_train)
y_pred = logisticRegressor.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of Logistic Regression')
print(confusionMatrix)
print('-------------------------')


#K-NN Classification
from sklearn.neighbors import KNeighborsClassifier
#n_neighbors and metric can changeable
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of K-NN')
print(confusionMatrix)
print('-------------------------')


#Support Vector Machine Classification
from sklearn.svm import SVC
#kernel can changeable
svc = SVC(kernel='poly')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of Support Vector Machine')
print(confusionMatrix)
print('-------------------------')


#Naive Bayes Classification
#Type of NB can changeable
from sklearn.naive_bayes import GaussianNB
gaussianNB = GaussianNB()
gaussianNB.fit(x_train,y_train)
y_pred = gaussianNB.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of Naive Bayes')
print(confusionMatrix)
print('-------------------------')


#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
#criterion can changeable
decisionTree = DecisionTreeClassifier(criterion='entropy')
decisionTree.fit(x_train,y_train)
y_pred = decisionTree.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of Decision Tree')
print(confusionMatrix)
print('-------------------------')


#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
#criterion and n_estimators can changeable
randomForest = RandomForestClassifier(criterion='entropy',n_estimators=10)
randomForest.fit(x_train,y_train)
y_pred = randomForest.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of Random Forest')
print(confusionMatrix)
print('-------------------------')


#ROC Curve, FPR, TPR
y_proba = randomForest.predict_proba(x_test)
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_proba[:,0],pos_label='e')
print(fpr)
print('-------------------------')
print(tpr)