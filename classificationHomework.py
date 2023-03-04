# - HOMEWORK -
# We're trying to predict iris (yaprak türü)
# Find the best classification algorith for this data set


#Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


#Data loading
data = pd.read_excel('./irisDataSet.xls')


#Visualization
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
plt.figure(2, figsize=(8, 6))
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.zaxis.set_ticklabels([])
plt.show()


#Test
print('RAW DATA')
print(data)
print('-------------------------')


#Data slicing
x = data.iloc[:,0:4]
y = data.iloc[:,-1]


#Seperating the data as x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=0.33)


#Logistic Regression Classification
from sklearn.linear_model import LogisticRegression
logisticRegressor = LogisticRegression(random_state=0)
logisticRegressor.fit(x_train,y_train)
y_pred = logisticRegressor.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of Logistic Regression')
print(confusionMatrix)
print('Accuracy')
print(accuracy_score(y_test,y_pred))
print('-------------------------')


#K-NN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of K-NN')
print(confusionMatrix)
print('Accuracy')
print(accuracy_score(y_test,y_pred))
print('-------------------------')


#Support Vector Machine Classification
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of SVC')
print(confusionMatrix)
print('Accuracy')
print(accuracy_score(y_test,y_pred))
print('-------------------------')


#Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
gaussianNB = GaussianNB()
gaussianNB.fit(x_train,y_train)
y_pred = gaussianNB.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of Naive Bayes')
print(confusionMatrix)
print('Accuracy')
print(accuracy_score(y_test,y_pred))
print('-------------------------')


#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
decisionTree = DecisionTreeClassifier(criterion='entropy')
decisionTree.fit(x_train,y_train)
y_pred = decisionTree.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of Decision Tree')
print(confusionMatrix)
print('Accuracy')
print(accuracy_score(y_test,y_pred))
print('-------------------------')


#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(n_estimators=10,criterion='entropy')
randomForest.fit(x_train,y_train)
y_pred = randomForest.predict(x_test)
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix of Random Forest')
print(confusionMatrix)
print('Accuracy')
print(accuracy_score(y_test,y_pred))
print('-------------------------')