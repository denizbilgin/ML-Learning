import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Data loading
data = pd.read_csv('./clusteringData.csv')


#Test
print('RAW DATA')
print(data)
print('-----------------------')


#Data slicing
x = data[['Hacim','Maas']].values


#Importing the module
from sklearn.cluster import KMeans
kMeans = KMeans(n_clusters=3,init='k-means++')
#Train the model
kMeans.fit(x)


#Get details about the algorithm
print('Locations of Data Centers')
print(kMeans.cluster_centers_)
print('-----------------------')


#Calculating the optimum value of K (WCSS)
results = []
for i in range(1,11):
    #random_state is given, because we want to start same point within every loop
    kMeans = KMeans(n_clusters=i,init='k-means++',random_state=123)
    kMeans.fit(x)
    #inertia is the value of WCSS
    results.append(kMeans.inertia_)


#Visualization for selecting K
plt.plot(range(1,11),results)
plt.show()
#You can choose 4 from the graph


kMeans = KMeans(n_clusters=4,init='k-means++',random_state=123)
y_pred = kMeans.fit_predict(x)
print('Her bir veri noktasının hangi clusterda olduğuna bir bakalım.')
print('y_pred şu şekildedir')
print(y_pred)
print('---------------------')


#Visualization
plt.title('Situation After Clustering')
plt.scatter(x[y_pred == 0,0],x[y_pred == 0,1],s=100,c='r')
plt.scatter(x[y_pred == 1,0],x[y_pred == 1,1],s=100,c='b')
plt.scatter(x[y_pred == 2,0],x[y_pred == 2,1],s=100,c='g')
plt.scatter(x[y_pred == 3,0],x[y_pred == 3,1],s=100,c='y')
plt.show()