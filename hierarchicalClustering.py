import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Data loading
data = pd.read_csv('./clusteringData.csv')


#Test
print('RAW DATA')
print(data)
print('---------------------')


#Data slicing
x = data[['Hacim','Maas']].values
print('x şu şekildedir')
#print(x)
print('---------------------')


#Importing the module
from sklearn.cluster import AgglomerativeClustering
#affinity is the method of calculating distance between points
#linkage is the method of calculating distance between clusters
aggloClustering = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
#Train the model
y_pred = aggloClustering.fit_predict(x)
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


#Dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.show()