import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Bu veri seti belirli özelliklere göre elinizdeki şarabın hangi classa ait olduğunu buluyor
#Data loading
data = pd.read_csv('./pcaDimensionReductionData.csv')


#Data slicing
x = data.iloc[:,0:13].values
y = data.iloc[:,-1].values


#Seperating the data as x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)


#Data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#PCA
from sklearn.decomposition import PCA
#Veriyi 2 kolon olacak şekilde dimension reduction işemine tabi tutacağız
pca = PCA(n_components=2)
#Hem eğitim hem de transform (dönüştürme) işlemini uyguluyoruz
x_train2 = pca.fit_transform(x_train)
#x_test'i de daha önceden eğittiğimiz pca ile sadece transform ediyoruz
x_test2 = pca.transform(x_test)


#Artık indirgenmiş veri üzerinden bildiğimiz classification algoritmalarından
#birini kullanarak tahminler yapabiliriz. Logistic Regression kullanalım
#PCA'li Logistic Regression (2 columns)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train2,y_train)

#PCA'siz Logistic Regression (13 columns)
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(x_train,y_train)


#Predictions
y_predPCA = classifier.predict(x_test2)
y_pred = classifier2.predict(x_test)


#Confusion Matrix for Comparing
from sklearn.metrics import confusion_matrix
confusionMatrixPCA = confusion_matrix(y_test,y_predPCA)
confusionMatrix = confusion_matrix(y_test,y_pred)

print('PCA kullanılmış veri üzerinden, eğitilen makinenin CMsi')
print(confusionMatrixPCA)
print('----------------------------')
print('PCAsiz veri üzerinden eğitilen makinenin CMsi')
print(confusionMatrix)

#Verideki kolon sayısı nerdeyse 6'da birine inmesine rağmen çok az hata yapmışız
#Yani çok büyük veri setlerinde PCA kullanımı bize büyük hız kazandırırken çok küçük kayıplar
#yaşatabilir
#Farklı durumlarda PCA kullanımı başarı da arttırabilir