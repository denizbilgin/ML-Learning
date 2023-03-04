import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#Veriyi 2 kolon olacak şekilde dimension reduction işemine tabi tutacağız
lda = LDA(n_components=2)
#Hem eğitim hem de transform (dönüştürme) işlemini uyguluyoruz, LDA'in PCA'den farklı
#olarak ilave bir parametresi de var. Çünkü classa göre en iyi ayrımı dener LDA
x_train_lda = lda.fit_transform(x_train,y_train)
#x_test'i de daha önceden eğittiğimiz lda ile sadece transform ediyoruz
x_test_lda = lda.transform(x_test)


#Artık indirgenmiş veri üzerinden bildiğimiz classification algoritmalarından
#birini kullanarak tahminler yapabiliriz. Logistic Regression kullanalım
from sklearn.linear_model import LogisticRegression
classifierLDA = LogisticRegression(random_state=0)
#LDA'li modeli eğitelim (2 columns)
classifierLDA.fit(x_train_lda,y_train)


#LDA'siz model (13 columns)
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


#Predictions
y_pred_lda = classifierLDA.predict(x_test_lda)
y_pred = classifier.predict(x_test)


#Confusion matrixes for comparing
from sklearn.metrics import confusion_matrix
confusionMatrixLDA = confusion_matrix(y_test,y_pred_lda)
confusionMatrix = confusion_matrix(y_test,y_pred)


print('LDA kullanılmış veri üzerinden, eğitilen makinenin CMsi')
print(confusionMatrixLDA)
print('----------------------------')
print('PCAsiz veri üzerinden eğitilen makinenin CMsi')
print(confusionMatrix)