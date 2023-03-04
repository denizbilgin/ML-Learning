import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras


#Data loading
data = pd.read_csv('./churnModelling.csv')


#Test
print(data)
#In this dataset, we don't need RowNumber,CustomerId and Surname columns
print('---------------------')


#Data slicing
x = data.iloc[:,3:13].values
y = data[['Exited']].values


#Encoding
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
x[:,1] = labelEncoder.fit_transform(x[:,1])     #Encoding geography column
labelEncoder2 = LabelEncoder()
x[:,2] = labelEncoder2.fit_transform(x[:,2])    #Encoding gender column


#OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#ColumnsTransformer birden fazla kolonun aynı anda encode edilmesini sağlar
ohe = ColumnTransformer([('ohe',OneHotEncoder(dtype=float),[1])],remainder='passthrough')
x = ohe.fit_transform(x)
x = x[:,1:]


#Seperating the data as x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=0.33)


#Data scaling (Standardizing features)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#Artificial Neural Networks
import keras
from keras.models import Sequential
#Sequential aslında keras'a, ben bir yapay sinir ağı oluşturuyorum demektir
from keras.layers import Dense
#Dense, keras'ta layer oluşturabilmek için gerekir

classifier = Sequential()   #Classification yapan bir yapay sinir ağı oluşturuldu
#Hidden layer'daki nöron sayısını input sayısı ve output sayısının ortalaması olarak alabilirsiniz
#Yaptığımız işlemlerin ardından x_train'de 11 input var, totalde 1 output var. Ortalama 6 olacaktır
#init verilerimizi nasıl başlatacağımız ile ilgili, daha önce 0'a yakın başlatmak gerek demiştik hatırlayın (sinapsisler initialize edilir)
#activation ise hangi activation fonksiyonunu kullanacağını belirler, keras dokümantasyonundan diğerlerine bakabilirsiniz
#input_dim, totalde kaç input olduğu ister (indirgeme ve ohe işlemleri sonucu 11 kolonumuz var)
classifier.add(Dense(6, kernel_initializer='glorot_uniform', activation='relu', input_dim=11))
#Bir tane daha hidden layer ekleyelim
classifier.add(Dense(6, kernel_initializer='glorot_uniform', activation='relu'))
#Her bir add işlemi ile ayrı bir hidden layer ekleye ekleye ilerleyebilirsiniz.
#Genelde input ve hidden layerlarda linear activation fonksiyonları kullanılması önerilir
#Output layer'da ise sigmoid fonksiyonu kullanılması önerilir

#Şimdi output layer ekleyelim
classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))

#Artık yapay sinir ağımızı compile edebiliriz
#Optimizer aslında kullanacağımız gradient descent tipini belirler (Stochastic burada SGD adıyla geçiyor)
#loss fonksiyonu bildiğimiz hata ölçmeye yarayan fonksiyonlardır, birden fazla çeşidi vardır
#loss fonksiyonlarına göre aslında hata payları bulunur ve ona göre optimize işlemi yapılır
#biz loss fonksiyonu olarak binary_crossentropy kullanacağız çünkü çıktımız tek bir satır ve o satır 0 ve 1 den oluşuyor
#normal los fonksiyonlarında tahmin-gerçek gibi işlemler vardı, burada 1-0 yapmak mantıklı olmayacağı için bunu kullanıyoruz
#metrics neyi optimize edeceğinin metriğini vermemiz gerekiyor
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#Şimdi nöral ağımızı çalıştırmaya geçebiliriz
#ANN'yi eğitelim
#Epochs kaç çağ tamamlandığında öğrenme işleminin duracağını belirler
#Ayrıca batch'li bir yapı kullanıyorsanız batch büyüklüğünü de fit fonksiyonunda belirtmelisiniz
classifier.fit(x_train,y_train,epochs=50)


#Predictions
#Burada elde edilen y_pred değerleri o kişinin yüzde kaç bırakıp bırakmayacağını söyler
#Örneğin 0.20 ise bu kişinin bırakmayacağına yüzde 80 gözüyle bakarız
y_pred = classifier.predict(x_test)

#Oranlar ile uğraşmayıp direkt bir threshold'un üstündekileri bırakacak şekline çevirelim
y_pred = y_pred > 0.5

print('---------------------')

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusionMatrix)