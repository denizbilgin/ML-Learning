#1.kutuphaneler
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Veri Önişleme
# Veri Yükleme
data = pd.read_csv('./monthlySellingData.csv')

#test
print('Verinin Ham hali')
print(data)
print('-------------------------')

#veri on isleme
months = data[['Aylar']]
print(months)
print('-------------------------')
sales = data[['Satislar']]
print(sales)
print('-------------------------')
salesArray = data.iloc[:,:1].values
print('Satışların Numpy Array Hali')
print(salesArray)
print('-------------------------')

# Verilerin train ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(months,sales,test_size=0.33, random_state=0)

# Verilerin ölçeklendirilmesi

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)


# Linear regression'ı kullanmak için gerekli import'u yapalım
from sklearn.linear_model import LinearRegression

# Fonksiyonlara erişmek için objeyi çağıralım
lr = LinearRegression()

# Modelimizi oluşturmaya başlayalım
lr.fit(x_train,y_train)

# Şimdi oluşturduğumuz modeli kullanarak prediction yapalım
# Bunu yaparken X_test'i kullanarak Y_testi bulmasını bekliyoruz denilebilir
prediction = lr.predict(x_test)
print("X_test'i kullanarak tahminler edelim:")
print(prediction)
print('-------------------------')

# Test
print(type(X_train), " is type of X_train")
print(type(x_train), " is type of x_train")


# Veriyi görselleştirelim
# Kalemi kaldırmadan çizme işlemi uygulanacağı için küçükten büyüğe çizmeli ki karışıklık olmasın
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.xlabel('Aylar')
plt.ylabel('Satışlar')
plt.title('Aylara göre satış')
plt.plot(x_train,y_train)
# Her bir x_test için onun y'de karşılığı olan şey zaten doğrunun kendisi olacaktır
plt.plot(x_test,lr.predict(x_test))
plt.show()