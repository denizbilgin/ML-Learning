import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Verinin yüklenmesi
data = pd.read_csv('data.csv')

# test
print(data)
print('-----------------')

#Encoder: Kategorik -> Numeric
country = data.iloc[:,0:1].values
print(country)
print('-----------------')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(data.iloc[:,0])
print(country)
print('-----------------')
# One Hot Encoder
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print('Ülkelerin OHE kullanılarak arraye çevrilmiş hali')
print(country)
print('-----------------')

#encoder: Kategorik -> Numeric
gender = data.iloc[:,-1:].values
print('Cinsiyet verilerinin ham array hali')
print(gender)
print('-----------------')
gender[:,-1] = le.fit_transform(data.iloc[:,-1])
print('Cinsiyet verisinin 0,1 li array hali')
print(gender)
print('-----------------')
gender = ohe.fit_transform(gender).toarray()
print('OHE kullanılıp 2 kolonlu array hali (DF değil)')
print(gender)
print('-----------------')

#numpy dizileri dataframe donusumu
countryLastDF = pd.DataFrame(data=country, index = range(22), columns = ['fr','tr','us'])
print('Tüm ülke berisinin DF hali')
print(countryLastDF)
print('-----------------')
wha = data[['boy','kilo','yas']]
print('Boy,Kilo,Yaş kolonlarının olduğu DF')
print(wha)
print('-----------------')
genderLastDF = pd.DataFrame(data = gender[:,:1], index = range(22), columns = ['cinsiyet'])
print('Cinsiyet ile ilgili işlemlerin ardından elde edilen DF')
print(genderLastDF)
print('-----------------')

#dataframe birlestirme islemi
countryLastDFWithWhaDF=pd.concat([countryLastDF,wha], axis=1)
print('countryLastDF ile wha DFlerinin birleştirilmiş hali')
print(countryLastDFWithWhaDF)
print('-----------------')
result=pd.concat([countryLastDFWithWhaDF,genderLastDF], axis=1)
print('Ön İşleme Sonucu kullanılacak DataFrame')
print(result)
print('-----------------')

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(countryLastDFWithWhaDF,genderLastDF,test_size=0.33, random_state=0)
print('x_train şu şekildedir')
print(type(x_train))
print(x_train)
print('-----------------')
print('y_train şu şekildedir')
print(y_train)
print('-----------------')

# Gerekli importu yapalım
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Modelimizi eğitelim
regressor.fit(x_train,y_train)

# Artık tahminler yapmay başlayabiliriz
y_pred = regressor.predict(x_test)
print('y_pred')
print(y_pred)
print('-----------------')
print('Modelin doğru bilip bilmediğine bakmak için y_test e bakalım')
print(y_test)
print('-----------------')

# Şimdi boy tahmin eden algoritma yazalım.
# İlk olarak hızlıca boy verisini son DF'den çekelim
height = result.iloc[:,3:4]
print('Boy verileri')
print(height)
print('-----------------')
leftDf = result.iloc[:,:3]
rightDf = result.iloc[:,4:]
dfToPredHeight = pd.concat([leftDf,rightDf],axis=1)
print('Boy kolonu hariç tüm verinin olduğu DF')
print(dfToPredHeight)
print('-----------------')

x_train, x_test, y_train, y_test = train_test_split(dfToPredHeight,height,test_size=0.33,random_state=0)
print('x_train şu şekildedir')
print(x_train)
print('-----------------')

regressor2 = LinearRegression()
regressor2.fit(x_train,y_train)
y_pred = regressor2.predict(x_test)
print('y_pred şu şekildedir')
print(y_pred)
print('-----------------')
print('y_test şu şekildedir')
print(y_test)

# gerekli importu yapalım
import statsmodels.api as sm

# Hangi feature'ın nasıl etkilediğini görebilmek için feature listesi elde edelim
# DF'ye tüm değerleri 1 olan ekstra 1 kolon ekleyelim
X = np.append(arr= np.ones((22,1)).astype(int),values=dfToPredHeight,axis=1)

# Şimdi her bir kolonun listesini oluşturalım ki etkisiz olanı ileride çıkarabilelim
X_list = dfToPredHeight.iloc[:,[0,1,2,3,4,5]].values
X_list = np.array(X_list,dtype=float)

# Tahmin edeceğimiz height Dfsi ile Tüm kolonları OLS'ye parametre verelim
# Bu işlem her kolonun height üzerindeki etksini ölçecektir
model = sm.OLS(height,X_list).fit()

# Modelimizin özetine bakalım
print(model.summary())
# Bu özet bize önemli bilgiler verecektir
# Bu rapordaki P değeri bizim için önemlidir
# P değeri büyük olursa (bu örnekteki outputta x5) o kolonu elemeliyiz
# Yani 4. elemanı elemeliyiz

print('************************')

# 4. elemanı elemiş halde tekrar rapora bakalım
X_list = dfToPredHeight.iloc[:,[0,1,2,3,5]].values
X_list = np.array(X_list,dtype=float)
model = sm.OLS(height,X_list).fit()
# Rapora tekrar bakalım
print(model.summary())

# Eğer P değeri 0.05'den büyükse silmelisiniz
# 2. raporda 0.031 görünüyor bu değeri silmeyeceğiz
# Normal projelerinizde bu elemelerin ardından tekrar regression işlemi yapmalısınız