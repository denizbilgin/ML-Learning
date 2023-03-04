import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv('./data.csv')

# Her şeyden önce o kolonu çekelim
# Bu kategorik verileri dizi şeklince çekmek için iloc fonksiyonunu kullanalım
country = data.iloc[:,0:1].values
# Buradaki ilk : tüm satırları almasını söyler, virgülden sonrası da alınacak sütunları söyler

# Çektiğimiz verilere göz atalım
print(country)
print('-----------------------')

# Kategorik kolonun değişmesi için encoder kullanacağız
le = preprocessing.LabelEncoder()
# Bildiğiniz üzere fit öğrenir, transform ise günceller, fit_transform ikisini de yaoar
country[:,0] = le.fit_transform(data.iloc[:,0])
print(country)
print('-----------------------')

# One Hot Encoder Kullanarak her bir ülkeyi 1 ile temsil eden aidiyet tablosu oluşturduk
oneHotEncoder = preprocessing.OneHotEncoder()
country = oneHotEncoder.fit_transform(country).toarray()
print(country)
print('-----------------------')

# Şuan ülkeler için ayrı bir dataFrame'imiz var fakat henüz bunda hangi kolonun ne olduğu
# ile ilgili bilgi bulunmuyor
# Şimdi bunları birleştirip tek bir DataFrame oluşturmaya bakalım
lastDataFrameOfCountries = pd.DataFrame(data=country, index= range(22), columns=['fr','tr','us'])
# DF'lerin her bir satırda kolayca işlem yapmak için indexleri vardır
print(lastDataFrameOfCountries)
print('-----------------------')

# Şimdi boy, kilo ve yaşın olduğu verileri çekip onu bir DF'ye çevirelim
hwa = data[['boy','kilo','yas']]

# Ardından cinsiyeti de alalım
gender = data.iloc[:,-1].values
# Şimdi bu diziyi DF'ye çevirelim
gender = pd.DataFrame(data=gender, index=range(22), columns=['cinsiyet'])
print(gender)
print('-----------------------')

# Son olarak bu 3 DF'yi birleştirelim
# Dikey konumda birleştirme yapacağı için çok fazla NaN göreceğiz
result = pd.concat([lastDataFrameOfCountries,hwa])
print(result)
print('-----------------------')

# Yatay konumda birleştirme yapması için axis'i 1 yapacağız ve satıra göre birleştirecek
result = pd.concat([lastDataFrameOfCountries,hwa], axis=1)
print(result)
print('-----------------------')

# Son olarak ise cinsitet DF'si ile de birleştirelim
result = pd.concat([result,gender], axis=1)
print(result)