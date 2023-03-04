from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('./data.csv')

# Cinsiyet tahmin edeceğimiz bir algoritma yazacağımızı düşünelim
# Bu yüzden cinsiyet kolonunun olmadığı bir DF'ye ihtiyacımız var
# Ayrıca train edilmesi için cinsiyet DF'si de gerekir
gender = data.iloc[:,-1].values
gender = pd.DataFrame(data=gender,index=range(22),columns=['cinsiyet'])

country = data.iloc[:,0:1].values


# Kategorik kolonun değişmesi için encoder kullanacağız
le = preprocessing.LabelEncoder()
# Bildiğiniz üzere fit öğrenir, transform ise günceller, fit_transform ikisini de yaoar
country[:,0] = le.fit_transform(data.iloc[:,0])

# One Hot Encoder Kullanarak her bir ülkeyi 1 ile temsil eden aidiyet tablosu oluşturduk
oneHotEncoder = preprocessing.OneHotEncoder()
country = oneHotEncoder.fit_transform(country).toarray()

# Şuan ülkeler için ayrı bir dataFrame'imiz var fakat henüz bunda hangi kolonun ne olduğu
# ile ilgili bilgi bulunmuyor
# Şimdi bunları birleştirip tek bir DataFrame oluşturmaya bakalım
lastDataFrameOfCountries = pd.DataFrame(data=country, index= range(22), columns=['fr','tr','us'])
# DF'lerin her bir satırda kolayca işlem yapmak için indexleri vardır
print(lastDataFrameOfCountries)
print('-----------------------')

# LastDFOfCountries ile Genel DF'yi birleştirelim
hwa = data[['boy','kilo','yas']]
result = pd.concat([lastDataFrameOfCountries,hwa], axis=1)
print('Sonuç DF:')
print(result)
print('-----------------------')


# 0.33'den kastımız verinin 33%'ünü test için kullanmak, 66%'sini train için kullanmaktır
x_train, x_test, y_train, y_test = train_test_split(result,gender,test_size=0.33,random_state=0)
print('X train şu şekildedir:')
print(x_train)
print('-----------------')

print('Y train şu şekildedir:')
print(y_train)
print('-----------------')

# Verilerin ölçekleri farklı olabilir bu yüzden program yavaş
# çalışabilir, bunu engellemek için veriyi ölçeklendirmeliyiz
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# fit_transform fonksiyonu ile StandardScaler otomatik scale eder
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

print('Scale edilmiş x_train:')
print(X_train)