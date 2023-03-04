import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv('./data.csv')

height = data[['boy']]
print(height)
print("---------------------------")

heightAndWeight = data[['boy','kilo']]
print(heightAndWeight)
print("---------------------------")


# Missing values
missingValues = pd.read_csv('./missingValues.csv')
print(missingValues)
print("---------------------------")

# Eksik verilerin yerine o sütunun ortalamasını yazabiliriz
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
# Şimdi eksik veriler ortalamaya döndü mü diye bakalım
age = missingValues.iloc[:,1:4].values
print(age)
print("---------------------------")
# Fit fonksiyonu ile birlikte öğrenilecek olan değeri hesaplarız
imputer = imputer.fit(age[:,1:4])
# Transform fonksiyonu ile de o değeri yerine yazarız
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)