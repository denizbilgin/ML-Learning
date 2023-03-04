import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Data loading
data = pd.read_csv('./aprioriData.csv', header=None)


#Data preprocessing
transactions = []
for i in range(0,7501):
    transactions.append([str(data.values[i,j]) for j in range(0,20)])


#Module importing
from apyori import apriori
rules = apriori(transactions,min_support=0.01, min_confidence=0.2, min_lift=3, min_lenght=2)
print(list(rules))
