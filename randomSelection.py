import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Data loading
data = pd.read_csv('./Ads_CTR_Optimisation.csv')


#Generate random number
import random
N = 10000
numberOfAdvetisements = 10
total = 0
selecteds = []
for i in range(0,N):
    randomNumber = random.randrange(numberOfAdvetisements)
    selecteds.append(randomNumber)
    reward = data.values[i,randomNumber]
    total += reward

print('Rastgele durumlarda elde edilen ödül sayısı')
print(total)
print('------------------------')


#Visualization
plt.hist(selecteds)
plt.show()