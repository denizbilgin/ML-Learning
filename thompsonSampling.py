import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Data loading
data = pd.read_csv('./Ads_CTR_Optimisation.csv')


numberOfIterations = 10000
numberOfAdvertisements = 10
total = 0
selecteds = []
ones = [0] * numberOfAdvertisements
zeros = [0] * numberOfAdvertisements

for n in range(1,numberOfIterations):
    ad = 0 #Advertisement that selected
    maxThompson = 0

    #Looping around the advertisements for all users
    for i in range(0,numberOfAdvertisements):

        #Now we need to calculate random beta
        randomBeta = random.betavariate(ones[i]+1, zeros[i]+1)

        #max finding process
        if randomBeta > maxThompson:
            maxThompson = randomBeta
            ad = i

    selecteds.append(ad)
    reward = data.values[n,ad] #Real information
    #Is reward 1 or 0
    if reward == 1:
        ones[ad] += 1
    else:
        zeros[ad] += 1
    total += reward

print('Thompson Sampling algoritmasının zeki seçimlerle elde ettiği toplam ödül sayısı:')
print(total)
print('-----------------')


#Visualization
plt.hist(selecteds)
plt.show()