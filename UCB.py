import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


#Data loading
data = pd.read_csv('./Ads_CTR_Optimisation.csv')


numberOfIterations = 10000
numberOfAdvertisements = 10
# Ri(n)
rewards = [0] * numberOfAdvertisements #At initial, all the advertisements' reward is 0
# Ni(n)
clicks = [0] * numberOfAdvertisements #All the clicks until that moment
total = 0
selecteds = []
for n in range(0, numberOfIterations):
    #We need to calculate UCB values of all advertisements in each loop
    ad = 0 #Advertisement to select
    maxUCB = 0
    # With this loop, we'll choose an advertisement from UCB values
    for i in range(0, numberOfAdvertisements):
        if (clicks[i] > 0):
            average = rewards[i] / clicks[i]
            delta = math.sqrt(3 / 2 * math.log(n) / clicks[i])
            ucb = average + delta
        else:
            ucb = numberOfIterations * 10

        # Finding max USB
        if maxUCB < ucb:
            maxUCB = ucb
            ad = i
    selecteds.append(ad)
    clicks[ad] = clicks[ad] + 1
    reward = data.values[n, ad]
    rewards[ad] = rewards[ad] + reward
    total = total + reward


print('UCB algoritmasının zeki seçimlerle elde ettiği toplam ödül sayısı:')
print(total)
print('-----------------')


#Visualization
plt.hist(selecteds)
plt.show()







