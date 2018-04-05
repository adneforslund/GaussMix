# -*- encoding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

dataset = pd.read_table("seeds_dataset.txt", delim_whitespace=True)


# Sorter vekk klasser
y = dataset.iloc[:,-1]
X = dataset.iloc[:,:-1]

# Dekomponer til to akser
pca = PCA(n_components=2)
pca.fit(X)

# Gauss clustering
gauss = GaussianMixture(n_components=3)

# Datasettet er i range 1..3, sett til 0..2
rescale_test = [i - 1 for i in y.values]

gauss.fit(X)

preds = gauss.predict(X)

print("real: {}, pred: {}".format(rescale_test, gauss.predict(X)))

err = 0
tot = 0
for i in range(0, len(rescale_test)):
    if rescale_test[i] != preds[i]:
        err += 1
    tot += 1

err = float(err) / float(tot)
print("Error rate: {}".format(err))


plt.scatter(pca.components_[0], pca.components_[1])

plt.show()