# -*- encoding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

dataset = pd.read_table("seeds_dataset.txt", delim_whitespace=True)


# Sorter vekk klasser
y = dataset.iloc[:,-1]
X = dataset.iloc[:,:-1]

# Dekomponer til to akser
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)

# Gauss clustering
gauss = GaussianMixture(n_components=3)
means = KMeans(n_clusters=3)
# Datasettet er i range 1..3, sett til 0..2
rescale_test = [i - 1 for i in y.values]
means.fit(x_pca)
gauss.fit(x_pca)

preds = gauss.predict(x_pca)
meanPredict = means.predict(x_pca)
print("real: {}, pred: {}".format(rescale_test, preds))

def errorRate(preds, verification):
    err = 0
    tot = 0
    for i in range(0, len(verification)):
        if verification[i] != preds[i]:
            err += 1
        tot += 1
    return float(err)/tot

rateGauss = errorRate(preds, rescale_test)
rateKmeans = errorRate(meanPredict, rescale_test)


print("Gauss Error rate: {}".format(rateGauss))
print("kMeans Error rate: {}".format(rateKmeans))


plt.scatter(x_pca[:,0],x_pca[:,1])
plt.scatter(x_pca[:, 0], x_pca[:, 1])
plt.show()
