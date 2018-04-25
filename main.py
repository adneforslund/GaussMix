# -*- encoding: utf-8 -*-

import argparse

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import Circle
# Last datasettet
dataset = pd.read_table("seeds_dataset.txt", delim_whitespace=True)


# Fjern klassene fra datasettet
y = dataset.iloc[:,-1]
X = dataset.iloc[:,:-1]
n = 3

# Dekomponer til to akser
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)

# Gauss clustering
gauss = GaussianMixture(n_components=n)
means = KMeans(n_clusters=n)

# Datasettets klasser har verdiene 1..3, sett til 0..2
rescale_test = [i - 1 for i in y.values]
means.fit(X)
gauss.fit(X)

gaussPrediction = gauss.predict(X)
meanPrediction = means.predict(X)

def errorRate(pred, test):
    tot = 0
    err = 0
    for i in range(0, len(test)):
        if pred[i] != test[i]:
            err += 1
        tot += 1
    return float(err) / float(tot)

def flipLabels(pred):
    l = []
    for p in pred:
        if p == 0:
            l.append(1)
        elif p == 1:
            l.append(0)
        else:
            l.append(2)
    return l
            
def rotateLabels(pred, n_clusters):
    return [(p + 1) % n_clusters for p in pred]
    

def alignLabels(preds, test, n_clusters):
    smallest = 1.1
    currentRotation = 0
    flipped = 0
    pred = preds
    for c in range(0,2):
        for i in range(0, n_clusters):
            err = errorRate(pred, test)
            if err < smallest:
                smallest = err
                currentRotation = i
                flipped = c % 2
            pred = rotateLabels(pred, n_clusters)
        pred = flipLabels(pred)
        
    pred = preds
    if flipped == 1:
        pred = flipLabels(pred)

    for i in range(0, currentRotation):
        pred = rotateLabels(pred, n_clusters)
    return pred

gaussPrediction = alignLabels(gaussPrediction, rescale_test, n)
meanPrediction = alignLabels(meanPrediction, rescale_test, n)

circle1 = Circle((0.5, 0.5), color='r', alpha = 0.3, radius=5)
fig, axes = plt.subplots(1,2)
axes[0].add_artist(circle1)

fig.suptitle("KMeans and Gaussian Mixture Clustering")

axes[0].set_title("Guassian Mixture")
axes[0].scatter(x_pca[:, 0], x_pca[:, 1], c=rescale_test,
                marker='o', s=100, label="Test data")
axes[0].scatter(x_pca[:, 0], x_pca[:, 1], c=gaussPrediction,
                marker='^', s=30, label="Clustering", edgecolors='white')
axes[0].set_xlabel("Error rate: {:.2f}%".format(errorRate(gaussPrediction, rescale_test) * 100.0))
axes[0].legend()


axes[1].set_title("KMeans")
axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=rescale_test, marker='o', s=100, label="Test data")
axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=meanPrediction,
                marker='^', s=30, label="Clustering", edgecolors='white')
axes[1].set_xlabel("Error rate: {:.2f}%".format(errorRate(meanPrediction, rescale_test) * 100.0))
axes[1].legend()
plt.show()
