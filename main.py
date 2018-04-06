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
n = 3


# Dekomponer til to akser
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)

# Gauss clustering
gauss = GaussianMixture(n_components=n)
means = KMeans(n_clusters=n)
# Datasettet er i range 1..3, sett til 0..2
rescale_test = [i - 1 for i in y.values]
means.fit(x_pca)
gauss.fit(x_pca)

preds = gauss.predict(x_pca)
meanPredict = means.predict(x_pca)

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

preds = alignLabels(preds, rescale_test, n)
meanPredict = alignLabels(meanPredict, rescale_test, n)

plt.figure(1)
plt.suptitle("KMeans and Gaussian Mixture Clustering")
plt.subplot(121)
plt.title("Guassian Mixture")
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=rescale_test, marker='o', s=70, label="Test data")
plt.scatter(x_pca[:,0], x_pca[:, 1], c=preds, marker='x', s=30, label="Clustering")
plt.xlabel("Error rate: {:.2f}%".format(errorRate(preds, rescale_test) * 100.0))
plt.legend()
plt.subplot(122)
plt.title("KMeans")
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=rescale_test, marker='o', s=70, label="Test data")
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=meanPredict, marker='x', s=30, label="Clustering")
plt.xlabel("Error rate: {:.2f}%".format(errorRate(meanPredict, rescale_test) * 100.0))
plt.legend()
plt.show()
