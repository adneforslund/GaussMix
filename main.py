# -*- encoding: utf-8 -*-

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import Circle, Ellipse
from scipy import linalg

def run(display_intermediaries):    
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

    if display_intermediaries:
        fig, axes = plt.subplots(7, 7, figsize = (12, 12),
                                 subplot_kw = {'xticks': (), 'yticks': ()})
        for x in range(0,7):
            for y in range(0,7):
                axes[x,y].scatter(X.iloc[:, x], X.iloc[:, y], s=40)

        fig.suptitle("Features of seed dataset")
        plt.show()
        
    gaussPrediction = alignLabels(gaussPrediction, rescale_test, n)
    meanPrediction = alignLabels(meanPrediction, rescale_test, n)

    fig, axes = plt.subplots(1,2)
    
    fig.suptitle("KMeans and Gaussian Mixture Clustering")
    means.fit(x_pca)
    gauss.fit(x_pca)
    
    # Tegn KMeans cluster overlays
    C = means.cluster_centers_
    colors = ['r', 'g', 'b']
    for i, color in zip(C, colors):
        circle = Circle((i[0], i[1]),radius=2, alpha=0.3, color=color)
        axes[1].add_artist(circle)
        center_circle = Circle((i[0], i[1]), radius = 0.1, alpha = 1, color = "black")
        axes[1].add_artist(center_circle)

    # Kode fra http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html
    for i, (mean, covar, color) in enumerate(zip(gauss.means_, gauss.covariances_, colors)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v) 
        # Plot an ellipse to show the Gaussian component
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_alpha(0.2)
        axes[0].add_artist(ell)
        print(v)
        
    
    axes[0].set_title("Guassian Mixture")
    axes[0].scatter(x_pca[:, 0], x_pca[:, 1], c=rescale_test,
                    marker='o', s=100, label="Test data")
    axes[0].scatter(x_pca[:, 0], x_pca[:, 1], c=gaussPrediction,
                    marker='^', s=30, label="Clustering", edgecolors='white')
    # axes[0].set_xlabel("Error rate: {:.2f}%".format(errorRate(gaussPrediction, rescale_test) * 100.0))
    axes[0].legend()

    axes[1].set_title("KMeans")
    axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=rescale_test, marker='o', s=100, label="Test data")
    axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=meanPrediction,
                    marker='^', s=30, label="Clustering", edgecolors='white')
    # axes[1].set_xlabel("Error rate: {:.2f}%".format(errorRate(meanPrediction, rescale_test) * 100.0))
    axes[1].legend()
    plt.show()
    

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Display clustering for seed dataset")
    parser.add_argument("-x", "--extra", help="Display extra plots", action="store_true")
    args = parser.parse_args()
    display_intermediaries = args.extra
    run(display_intermediaries)
