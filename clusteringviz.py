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

def run(display_intermediaries, path, save_path):
    # Load the dataset
    dataset = pd.read_table(path, delim_whitespace=True)
    
    # Remove classes from dataset
    y = dataset.iloc[:,-1]
    X = dataset.iloc[:,:-1]
    n = 3

    # Decompose to two axises
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(X)

    # Gauss clustering
    gauss = GaussianMixture(n_components=n, random_state=0)
    kmeans = KMeans(n_clusters=n, random_state=0)

    # Change the dataset class values from 1..3 to 0..2
    rescale_test = [i - 1 for i in y.values]
    kmeans.fit(X)
    gauss.fit(X)

    mean_prediction = kmeans.predict(X)
    gauss_prediction = gauss.predict(X)

    if display_intermediaries:
        fig, axes = plt.subplots(7, 7, figsize = (12, 12),
                                 subplot_kw = {'xticks': (), 'yticks': ()})
        for x in range(0,7):
            for y in range(0,7):
                axes[x,y].scatter(X.iloc[:, x], X.iloc[:, y], s=40)

        fig.suptitle("Features of seed dataset")
        plt.show()
        
    gauss_prediction = align_labels(gauss_prediction, rescale_test, n)
    mean_prediction = align_labels(mean_prediction, rescale_test, n)

    fig, axes = plt.subplots(1,2)
    
    fig.suptitle("KMeans and Gaussian Mixture Clustering")
    kmeans.fit(x_pca)
    gauss.fit(x_pca)
    
    # Draw KMeans cluster overlays
    C = kmeans.cluster_centers_
    colors = ['r', 'g', 'b']
    for i, color in zip(C, colors):
        circle = Circle((i[0], i[1]),radius=2, alpha=0.2, color=color)
        axes[1].add_artist(circle)
        center_circle = Circle((i[0], i[1]), radius = 0.1, alpha = 1, color = "black")
        axes[1].add_artist(center_circle)

    # Code from http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html
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
        
    axes[0].set_title("Guassian Mixture")
    axes[0].scatter(x_pca[:, 0], x_pca[:, 1], c=rescale_test,
                    marker='o', s=100, label="Test data")
    axes[0].scatter(x_pca[:, 0], x_pca[:, 1], c=gauss_prediction,
                    marker='^', s=30, label="Clustering", edgecolors='white')
    axes[0].set_xlabel("Error rate: {:.2f}%".format(error_rate(gauss_prediction, rescale_test) * 100.0))
    axes[0].legend()

    axes[1].set_title("KMeans")
    axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=rescale_test, marker='o', s=100, label="Test data")
    axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=mean_prediction,
                    marker='^', s=30, label="Clustering", edgecolors='white')
    axes[1].set_xlabel("Error rate: {:.2f}%".format(error_rate(mean_prediction, rescale_test) * 100.0))
    axes[1].legend()
    
    if save_path != None:
        fig.savefig(save_path)
        
    plt.show()


    # Calculate the error rate when there are known classes for the clusters
def error_rate(pred, test):
    tot = 0
    err = 0
    for i in range(0, len(test)):
        if pred[i] != test[i]:
            err += 1
        tot += 1
    return float(err) / float(tot)

    # Changes labels for clusters
def flip_labels(pred):
    labels = []
    for prediction in pred:
        if prediction == 0:
            labels.append(1)
        elif prediction == 1:
            labels.append(0)
        else:
            labels.append(2)
    return labels


def rotate_labels(pred, n_clusters):
    return [(p + 1) % n_clusters for p in pred]
    

    # Tries to match labels from the dataset with the real labels
def align_labels(preds, test, n_clusters):
    smallest = 1.1
    current_rotation = 0
    flipped = 0
    pred = preds
    for do_flip in range(0,2):
        for current_cluster in range(0, n_clusters):
            err = error_rate(pred, test)
            if err < smallest:
                smallest = err
                current_rotation = current_cluster
                flipped = do_flip % 2
            pred = rotate_labels(pred, n_clusters)
        pred = flip_labels(pred)
        
    pred = preds
    if flipped == 1:
        pred = flip_labels(pred)

    for i in range(0, current_rotation):
        pred = rotate_labels(pred, n_clusters)
        return pred

    # handles commandline arguments and runs program
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Display clustering for seed dataset")
    parser.add_argument("-f", "--file", dest="my_path",
                        help="Give a path to your dataset", required=True
                        )
    parser.add_argument("-x", "--extra", help="Display extra plots", action="store_true", required = False)
    parser.add_argument("-s", "--save", help="Save plots as png, given pathname", required = False, dest="save_path"  )
    args = parser.parse_args()
    path = args.my_path
    save_path = args.save_path
    
    display_intermediaries = args.extra
    run(display_intermediaries, path, save_path)
