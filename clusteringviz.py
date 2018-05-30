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
from sklearn.metrics import silhouette_score

def run(display_intermediaries, path, save_path):
    # Load the dataset
    dataset = pd.read_table(path, delim_whitespace=True)
    
    # Remove classes from dataset
    y = dataset.iloc[:,-1]
    X = dataset.iloc[:,:-1]
    number_of_clusters = 3

    # Decompose to two axes
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(X)

    # Gauss clustering
    gauss = GaussianMixture(n_components=number_of_clusters, random_state=0, reg_covar=0.00001, covariance_type='full', tol=0.00001, max_iter=100, n_init=1)
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0, n_init=1, tol=0.0001, max_iter=300)

    # Change the dataset class values from 1..3 to 0..2
    rescale_test = [i - 1 for i in y.values]
    kmeans.fit(X)
    gauss.fit(X)

    kmean_prediction = kmeans.predict(X)
    gauss_prediction = gauss.predict(X)

    kmeans_silhouette = silhouette_score(X, kmean_prediction, random_state=0 )
    gaus_silhouette = silhouette_score(X, gauss_prediction, random_state=0)

    if display_intermediaries:
        fig, axes = plt.subplots(7, 7, figsize = (12, 12),
                                 subplot_kw = {'xticks': (), 'yticks': ()})
        for x in range(0,7):
            for y in range(0,7):
                axes[x,y].scatter(X.iloc[:, x], X.iloc[:, y], s=40)

        fig.suptitle("Features of seed dataset")
        plt.show()
        
    gauss_prediction = align_labels(gauss_prediction, rescale_test, number_of_clusters)
    kmean_prediction = align_labels(kmean_prediction, rescale_test, number_of_clusters)

    fig, axes = plt.subplots(1,2)
    
    fig.suptitle("KMeans and Gaussian Mixture Clustering")
    kmeans.fit(x_pca)
    gauss.fit(x_pca)
    
    # Draw KMeans cluster overlays
    cluster_centers = kmeans.cluster_centers_
    colors = ['r', 'g', 'b']
    for i, color in zip(cluster_centers, colors):
        circle = Circle((i[0], i[1]),radius=2, alpha=0.2, color=color)
        axes[1].add_artist(circle)

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
    axes[0].set_xlabel("Error rate: {:.2f}% \nSilhouette score: {:.2f}".format(error_rate(gauss_prediction, rescale_test) * 100.0, gaus_silhouette))
    axes[0].legend()

    axes[1].set_title("KMeans")
    axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=rescale_test, marker='o', s=100, label="Test data")
    axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=kmean_prediction,
                    marker='^', s=30, label="Clustering", edgecolors='white')
    axes[1].set_xlabel("Error rate: {:.2f}% \nSilhouette score: {:.2f}".format(error_rate(kmean_prediction, rescale_test) * 100.0, kmeans_silhouette))
    axes[1].legend()

    if save_path != None:
        fig.savefig(save_path)

    plt.show()


    # Calculate the error rate when there are known classes for the clusters
def error_rate(pred, test):
    total = 0
    errors = 0
    for i in range(0, len(test)):
        if pred[i] != test[i]:
            errors += 1
        total += 1
    return float(errors) / float(total)

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
    smallest_error = 1.1
    current_rotation = 0
    flipped = 0

    # Set the list positions to the default state
    pred = preds
    for do_flip in range(0,2):
        for current_cluster in range(0, n_clusters):
            error = error_rate(pred, test)
            if error < smallest_error:
                smallest_error = error
                current_rotation = current_cluster
                flipped = do_flip % 2
            pred = rotate_labels(pred, n_clusters)
        pred = flip_labels(pred)

    # Reset the list positions and rotate them to the best prediction
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
