"""
K - Means Clustering.

Simply put this algorithm groups data points into clusters and anomalies are those that fall outside the clusters

Especially useful if you need to quickly need to discover insights from unlabeled data.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import plotly.express as px

'''
We'll start with our own data. We'll do this by creating our own random dataset.
'''

np.random.seed(42)

# X is the feature matrix, y is the response vector.
X, y = make_blobs(n_samples=5000,                           # Number of points. These will be divided equally amongst the clusters.
                  centers=[[4, 4],[-2, -1],[2, 3],[1, 1]],  # The number of centres to generate, fixed locations
                  cluster_std=0.9)                          # The standard deviation of the clusters

plt.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.3, ec='k')


# Setting up K-Means

k_means = KMeans(init="k-means++", # This sets up in a smarter way. It spreads them out, reducing the chance of bad clustering and helps faster convergence
                 n_clusters=4,     # Number of centroids to generate (that the algo will try to find)
                 n_init=12)        # How many times the algo will run

k_means.fit(X) # fit the model / run the model

k_means_labels = k_means.labels_
print(k_means_labels) # labels for each point in the model

k_means_cluster_centers = k_means.cluster_centers_ # The coordinates of the cluster centres
print(k_means_cluster_centers)

'''
Plotting the model
'''

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels)))) # this sets unique colours for each label
ax = fig.add_subplot(1, 1, 1) # subplot

for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors): # Plots the data points and centroids.
    my_members = (k_means_labels == k) # Creates data point list, true in cluster else false
    cluster_center = k_means_cluster_centers[k] # defines the centre
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10) # plots the data points with colour col
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6) # plots centroids with darker outline

ax.set_title('KMeans')
ax.set_xticks(()) # remove ticks for cleaner look
ax.set_yticks(())

plt.show()

 # An example with k==3
k_means3 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6,4))
colors = plt.cm.tab10(np.linspace(0,1,len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.show()