"""
So here we will work with a customer dataset and apply customer segmentation. (Partition the base into individuals that have similar characteristics)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import plotly.express as px
#
# cust_df = pd.read_csv("DataSets/Cust_Segmentation.csv") # this dataset is used from IBM
# print(cust_df.head())
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
cust_df
'''
Pre processing the data
'''
# As the address column contains categorical data we will drop that as k-mean doesn't work with them (Euclidean distance isn't meaningful)
cust_df = cust_df.drop(columns=['Address'], axis=1)
cust_df = cust_df.dropna()
cust_df.info()

# Standardisation
X = cust_df.values[:,1:] # Leave out customer ID (: = all rows, 1: from col 1)
Clus_dataSet = StandardScaler().fit_transform(X)

'''
Modelling 
'''

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_

cust_df["Clus_km"] = labels

cust_df.groupby('Clus_km').mean()

area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k',alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

'''
Here is a 3D interactive scatter plot for better viewing

'''
fig = px.scatter_3d(X, x=1, y=0, z=3, opacity=0.7, color=labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))  # Remove color bar, resize plot

fig.show()


# from the 3d plot we can see that are 3 clusters
# - Late career, affluent and educated.
# - Mid career, middle income
# - Early career and los imcome