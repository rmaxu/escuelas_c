# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:55:57 2019

@author: uhrma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

    
#------------------------- Secundarias ----------------------------------------
    
df_sec = pd.read_csv("secundarias.csv")
x_sec = df_sec.iloc[:, 5:]

#Scale the features
x_sec_std = StandardScaler().fit_transform(x_sec)

#Isolation Forest for outlier detection
rs=np.random.RandomState(123)
clf = IsolationForest(max_samples=500,random_state=rs,behaviour="new", contamination=.05) 
clf.fit(x_sec_std)
if_scores = clf.decision_function(x_sec_std)
if_anomalies=clf.predict(x_sec_std)

#Data without outliers
df_sec['anomalies'] = if_anomalies
df_sec = df_sec[df_sec['anomalies'] == 1]
df_sec = df_sec.drop(['anomalies'],1)
df_sec.reset_index(drop=True,inplace=True)
x_sec = df_sec.iloc[:, 5:]
x_sec_new = StandardScaler().fit_transform(x_sec)

#PCA
pca_sec = PCA()
pca_sec.fit(x_sec_new)

#Indentify how many principal components do we need
varr = pca_sec.explained_variance_ratio_
count = 1
for i in varr :
    print("Explained Variance ", count, " : ", i)
    count += 1
temp = 0
count = 1
for i in varr :
    temp += i
    print("Cumulative Variance ", count, " : ", temp)
    count += 1
    
x_sec_pca = pca_sec.transform(x_sec_new)[:,:62] #90% of the total variance

x_sec_2d = x_sec_pca[:,0:2]

#Plot the first couple of principal components
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(x_sec_2d[:,0], x_sec_2d[:,1], c='black', s=7)
ax.grid()
plt.show()

#Elbow Method

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(x_sec_pca)
    kmeanModel.fit(x_sec_pca)
    distortions.append(sum(np.min(cdist(x_sec_pca, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) /x_sec_pca.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#K-means

kmeans = KMeans(n_clusters=3).fit(x_sec_pca)
centers = kmeans.cluster_centers_
labels = kmeans.labels_.tolist()

#Plot the clusters
pca_sec_2d_df = pd.DataFrame(data = x_sec_2d, columns = ['pc1', 'pc2'])
COL = ['red', 'blue', 'green', 'yellow', 'gray', 'pink', 'violet', 
           'brown','cyan', 'magenta']
num_clusters = len(centers)
clusters = list(set(labels))
colors = COL[:num_clusters]
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(1, 1, 1)
targets = pd.DataFrame(data = labels, columns=['target'])
for color, cluster in zip(colors,clusters):
    indicesToKeep = targets['target'] == cluster
    ax.scatter(pca_sec_2d_df.loc[indicesToKeep,'pc1'], 
               pca_sec_2d_df.loc[indicesToKeep,'pc2'], 
               c=color, s=7)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Clustering', fontsize = 20)
ax.legend(clusters)
plt.show()

df_sec_c = targets.join(df_sec)

df_sec_c.to_csv("secundarias_clustered.csv", encoding='utf-8-sig', index=False)

#------------------------- Primarias ------------------------------------------

df_prim = pd.read_csv("primarias.csv")
x_prim = df_prim.iloc[:, 5:]

#Scale the features
x_prim_std = StandardScaler().fit_transform(x_prim)

#Isolation Forest for outlier detection
rs=np.random.RandomState(123)
clf = IsolationForest(max_samples=500,random_state=rs,behaviour="new", contamination=.05) 
clf.fit(x_prim_std)
if_scores = clf.decision_function(x_prim_std)
if_anomalies=clf.predict(x_prim_std)

#Data without outliers
df_prim['anomalies'] = if_anomalies
df_prim = df_prim[df_prim['anomalies'] == 1]
df_prim = df_prim.drop(['anomalies'],1)
df_prim.reset_index(drop=True,inplace=True)
x_prim = df_prim.iloc[:, 5:]
x_prim_new = StandardScaler().fit_transform(x_prim)

#PCA
pca_prim = PCA()
pca_prim.fit(x_prim_new)

#Indentify how many principal components do we need
varr = pca_prim.explained_variance_ratio_
count = 1
for i in varr :
    print("Explained Variance ", count, " : ", i)
    count += 1
temp = 0
count = 1
for i in varr :
    temp += i
    print("Cumulative Variance ", count, " : ", temp)
    count += 1
    
x_prim_pca = pca_prim.transform(x_prim_new)[:,:58] #90% of the total variance

x_prim_2d = x_prim_pca[:,0:2]

#Plot the first couple of principal components
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(x_prim_2d[:,0], x_prim_2d[:,1], c='black', s=7)
ax.grid()
plt.show()

#Elbow Method

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(x_prim_pca)
    kmeanModel.fit(x_prim_pca)
    distortions.append(sum(np.min(cdist(x_prim_pca, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) /x_prim_pca.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#K-means

kmeans = KMeans(n_clusters=3).fit(x_prim_pca)
centers = kmeans.cluster_centers_
labels = kmeans.labels_.tolist()

#Plot the clusters
pca_prim_2d_df = pd.DataFrame(data = x_prim_2d, columns = ['pc1', 'pc2'])
COL = ['red', 'blue', 'green', 'yellow', 'gray', 'pink', 'violet', 
           'brown','cyan', 'magenta']
num_clusters = len(centers)
clusters = list(set(labels))
colors = COL[:num_clusters]
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(1, 1, 1)
targets = pd.DataFrame(data = labels, columns=['target'])
for color, cluster in zip(colors,clusters):
    indicesToKeep = targets['target'] == cluster
    ax.scatter(pca_prim_2d_df.loc[indicesToKeep,'pc1'], 
               pca_prim_2d_df.loc[indicesToKeep,'pc2'], 
               c=color, s=7)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Clustering', fontsize = 20)
ax.legend(clusters)
plt.show()

df_prim_c = targets.join(df_prim)

df_prim_c.to_csv("primarias_clustered.csv", encoding='utf-8-sig', index=False)

