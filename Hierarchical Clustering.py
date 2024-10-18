#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# In[2]:


df = pd.read_csv('riceClassification.csv')
df


# In[3]:


df.drop(["id","Class"],axis=1,inplace=True)


# In[4]:


df.info()


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


scaler=StandardScaler()
scaler.fit(df)


# In[7]:


scaled_data=scaler.transform(df)
scaled_data


# In[8]:


from sklearn.decomposition import PCA


# In[9]:


pca=PCA(n_components=2)


# In[10]:


pca.fit(scaled_data)


# In[11]:


x_pca=pca.transform(scaled_data)


# In[12]:


scaled_data.shape


# In[13]:


x_pca.shape


# In[14]:


x_pca


# In[15]:


x_pca


# In[17]:


X=x_pca


# In[25]:


agg_clustering = AgglomerativeClustering(n_clusters=2)
cluster_assignments = agg_clustering.fit_predict(X)

# Plot the data points with colored cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', edgecolors='k', s=50)

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hierarchical Agglomerative Clustering with Colored Cluster Assignments')

# Show the plot
plt.show()


# In[20]:


# Dendrogram
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()


# In[21]:


cluster_assignments = agg_clustering.fit_predict(X)


# In[22]:


from sklearn.metrics import silhouette_score


# In[23]:


silhouette_avg = silhouette_score(X, cluster_assignments)
print(f"Silhouette Score (Hierarchical Clustering): {silhouette_avg}")


# In[ ]:




