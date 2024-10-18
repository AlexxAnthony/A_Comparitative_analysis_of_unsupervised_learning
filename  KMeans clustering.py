#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing standard libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# ignore warnings


# In[4]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


# Read Dataset


# In[7]:


df=pd.read_csv("riceClassification.csv")
df


# In[8]:


# Exploratory Data Analysis


# In[10]:


df.shape


# In[11]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


# we will drop id and class


# In[14]:


df.drop(["id","Class"],axis=1,inplace=True)
df


# In[15]:


plt.figure(figsize=(20,20))
plt.title("Heatmap Corralation")
sns.heatmap(data=pd.get_dummies(df).corr(),annot=True)


# In[16]:


df.info()


# In[17]:


#Feature scaling


# In[18]:


x=df


# In[20]:


cols=x.columns


# In[21]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[22]:


ms= MinMaxScaler()
x=ms.fit_transform(x)


# In[23]:


x=pd.DataFrame(x,columns=[cols])


# In[24]:


x.head()


# In[30]:


from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state = 0)
    kmeans.fit(x)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# In[40]:


y_predicted=kmeans.fit_predict(df[["Area","MajorAxisLength","MinorAxisLength","Eccentricity","ConvexArea","EquivDiameter","Extent","Perimeter","Roundness","AspectRation"]])
y_predicted


# In[41]:


y=y_predicted


# In[42]:


# Kmeans with k=2


# In[43]:


kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(x)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[45]:


# We recieved accuracy score of 67% from this algorithm


# In[47]:


from sklearn.preprocessing import StandardScaler


# In[52]:


selected_columns = ["MajorAxisLength", "Eccentricity"]
df_selected = df[selected_columns]

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(df_selected)

# Choose the number of clusters (k)
k = 2  # You need to decide the appropriate number of clusters for your data

# Apply KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(data_standardized)

# Compute silhouette score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(data_standardized, labels)
print(f"Silhouette Score for KMeans clustering: {silhouette_avg}")


# In[ ]:




