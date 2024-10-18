#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('riceClassification.csv')

df


# In[10]:


df.drop(["id","Class"],axis=1,inplace=True)
df


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


scaler=StandardScaler()
scaler.fit(df)


# In[13]:


scaled_data=scaler.transform(df)
scaled_data


# In[14]:


from sklearn.decomposition import PCA


# In[15]:


pca=PCA(n_components=2)


# In[16]:


pca.fit(scaled_data)


# In[17]:


x_pca=pca.transform(scaled_data)


# In[18]:


scaled_data.shape


# In[19]:


x_pca.shape


# In[20]:


x_pca


# In[21]:


X=x_pca


# In[23]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_assignments = dbscan.fit_predict(X)

# Plot the data points with colored cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', edgecolors='k', s=50)

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering with Colored Cluster Assignments')

# Show the plot
plt.show()


# In[24]:


df['labels'] = dbscan.labels_

df


# In[25]:


# accurate epsilon and min samples value


# In[26]:


epsilons = np.linspace(0.01, 1, num=15)
epsilons


# In[27]:


min_samples = np.arange(2, 20, step=3)
min_samples


# In[28]:


import itertools

combinations = list(itertools.product(epsilons, min_samples))
combinations


# In[29]:


N = len(combinations)
N


# In[30]:


from sklearn.metrics import silhouette_score as ss


# In[31]:


def get_scores_and_labels(combinations, X):
  scores = []
  all_labels_list = []

  for i, (eps, num_samples) in enumerate(combinations):
    dbscan_cluster_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
    labels = dbscan_cluster_model.labels_
    labels_set = set(labels)
    num_clusters = len(labels_set)
    if -1 in labels_set:
      num_clusters -= 1
    
    if (num_clusters < 2) or (num_clusters > 50):
      scores.append(-10)
      all_labels_list.append('bad')
      c = (eps, num_samples)
      print(f"Combination {c} on iteration {i+1} of {N} has {num_clusters} clusters. Moving on")
      continue
    
    scores.append(ss(X, labels))
    all_labels_list.append(labels)
    print(f"Index: {i}, Score: {scores[-1]}, Labels: {all_labels_list[-1]}, NumClusters: {num_clusters}")

  best_index = np.argmax(scores)
  best_parameters = combinations[best_index]
  best_labels = all_labels_list[best_index]
  best_score = scores[best_index]

  return {'best_epsilon': best_parameters[0],
          'best_min_samples': best_parameters[1], 
          'best_labels': best_labels,
          'best_score': best_score}

best_dict = get_scores_and_labels(combinations, X)


# In[39]:


best_dict


# In[40]:


dbscan = DBSCAN(eps=0.7171428571428572, min_samples=5)
cluster_assignments = dbscan.fit_predict(X)

# Plot the data points with colored cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', edgecolors='k', s=50)

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering with Colored Cluster Assignments')

# Show the plot
plt.show()


# In[50]:


df['labels'] = best_dict['best_labels']

df['labels'].value_counts()


# In[42]:


cluster_assignments = dbscan.fit_predict(X)


# In[43]:


from sklearn.metrics import silhouette_score


# In[44]:


silhouette_avg = silhouette_score(X, cluster_assignments)
print(f"Silhouette Score (DBSCAN): {silhouette_avg}")


# In[45]:


import numpy as np

# Assuming 'labels' is an array containing the cluster labels assigned by DBSCAN
# 'labels' should be a 1D array with each element representing the cluster label for corresponding data point

# Generate some example data for demonstration
labels = np.array([0, 1, -1, 0, 1, 1, -1, 0, -1, 1])

# Extract labeled and unlabeled data points based on cluster labels
labeled_indices = np.where(labels != -1)[0]  # Indices of labeled data points
unlabeled_indices = np.where(labels == -1)[0]  # Indices of unlabeled data points

# Separate data into labeled and unlabeled subsets
labeled_data = 'riceClassification.csv'[labeled_indices]  # Replace 'your_data' with your actual data array
unlabeled_data = 'riceClassification.csv'[unlabeled_indices]

# Example usage:
print("Labeled Data:")
print(labeled_data)
print("Unlabeled Data:")
print(unlabeled_data)


# In[46]:


import numpy as np

# Assuming 'labels' is an array containing the cluster labels assigned by DBSCAN
# 'labels' should be a 1D array with each element representing the cluster label for corresponding data point

# Generate some example data for demonstration
labels = np.array([0, 1, -1, 0, 1, 1, -1, 0, -1, 1])

# Extract labeled and unlabeled data points based on cluster labels
labeled_indices = np.where(labels != -1)[0]  # Indices of labeled data points
unlabeled_indices = np.where(labels == -1)[0]  # Indices of unlabeled data points

# Separate data into labeled and unlabeled subsets
labeled_data = 'riceClassification.csv'[labeled_indices]  # Replace 'your_data' with your actual data array
unlabeled_data = 'riceClassification.csv'[unlabeled_indices]

# Example usage:
print("Labeled Data:")
print(labeled_data)
print("Unlabeled Data:")
print(unlabeled_data)


# In[47]:


import pandas as pd
import numpy as np

# Load your data from the CSV file
data = pd.read_csv("riceClassification.csv")

# Assuming 'labels' is an array containing the cluster labels assigned by DBSCAN
# 'labels' should be a 1D array with each element representing the cluster label for corresponding data point

# Example cluster labels array (replace this with your actual cluster labels array)
labels = np.array([0, 1, -1, 0, 1, 1, -1, 0, -1, 1])

# Extract labeled and unlabeled data points based on cluster labels
labeled_indices = np.where(labels != -1)[0]  # Indices of labeled data points
unlabeled_indices = np.where(labels == -1)[0]  # Indices of unlabeled data points

# Separate data into labeled and unlabeled subsets
labeled_data = data.iloc[labeled_indices]  # Extract labeled data points
unlabeled_data = data.iloc[unlabeled_indices]  # Extract unlabeled data points

# Example usage:
print("Labeled Data:")
print(labeled_data)
print("Unlabeled Data:")
print(unlabeled_data)


# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Assuming you have 'labeled_data' and 'unlabeled_data' as labeled and unlabeled subsets

# Split labeled data into train and validation sets
X_labeled = labeled_data.drop(columns=['labels'])  # Assuming 'label' is the target variable
y_labeled = labeled_data['labels']
X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

# Train a supervised learning model on the labeled data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Use the trained model to predict labels for the unlabeled data
X_unlabeled = unlabeled_data.drop(columns=['labels'])  # Assuming 'label' is the target variable
pseudo_labels = model.predict(X_unlabeled)

# Combine the unlabeled data with the predicted labels
pseudo_labeled_data = unlabeled_data.copy()
pseudo_labeled_data['labels'] = pseudo_labels

# Concatenate the pseudo-labeled data with the labeled data
augmented_data = pd.concat([labeled_data, pseudo_labeled_data])

# Retrain the model on the augmented dataset
X_augmented = augmented_data.drop(columns=['labels'])  # Assuming 'label' is the target variable
y_augmented = augmented_data['labels']
model.fit(X_augmented, y_augmented)

# Evaluate the model performance on the validation set
accuracy = model.score(X_val, y_val)
print("Validation accuracy after pseudo-labeling:", accuracy)


# In[6]:


# Assuming 'target_column' is the actual column name containing the labels
X_labeled = labeled_data.drop(columns=['target_column'])  
y_labeled = labeled_data['target_column']


# In[52]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Assuming X contains your feature vectors and cluster_assignments contains the cluster labels assigned by DBSCAN

# Split data into labeled and unlabeled subsets based on cluster assignments
labeled_indices = np.where(cluster_assignments != -1)[0]  # Indices of labeled data points
unlabeled_indices = np.where(cluster_assignments == -1)[0]  # Indices of unlabeled data points
X_labeled = X[labeled_indices]  # Labeled data points
y_labeled = cluster_assignments[labeled_indices]  # Cluster labels for labeled data points
X_unlabeled = X[unlabeled_indices]  # Unlabeled data points

# Split labeled data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

# Train a supervised learning model on the labeled data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Use the trained model to predict labels for the unlabeled data
pseudo_labels = model.predict(X_unlabeled)

# Combine the unlabeled data with the predicted labels
pseudo_labeled_data = np.column_stack((X_unlabeled, pseudo_labels))

# Concatenate the pseudo-labeled data with the labeled data
augmented_data = np.vstack((X_labeled, pseudo_labeled_data))

# Retrain the model on the augmented dataset
X_augmented = augmented_data[:, :-1]  # Features
y_augmented = augmented_data[:, -1]  # Labels
model.fit(X_augmented, y_augmented)

# Evaluate the model performance on the validation set
accuracy = model.score(X_val, y_val)
print("Validation accuracy after pseudo-labeling:", accuracy)


# In[55]:


import numpy as np

# Example arrays with different sizes along axis 1
array1 = np.random.rand(5, 2)  # Array with 5 rows and 2 columns
array2 = np.random.rand(5, 3)  # Array with 5 rows and 3 columns

# Concatenate arrays along axis 1
# This will raise an error because the dimensions along axis 1 don't match
concatenated_array = np.concatenate((array1, array2), axis=1)


# In[56]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Assuming X contains your feature vectors and cluster_assignments contains the cluster labels assigned by DBSCAN

# Split data into labeled and unlabeled subsets based on cluster assignments
labeled_indices = np.where(cluster_assignments != -1)[0]  # Indices of labeled data points
unlabeled_indices = np.where(cluster_assignments == -1)[0]  # Indices of unlabeled data points
X_labeled = X[labeled_indices]  # Labeled data points
y_labeled = cluster_assignments[labeled_indices]  # Cluster labels for labeled data points
X_unlabeled = X[unlabeled_indices]  # Unlabeled data points

# Split labeled data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

# Train a supervised learning model on the labeled data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Use the trained model to predict labels for the unlabeled data
pseudo_labels = model.predict(X_unlabeled)

# Combine the unlabeled data with the predicted labels
pseudo_labeled_data = np.column_stack((X_unlabeled, pseudo_labels))

# Concatenate the pseudo-labeled data with the labeled data
augmented_data = np.vstack((X_labeled, pseudo_labeled_data))

# Retrain the model on the augmented dataset
X_augmented = augmented_data[:, :-1]  # Features
y_augmented = augmented_data[:, -1]  # Labels
model.fit(X_augmented, y_augmented)

# Evaluate the model performance on the validation set
accuracy = model.score(X_val, y_val)
print("Validation accuracy after pseudo-labeling:", accuracy)


# In[ ]:




