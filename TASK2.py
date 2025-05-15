# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('"C:\Users\VICTUS\Downloads\ifood_df.csv"')
print("First 5 rows:")
print(df.head())

# Data cleaning
print("\nMissing values:")
print(df.isnull().sum())

# Exploratory Data Analysis
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.boxplot(y='Age', data=df)
plt.title('Age Distribution')

plt.subplot(1,3,2)
sns.boxplot(y='Annual Income (k$)', data=df)
plt.title('Income Distribution')

plt.subplot(1,3,3)
sns.boxplot(y='Spending Score (1-100)', data=df)
plt.title('Spending Score Distribution')

plt.tight_layout()
plt.show()

# Select relevant features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using Elbow Method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means with optimal clusters (let's choose 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Analyze each cluster
print("\nCluster Analysis:")
for cluster in sorted(df['Cluster'].unique()):
    print(f"\nCluster {cluster}:")
    cluster_data = df[df['Cluster'] == cluster]
    print(cluster_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())