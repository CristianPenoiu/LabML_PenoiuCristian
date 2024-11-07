import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Importarea și pre-procesarea datelor
# Încarcă dataset-ul
data = pd.read_csv("C:\\Users\\cristi\\Desktop\\An4_sem1\\ML_Lab\\lab4\\CC GENERAL.csv")

# Eliminăm coloana CUST_ID și gestionăm valorile lipsă
data.drop("CUST_ID", axis=1, inplace=True)
data.fillna(data.mean(), inplace=True)

# Normalizare pentru clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 2. Aplicarea algoritmilor de clustering

# K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(data_scaled)

# 3. Vizualizarea clustering-ului

# PCA pentru vizualizare 2D
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(14, 6))

# Vizualizarea K-means
plt.subplot(1, 2, 1)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap="viridis", s=5)
plt.title("K-means Clustering (PCA)")

# Vizualizarea DBSCAN
plt.subplot(1, 2, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=dbscan_labels, cmap="viridis", s=5)
plt.title("DBSCAN Clustering (PCA)")

plt.show()

# t-SNE pentru vizualizare alternativă
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_scaled)

plt.figure(figsize=(14, 6))

# Vizualizarea K-means
plt.subplot(1, 2, 1)
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=kmeans_labels, cmap="viridis", s=5)
plt.title("K-means Clustering (t-SNE)")

# Vizualizarea DBSCAN
plt.subplot(1, 2, 2)
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=dbscan_labels, cmap="viridis", s=5)
plt.title("DBSCAN Clustering (t-SNE)")

plt.show()

# 4. Evaluarea performanței clusterelor

# Evaluarea K-means
kmeans_silhouette = silhouette_score(data_scaled, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(data_scaled, kmeans_labels)

# Evaluarea DBSCAN (ignoram -1 pentru zgomot)
dbscan_labels_no_noise = dbscan_labels[dbscan_labels != -1]
data_scaled_no_noise = data_scaled[dbscan_labels != -1]
dbscan_silhouette = silhouette_score(data_scaled_no_noise, dbscan_labels_no_noise)
dbscan_davies_bouldin = davies_bouldin_score(data_scaled_no_noise, dbscan_labels_no_noise)

# Afișarea rezultatelor
print("Evaluare K-means:")
print(f"Silhouette Score: {kmeans_silhouette}")
print(f"Davies-Bouldin Index: {kmeans_davies_bouldin}")

print("\nEvaluare DBSCAN:")
print(f"Silhouette Score (fără zgomot): {dbscan_silhouette}")
print(f"Davies-Bouldin Index (fără zgomot): {dbscan_davies_bouldin}")
