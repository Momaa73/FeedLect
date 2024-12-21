import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler, MaxAbsScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from kneed import KneeLocator
import matplotlib.pyplot as plt


# Load the data
data = pd.read_excel('תשפב מנוקה.xlsx', engine='openpyxl')

# Select relevant columns
questions_avg = data[['ממוצע שאלה  1', 'ממוצע שאלה  2', 'ממוצע שאלה  3', 'ממוצע שאלה  4']]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(questions_avg)

# Apply PCA to reduce to 2 dimensions also reduce the noise
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Elbow Method for Optimal Clusters
wcss_values = []
for k in range(1, 11):  # Try different numbers of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_data)
    wcss_values.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss_values, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Automatically find the optimal number of clusters so we don't need to change it.
knee_locator = KneeLocator(range(1, 11), wcss_values, curve="convex", direction="decreasing")
optimal_clusters = knee_locator.knee
print(f"Optimal number of clusters (Elbow): {optimal_clusters}")

# K-Means
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_data)
data['KMeans_Cluster'] = kmeans_labels

# plot with centroids for each cluster
centroids = kmeans.cluster_centers_
plt.scatter(pca_data [:, 0], pca_data [:, 1], c=kmeans_labels, cmap='viridis', marker='o', s=50, label='Data points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


#Print the WCSS for the chosen number of clusters (by the elbow method)
print(f"WCSS for optimal number of clusters ({optimal_clusters}): {kmeans.inertia_}")

# Hierarchical Clustering
linked = linkage(pca_data, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=data.index, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram for Hierarchical Clustering With PCA')
plt.show()

# Assign clusters for hierarchical clustering
h_clusters = fcluster(linked, t=optimal_clusters, criterion='maxclust')
data['Hierarchical_Cluster'] = h_clusters

# Save the clustered data with both methods in the same file
data.to_excel('clustered_data_B.xlsx', index=False)


# Perform Agglomerative clustering withOut the dendogarm usage and can
# also be Changed the why of matrix distance function calculated

agglo_clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='average',
metric='manhattan')
agglo_labels = agglo_clustering.fit_predict(pca_data)
data['Agglomerative_Cluster Metric M'] = agglo_labels

# Plot Agglomerative Clustering
plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=agglo_labels, cmap='viridis', marker='o', s=50, label='Data points')
plt.title("Agglomerative Clustering With PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# we want to check if the clusteres are similar, so we added a column with the 2 clusteres from each algorithm. 1,0 is not 0,1!
data['Clusters_Combined'] = data['KMeans_Cluster'].astype(str) + ',' + data['Hierarchical_Cluster'].astype(str)
# Save the updated data to a CSV file
data.to_excel('clustered_data_B.xlsx', index=False)

# Count the occurrences of each cluster pair in the 'Clusters_Combined' column
cluster_counts = data['Clusters_Combined'].value_counts()

# Save the results to a text file
with open('pair_countsB.txt', 'w') as file:
    for pair, count in cluster_counts.items():
        file.write(f"{pair} - {count}\n")

###################### other tests

# Regular Use "linkage='ward'" euclidian distance
agglo_clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
agglo_labels = agglo_clustering.fit_predict(pca_data)
data['Agglomerative_Cluster'] = agglo_labels

