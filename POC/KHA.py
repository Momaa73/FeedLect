import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, Birch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from kneed import KneeLocator
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('תשפא מנוקה.xlsx', engine='openpyxl')

# Select relevant columns
questions_avg = data[['ממוצע שאלה  1', 'ממוצע שאלה  2', 'ממוצע שאלה  3', 'ממוצע שאלה  4']]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(questions_avg)

# Apply PCA to reduce to 2 dimensions also reduce the noise
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Elbow Method for Optimal Clusters
wcss_values = []
for k in range(1, 11):  # Try different numbers of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reduced_data)
    wcss_values.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss_values, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (SSE)')
plt.show()

# Automatically find the optimal number of clusters so we don't need to change it.
knee_locator = KneeLocator(range(1, 11), wcss_values, curve="convex", direction="decreasing")
optimal_clusters = knee_locator.knee
print(f"Optimal number of clusters (Elbow): {optimal_clusters}")

# K-Means
kmeans = KMeans(n_clusters=optimal_clusters)
kmeans_labels = kmeans.fit_predict(reduced_data)
data['KMeans_Cluster'] = kmeans_labels

# Print the WCSS for the chosen number of clusters (by the elbow method)
print(f"WCSS for optimal number of clusters ({optimal_clusters}): {kmeans.inertia_}")

# Hierarchical Clustering
linked = linkage(reduced_data, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=data.index, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()

# Assign clusters for hierarchical clustering
h_clusters = fcluster(linked, t=optimal_clusters, criterion='maxclust')
data['Hierarchical_Cluster'] = h_clusters

# Save the clustered data with both methods in the same file
data.to_excel('clustered_data_A.xlsx', index=False)

# we want to check if the clusteres are similar, so we added a column with the 2 clusteres from each algorithm. 1,0 is not 0,1!
data['Clusters_Combined'] = data['KMeans_Cluster'].astype(str) + ',' + data['Hierarchical_Cluster'].astype(str)
# Save the updated data to a CSV file
data.to_csv('clustered_data_A.csv', index=False)

# Count the occurrences of each cluster pair in the 'Clusters_Combined' column
cluster_counts = data['Clusters_Combined'].value_counts()

# Save the results to a text file
with open('pair_countsA.txt', 'w') as file:
    for pair, count in cluster_counts.items():
        file.write(f"{pair} - {count}\n")




### Gaussian Mixture Models
gmm = GaussianMixture(n_components=3, random_state=42,covariance_type="spherical")
gmm.fit(reduced_data)
labels = gmm.predict(reduced_data)

# Compute Silhouette Score
score = silhouette_score(reduced_data, labels)
print(f"Silhouette Score gmm: {score}")

# Compute Calinski-Harabasz Score
calinski_harabasz = calinski_harabasz_score(reduced_data, labels)
print(f"Calinski-Harabasz Score GMM: {calinski_harabasz}")

### Initialize K-Means++ with desired parameters
kmeans_plus = KMeans(n_clusters=3, init='k-means++',random_state=42)

# Fit the model and predict cluster labels
kmeans_labels = kmeans_plus.fit_predict(reduced_data)

# Compute Silhouette Score
silhouette = silhouette_score(reduced_data, kmeans_labels)
print(f"Silhouette Score_k++: {silhouette}")

# Compute Calinski-Harabasz Score
calinski_harabasz = calinski_harabasz_score(reduced_data, kmeans_labels)
print(f"Calinski-Harabasz Score_k++: {calinski_harabasz}")


### Initialize Spectral Clustering
spectral_kmeans = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42)

# Fit the model and predict cluster labels
spectral_labels = spectral_kmeans.fit_predict(reduced_data)

# Compute Silhouette Score
spectral_silhouette = silhouette_score(reduced_data, spectral_labels)
print(f"Silhouette Score (Spectral K-Means): {spectral_silhouette}")

# Compute Calinski-Harabasz Score
spectral_calinski_harabasz = calinski_harabasz_score(reduced_data, spectral_labels)
print(f"Calinski-Harabasz Score (Spectral K-Means): {spectral_calinski_harabasz}")


### Initialize and fit BIRCH
birch = Birch(n_clusters=3, threshold=1, branching_factor=50)
birch_labels = birch.fit_predict(scaled_data)

# Evaluate clustering using Silhouette Score
silhouette = silhouette_score(scaled_data, birch_labels)
print(f"Silhouette Score (BIRCH): {silhouette}")

# Evaluate clustering using Calinski-Harabasz Score
calinski_harabasz = calinski_harabasz_score(scaled_data, birch_labels)
print(f"Calinski-Harabasz Score (BIRCH): {calinski_harabasz}")
