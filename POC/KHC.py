import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from kneed import KneeLocator
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('תשפג מנוקה.xlsx', engine='openpyxl')

# Select relevant columns
questions_avg = data[['ממוצע שאלה  1', 'ממוצע שאלה  2', 'ממוצע שאלה  3', 'ממוצע שאלה  4']]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(questions_avg)

# Elbow Method for Optimal Clusters
wcss_values = []
for k in range(1, 11):  # Try different numbers of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
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
kmeans_labels = kmeans.fit_predict(scaled_data)
data['KMeans_Cluster'] = kmeans_labels

# Print the WCSS for the chosen number of clusters (by the elbow method)
print(f"WCSS for optimal number of clusters ({optimal_clusters}): {kmeans.inertia_}")

# Hierarchical Clustering
linked = linkage(scaled_data, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=data.index, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()

# Assign clusters for hierarchical clustering
h_clusters = fcluster(linked, t=optimal_clusters, criterion='maxclust')
data['Hierarchical_Cluster'] = h_clusters

# Save the clustered data with both methods in the same file
data.to_csv('clustered_data_C.csv', index=False)

# we want to check if the clusteres are similar, so we added a column with the 2 clusteres from each algorithm. 1,0 is not 0,1!
data['Clusters_Combined'] = data['KMeans_Cluster'].astype(str) + ',' + data['Hierarchical_Cluster'].astype(str)
# Save the updated data to a CSV file
data.to_csv('clustered_data_C.csv', index=False)

# Count the occurrences of each cluster pair in the 'Clusters_Combined' column
cluster_counts = data['Clusters_Combined'].value_counts()

# Save the results to a text file
with open('pair_countsC.txt', 'w') as file:
    for pair, count in cluster_counts.items():
        file.write(f"{pair} - {count}\n")


