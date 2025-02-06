import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler, MaxAbsScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics import davies_bouldin_score


# Load the data
data = pd.read_excel('merged.xlsx', engine='openpyxl')

# Select relevant columns
questions_avg = data[['ממוצע שאלה  1', 'ממוצע שאלה  2', 'ממוצע שאלה  3', 'ממוצע שאלה  4']]

# # חישוב מטריצת המתאמים
# correlation_matrix = data[['ממוצע שאלה  1', 'ממוצע שאלה  2', 'ממוצע שאלה  3']].corr()

# print("Correlation Matrix:")
# print(correlation_matrix)
# Scale the data

scaler = StandardScaler()
scaled_data = scaler.fit_transform(questions_avg)


pca = PCA(n_components=2)  # שמירה על שני רכיבים עיקריים
reduced_data = pca.fit_transform(scaled_data)

# Elbow Method for Optimal Clusters
# wcss_values = []
# for k in range(1, 11):  # Try different numbers of clusters
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(reduced_data)
#     wcss_values.append(kmeans.inertia_)

# Plot the Elbow graph
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, 11), wcss_values, marker='o')
# plt.title('Elbow Method for Optimal Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.show()

# # K-Means
# for k in [2,3,5,10]:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans_labels = kmeans.fit_predict(reduced_data)
#     score = silhouette_score(reduced_data, kmeans_labels)
#     calinski_harabasz = calinski_harabasz_score(reduced_data, kmeans_labels)
#     davies_bouldin_score1 = davies_bouldin_score(reduced_data, kmeans_labels)
#     print(f'n_clusters={k}, Silhouette Score={score:.2f},Calinski-Harabasz Index: {calinski_harabasz:.2f},davies_bouldin_score: {davies_bouldin_score1:.2f}')


# # ריצה עם 2 אשכולות
# kmeans_2 = KMeans(n_clusters=2, random_state=42)
# labels_2 = kmeans_2.fit_predict(reduced_data)

# ריצה עם 3 אשכולות
kmeans_3 = KMeans(n_clusters=3, random_state=42)
labels_3 = kmeans_3.fit_predict(reduced_data)

# ניתוח מספר הנקודות בכל אשכול
# clusters_2 = pd.Series(labels_2).value_counts()
# clusters_3 = pd.Series(labels_3).value_counts()

# # ציור עבור 2 אשכולות
# plt.figure(figsize=(8, 6))
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_2, cmap='viridis', s=50)
# plt.title('K-Means with 2 Clusters')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.colorbar(label='Cluster')
# plt.show()

# # ציור עבור 3 אשכולות
# plt.figure(figsize=(8, 6))
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_3, cmap='viridis', s=50)
# plt.title('K-Means with 3 Clusters')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.colorbar(label='Cluster')
# plt.show()


# #EM Clustering
# for n in range(2, 6):  # בדוק מספרי אשכולות בין 2 ל-5
#     gmm = GaussianMixture(n_components=n, random_state=42)
#     gmm_labels = gmm.fit_predict(reduced_data)
#     score = silhouette_score(reduced_data, gmm_labels)
#     calinski_harabasz = calinski_harabasz_score(reduced_data, gmm_labels)
#     print(f'n_components={n}, Silhouette Score={score:.2f},Calinski-Harabasz Index: {calinski_harabasz:.2f}')



#בדיקת סילוהט הכי טוב ל-Dbscan
# # טווחים לבדיקה
# eps_values = [0.5, 0.7, 1.0, 1.3, 1.5]
# min_samples_values = [5, 10, 15, 20]

# # בדיקת שילובים
# best_eps = None
# best_min_samples = None
# best_silhouette = -1
# calinski_harabasz1 = -1
# best_clusters = -1

# for eps in eps_values:
#     for min_samples in min_samples_values:
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         labels = dbscan.fit_predict(scaled_data)
        
        # # בדיקת מספר אשכולות
        # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #
        # # חישוב Silhouette Score (רק אם יש יותר מאשכול אחד)
        # if n_clusters > 1:
        #     silhouette_avg = silhouette_score(scaled_data, labels)
        #     if silhouette_avg > best_silhouette:
        #         calinski_harabasz1 = calinski_harabasz_score(scaled_data, labels)
        #         best_silhouette = silhouette_avg
        #         best_eps = eps
        #         best_min_samples = min_samples
        #         best_clusters = n_clusters

#  תוצאות
# print(f"Best eps: {best_eps},Best clusters {best_clusters}, Best min_samples: {best_min_samples}, Best Silhouette Score: {best_silhouette:.2f},best_calinski_harabasz: {calinski_harabasz1:.2f}")


#בדיקה Calinski הכי טוב ל-Dbscan
# טווחים לבדיקה
# eps_values = [0.5, 0.7, 1.0, 1.3, 1.5]
# min_samples_values = [5, 10, 15, 20]
#
# # בדיקת שילובים
# best_eps = None
# best_min_samples = None
# best_calinski_harabasz = -1
# best_clusters = -1
# silhouette_score1=-1
#
# for eps in eps_values:
#     for min_samples in min_samples_values:
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         labels = dbscan.fit_predict(scaled_data)
#
#         # בדיקת מספר אשכולות
#         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#
#         # חישוב Silhouette Score (רק אם יש יותר מאשכול אחד)
#         if n_clusters > 1:
#             calinski_harabasz_score1 = calinski_harabasz_score(scaled_data, labels)
#             if calinski_harabasz_score1 > best_calinski_harabasz:
#                 silhouette_score1 = silhouette_score(scaled_data, labels)
#                 best_calinski_harabasz = calinski_harabasz_score1
#                 best_eps = eps
#                 best_min_samples = min_samples
#                 best_clusters = n_clusters
#
# # תוצאות
# print(f"Best eps: {best_eps},Best clusters {best_clusters}, Best min_samples: {best_min_samples},best_calinski_harabasz: {best_calinski_harabasz:.2f},silhouette_score1: {silhouette_score1:.2f}")


# #Agglomerative Clustering
# for n_clusters in range(2, 6):  # בדיקה בין 2 ל-5 אשכולות
#     agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
#     agg_labels = agg_clustering.fit_predict(reduced_data)
#
#     silhouette_avg = silhouette_score(reduced_data, agg_labels)
#     calinski_harabasz = calinski_harabasz_score(reduced_data, agg_labels)
#
#     print(f"n_clusters={n_clusters}, Silhouette Score={silhouette_avg:.2f}, Calinski-Harabasz Index={calinski_harabasz:.2f}")


### Gaussian Mixture Models
gmm = GaussianMixture(n_components=3, random_state=42)
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



