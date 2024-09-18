import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the storm surge data
with h5py.File('NACCStimeS_Output.mat', 'r') as file:
    resp_final_dis = np.array(file['Resp'])  # Shape: (170, 12603, 535)
print("Original storm surge data shape:", resp_final_dis.shape)

# Reshape the data from (170, 12603, 535) to (12603, 535 * 170)
reshaped_data = np.reshape(np.transpose(resp_final_dis, (1, 2, 0)), (12603, 535 * 170))
print("Reshaped data shape:", reshaped_data.shape)  # Expected output: (12603, 90950)

# Load the grid data (latitude and longitude) and transpose it as requested
with h5py.File('NACCS.mat', 'r') as file:
    grid_table = np.array(file['grid'])  
    grid_table_transposed = np.transpose(grid_table)
print(f"Shape of grid_table after transposition: {grid_table_transposed.shape}")  

latitudes = grid_table_transposed[:, 0]  # First column is latitude
longitudes = grid_table_transposed[:, 1]  # Second column is longitude
print(f"Latitude shape: {latitudes.shape}, Longitude shape: {longitudes.shape}")  # Expected shape: (12603,)

# Perform PCA
pca = PCA()
eigenvectors = pca.fit_transform(reshaped_data)  
print('The shape of eigenvectors is:', eigenvectors.shape)  # Should be (12603, n_components)
eigenvectors = eigenvectors[:, :10]  # First 10 components for dimensionality reduction

eigenvalues = pca.explained_variance_[:10]  
print(f"First 10 eigenvalues: {eigenvalues}")

# Define a_n (scaling factor)
a_n = 1  # Replace with actual value if necessary

# Calculate 'a' using the formula
a = (np.sum(eigenvalues) / np.sqrt(2)) * a_n
print(f"Calculated 'a': {a}")

# Calculate individual weights for latitude and longitude at each location
mu_lat = np.mean(latitudes)
mu_long = np.mean(longitudes)

w_lat = a / ((latitudes - mu_lat) ** 2)  # Weight for latitude at each location
w_long = a / ((longitudes - mu_long) ** 2)  # Weight for longitude at each location

print(f"w_lat shape: {w_lat.shape}, w_long shape: {w_long.shape}")

# Calculate weights for eigenvectors
w_eigenvectors = np.zeros_like(eigenvectors)
for g in range(10):
    mu_eigenvector = np.mean(eigenvectors[:, g])
    var_eigenvector = np.var(eigenvectors[:, g])  # Variance of each eigenvector across locations
    w_eigenvectors[:, g] = eigenvalues[g] / ((eigenvectors[:, g] - mu_eigenvector) ** 2)

print(f"w_eigenvectors shape: {w_eigenvectors.shape}")  # Should be (12603, 10)

# Combine the data into a DataFrame for clustering
weighted_features_df = pd.DataFrame({
    'Latitude': latitudes,
    'Longitude': longitudes,
    'Weighted_Latitude': w_lat,
    'Weighted_Longitude': w_long
})

# Add the first 10 weighted eigenvectors to the DataFrame
for g in range(10):
    weighted_features_df[f'Eigenvector_{g+1}'] = eigenvectors[:, g]
    weighted_features_df[f'Weighted_Eigenvector_{g+1}'] = w_eigenvectors[:, g]

print("Shape of weighted_features_df:", weighted_features_df.shape)

# Apply K-Means Clustering on the weighted features
n_clusters = 3000  # Choose the number of clusters (you can change this based on your needs)
kmeans = KMeans(n_clusters=n_clusters)

# Fit the K-Means model to the weighted features
weighted_features_df['Cluster'] = kmeans.fit_predict(weighted_features_df)

print(f"Cluster assignments: {weighted_features_df['Cluster'].unique()}")

# Save the clustered DataFrame to a CSV file
weighted_features_df.to_csv('clustered_weighted_data2.csv', index=False)
print("Clustered data saved to 'clustered_weighted_data.csv'")
print(weighted_features_df.shape)


# Save the eigenvectors with cluster information if needed
np.save('weighted_eigenvectors_with_clusters.npy', eigenvectors)
