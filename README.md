The autoencoder is specifically designed to reduce the dimensionality of the data from 2,142,510 features to 110. Initially, the data was in the shape (535, 170, 12,603), which was then flattened to (535, 2,142,510). Due to space constraints, data preprocessing was performed by dividing the locations into 3,000 clusters, where PCA and K-means clustering were applied. The weights for each location were calculated, and the first 10 eigenvalues were retained. The data extracted from this process had a shape of (595, 170, 3,000). This data was further flattened to (535, 510,000) and passed into the autoencoder. Optuna was employed for hyperparameter optimization, and multiple architectures were tested. The model successfully extracted important features and effectively reduced the data's dimensionality.
