import pandas as pd
import numpy as np

# Step 1: Load the two datasets
df1 = pd.read_csv('centroids_and_closest_nodes1.csv')  # Replace with your actual file path
df2 = pd.read_csv('centroids_and_closest_nodes2.csv')  # Replace with your actual file path

# Step 2: Find common nodes based on Node_Latitude and Node_Longitude
common_nodes = pd.merge(df1, df2, how='inner', on=['Node_Latitude', 'Node_Longitude'])

# Step 3: Save the common nodes to a CSV file
common_nodes.to_csv('common_latlong_nodes.csv', index=False)
print(f"Common nodes saved to 'common_latlong_nodes.csv' with {len(common_nodes)} entries.")

# Step 4: Find non-matching nodes from each dataset
non_matching_from_df1 = df1[~df1[['Node_Latitude', 'Node_Longitude']].apply(tuple, axis=1).isin(
    common_nodes[['Node_Latitude', 'Node_Longitude']].apply(tuple, axis=1))]

non_matching_from_df2 = df2[~df2[['Node_Latitude', 'Node_Longitude']].apply(tuple, axis=1).isin(
    common_nodes[['Node_Latitude', 'Node_Longitude']].apply(tuple, axis=1))]

# Step 5: Save the non-matching nodes to CSV files
non_matching_from_df1.to_csv('non_matching_from_df1.csv', index=False)
non_matching_from_df2.to_csv('non_matching_from_df2.csv', index=False)
print(f"Non-matching nodes saved to 'non_matching_from_df1.csv' and 'non_matching_from_df2.csv'.")

# Step 6: Calculate the shortest distance between each node in df1 and the nodes in df2
shortest_distances = pd.DataFrame(columns=['Node_Index_df1', 'Node_Latitude_df1', 'Node_Longitude_df1', 
                                           'Node_Index_df2', 'Node_Latitude_df2', 'Node_Longitude_df2', 'Distance'])

# Loop through each node in non_matching_from_df1 and calculate the distance from all nodes in non_matching_from_df2
for index1, row1 in non_matching_from_df1.iterrows():
    lat1 = row1['Node_Latitude']
    long1 = row1['Node_Longitude']
    
    # Calculate the Euclidean distance to all nodes in non_matching_from_df2
    distances = np.sqrt((non_matching_from_df2['Node_Latitude'] - lat1) ** 2 + 
                        (non_matching_from_df2['Node_Longitude'] - long1) ** 2)
    
    # Get the index of the closest node
    min_distance_idx = distances.idxmin()
    
    # Get the closest node's data from non_matching_from_df2
    closest_node = non_matching_from_df2.loc[min_distance_idx]
    
    # Store the result in the DataFrame
    shortest_distances = pd.concat([shortest_distances, pd.DataFrame([{
        'Node_Index_df1': row1['Node_Index'], 
        'Node_Latitude_df1': lat1, 
        'Node_Longitude_df1': long1, 
        'Node_Index_df2': closest_node['Node_Index'], 
        'Node_Latitude_df2': closest_node['Node_Latitude'], 
        'Node_Longitude_df2': closest_node['Node_Longitude'], 
        'Distance': distances[min_distance_idx]
    }])], ignore_index=True)

# Step 7: Save the shortest distances to a CSV file
shortest_distances.to_csv('shortest_distances_between_nodes.csv', index=False)
print("Shortest distances saved to 'shortest_distances_between_nodes.csv'.")

