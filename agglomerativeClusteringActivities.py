import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import os

# Directory
directory = '/Users/anastasiskiout/Documents/Πανεπηστήμιο/Εξόρυξη Δεδομένων/Υλοποιητικό Project/harth'

# Load - Combine all CSV files and add participant ID
def load_data(file_path):
    data_frames = []
    for file in file_path:
        df = pd.read_csv(file)
        participant_id = os.path.splitext(os.path.basename(file))[0]  # Use filename as participant ID
        df['participant_id'] = participant_id
        data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)
    return data

# Load data
csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
data = load_data(csv_files)

# Calculate the duration of each activity for each participant
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['duration'] = data.groupby('participant_id')['timestamp'].diff().dt.total_seconds().fillna(0)

# Filter data for relevant activities
relevant_labels = [1, 2, 7, 8, 13, 14, 130, 140]  # including walking and running for simplicity
activity_data = data[data['label'].isin(relevant_labels)]

# Calculate total duration spent on each relevant activity for each participant
activity_duration = activity_data.groupby(['participant_id', 'label'])['duration'].sum().unstack(fill_value=0)

# Rename columns for clarity
activity_duration = activity_duration.rename(columns={
    1: 'walking',
    2: 'running',
    7: 'sitting',
    8: 'lying',
    13: 'cycling_sit',
    14: 'cycling_stand',
    130: 'cycling_sit_inactive',
    140: 'cycling_stand_inactive'
})

# Aggregate durations for cycling
activity_duration['cycling'] = activity_duration[['cycling_sit', 'cycling_stand', 'cycling_sit_inactive', 'cycling_stand_inactive']].sum(axis=1)

# Select relevant columns for clustering
features = activity_duration[['cycling', 'running', 'sitting', 'lying', 'walking']]

# Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Perform Agglomerative Clustering to divide into 3 clusters
n_clusters = 3  # We want three clusters
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
clusters = agg_clustering.fit_predict(features_normalized)

# Add the cluster assignments to the activity duration data
activity_duration['Cluster'] = clusters

# Print cluster assignments for the participants
clustered_data = activity_duration.reset_index()[['participant_id', 'Cluster']]
print(clustered_data)

# Determine which cluster represents cyclists, runners, and the rest
mean_durations = activity_duration.groupby('Cluster')[['cycling', 'running']].mean()
print(mean_durations)

# Identify the clusters
cluster_cyclists = mean_durations['cycling'].idxmax()
cluster_runners = mean_durations['running'].idxmax()
cluster_rest = [i for i in range(n_clusters) if i not in [cluster_cyclists, cluster_runners]][0]

# Label the clusters
clusters_labels = {
    cluster_cyclists: 'Cyclists',
    cluster_runners: 'Runners',
    cluster_rest: 'Rest'
}

# Add labels to the clustered data
clustered_data['Cluster_Label'] = clustered_data['Cluster'].map(clusters_labels)

# Print the final labeled clusters
print(clustered_data)
