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

# Filter data for sitting (label 7) and lying (label 8)
sitting_lying_data = data[data['label'].isin([7, 8])]

# Calculate total duration spent sitting and lying for each participant
activity_duration = sitting_lying_data.groupby(['participant_id', 'label'])['duration'].sum().unstack(fill_value=0)

# Rename columns for clarity
activity_duration.columns = ['lying', 'sitting']

# Normalize the sitting and lying duration data
scaler = StandardScaler()
activity_duration_normalized = scaler.fit_transform(activity_duration)

# Perform Agglomerative Clustering to divide into 2 clusters
n_clusters = 2  # We want two clusters
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
clusters = agg_clustering.fit_predict(activity_duration_normalized)

# Add the cluster assignments to the activity duration data
activity_duration['Cluster'] = clusters

# Print cluster assignments for the participants
clustered_data = activity_duration.reset_index()[['participant_id', 'Cluster']]
print(clustered_data)

# Determine which cluster represents participants that sit/lie a lot
mean_durations = activity_duration.groupby('Cluster')[['sitting', 'lying']].mean()
print(mean_durations)

# Label the clusters
if mean_durations.loc[0, 'sitting'] + mean_durations.loc[0, 'lying'] > mean_durations.loc[1, 'sitting'] + mean_durations.loc[1, 'lying']:
    high_activity_cluster = 1
    low_activity_cluster = 0
else:
    high_activity_cluster = 0
    low_activity_cluster = 1

# Add labels to the clustered data
clustered_data['Cluster_Label'] = clustered_data['Cluster'].map({
    low_activity_cluster: 'Low Sitting/Lying',
    high_activity_cluster: 'High Sitting/Lying'
})

# Print the final labeled clusters
print(clustered_data)
