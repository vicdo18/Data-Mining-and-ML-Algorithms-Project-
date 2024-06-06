from sklearn.naive_bayes import GaussianNB
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


# Load the first CSV file
# df1 = pd.read_csv('harth/S006.csv')

# # Load the second CSV file
# df2 = pd.read_csv('harth/S008.csv')

# Concatenate the two dataframes
# merged_df = pd.concat([df1, df2], ignore_index=True)
# Directory
directory = '/Users/anastasiskiout/Documents/Πανεπηστήμιο/Εξόρυξη Δεδομένων/Υλοποιητικό Project/harth'


# Load - Combine all CSV files
def load_data(file_path):
    data_frames = [pd.read_csv(file) for file in file_path]
    data = pd.concat(data_frames, ignore_index=True)
    return data

csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
data = load_data(csv_files)

# Split data into features (X) and target variable (y)
X = data[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']]
y = data['label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Gaussian Naive Bayes classifier
bayes_classifier = GaussianNB()

# Fit the model to the training data
bayes_classifier.fit(X_train, y_train)

# Predict labels on the test set
y_pred = bayes_classifier.predict(X_test)
timestamps_test = data.loc[X_test.index, 'timestamp']

comparison_df = pd.DataFrame({
    'TimeStamp': timestamps_test,
    'Actual': y_test,
    'Predicted': y_pred
}, index=None)

activity_map = {
    1: 'walking',	
    2: 'running',	
    3: 'shuffling',
    4: 'stairs (ascending)',	
    5: 'stairs (descending)',	
    6: 'standing',	
    7: 'sitting',	
    8: 'lying',	
    13: 'cycling (sit)',	
    14: 'cycling (stand)',	
    130: 'cycling (sit, inactive)',
    140: 'cycling (stand, inactive)'
}

# Replace activity codes with activity names
comparison_df['Actual'] = comparison_df['Actual'].map(activity_map)
comparison_df['Predicted'] = comparison_df['Predicted'].map(activity_map)

# Print the comparison DataFrame
print(comparison_df.head(20).to_string(index=False))
