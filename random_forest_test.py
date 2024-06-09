import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Directory
directory = '/Users/anastasiskiout/Documents/Πανεπηστήμιο/Εξόρυξη Δεδομένων/Υλοποιητικό Project/harth'

# Load - Combine all CSV files
def load_data(file_path):
    data_frames = [pd.read_csv(file) for file in file_path]
    data = pd.concat(data_frames, ignore_index=True)
    return data

# Normalize Data
def normalize_data(data, scaler=None):
    features = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    if scaler is None:
        scaler = StandardScaler()
        data[features] = scaler.fit_transform(data[features])
    else:
        data[features] = scaler.transform(data[features])
    return data, scaler

# Load data
csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
data = load_data(csv_files)
data, scaler = normalize_data(data)

# Split data into features (X) and target variable (y)
X = data[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']]
y = data['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the model to a file
model_filename = 'random_forest_model.joblib'

# Load the model
loaded_rf_classifier = joblib.load(model_filename)

# Making predictions
y_pred = loaded_rf_classifier.predict(X_test)
timestamps_test = data.loc[X_test.index, 'timestamp']

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Create a DataFrame to compare actual vs. predicted actions
comparison_df = pd.DataFrame({
    'Timestamp': timestamps_test,
    'Actual': y_test,
    'Predicted': y_pred
}, index=None)

# Map activity codes to activity names
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
