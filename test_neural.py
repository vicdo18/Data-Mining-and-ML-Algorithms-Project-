import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Directory
directory = '/Users/anastasiskiout/Documents/Πανεπηστήμιο/Εξόρυξη Δεδομένων/Υλοποιητικό Project/harth'

# Load - Combine all CSV files
def load_data(file_path):
    data_frames = [pd.read_csv(file) for file in file_path]
    data = pd.concat(data_frames, ignore_index=True)
    return data

# Normalize Data
def normalize_data(data):
    scaler = StandardScaler()
    features = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    data[features] = scaler.fit_transform(data[features])
    return data, scaler

# Create time windows
def create_time_windows(data, window_size=50, step_size=25):
    X = []
    y = []
    timestamps = []
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        window_data = data.iloc[start:end]
        X.append(window_data[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']].values)
        y.append(window_data['label'].mode()[0])  # Majority Label in the window
        timestamps.append(window_data['timestamp'].values[0])  # Take the timestamp of the first entry in the window

    return np.array(X), np.array(y), np.array(timestamps)

# Load the data
csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
data = load_data(csv_files)
data, scaler = normalize_data(data)
X, y, timestamps = create_time_windows(data)

# Split into training and test sets
X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = train_test_split(X, y, timestamps, test_size=0.2, random_state=42)


# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Load the trained model without custom metrics
model = tf.keras.models.load_model('harth_model.h5', compile=False)

# Compile the model with standard metrics
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Create a DataFrame to compare actual vs predicted actions
comparison_df = pd.DataFrame({
    'Timestamp': timestamps_test,
    'Actual': y_true_classes,
    'Predicted': y_pred_classes
})

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
print(comparison_df.head(20))  # Print the first 20 rows for comparison
