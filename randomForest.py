import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Directory
directory = '/Users/anastasiskiout/Documents/Πανεπηστήμιο/Εξόρυξη Δεδομένων/Υλοποιητικό Project/harth'

# Load - Combine all CSV files
print("Loading and combining CSV files...")
def load_data(file_path):
    data_frames = [pd.read_csv(file) for file in file_path]
    data = pd.concat(data_frames, ignore_index=True)
    return data

# Normalize Data
print("Normalizing data...")
def normalize_data(data):
    scaler = StandardScaler()
    features = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    data[features] = scaler.fit_transform(data[features])
    return data, scaler

# Load data
csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
data = load_data(csv_files)
# data = pd.read_csv("harth/S006.csv")
data, scaler = normalize_data(data)

# Split data into features (X) and target variable (y)
print("Splitting data into features and target variable...")
X = data[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']]
y = data['label']

# Split into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Random Forest classifier
print("Instantiating the Random Forest classifier...")
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Fit the model to the training data
print("Fitting the model to the training data...")
rf_classifier.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
train_accuracy = rf_classifier.score(X_train, y_train)
test_accuracy = rf_classifier.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions
print("Making predictions...")
y_pred = rf_classifier.predict(X_test)