import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
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
def create_time_windows(data, window_size=50, step_size = 25):
    X = []
    y = []
    timestamps = []
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        window_data = data.iloc[start:end]
        X.append(window_data[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']].values)
        y.append(window_data['label'].mode()[0]) # Majority Label in the window
        timestamps.append(window_data['timestamp'].values[0])  # Take the timestamp of the first entry in the window

    return np.array(X), np.array(y), np.array(timestamps)

csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

data = load_data(csv_files)
data, scaler = normalize_data(data)
X, y, timestamps = create_time_windows(data)

# Print data shapes
print(f'X shape: {X.shape}, y shape: {y.shape}')


y = tf.keras.utils.to_categorical(y)
# Split into training and test sets
X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = train_test_split(X, y, timestamps, test_size=0.2, random_state=42)

# Print train and test shapes
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')


# Define model parameters
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = y_train.shape[1]
learning_rate = 0.001
epochs = 5
batch_size = 128

# Define the model
def cnn_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Build the model
model = cnn_lstm_model()
# Define custom metrics
def recall_m(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))



# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy', recall_m, precision_m, f1_m])

# Summary of the model
model.summary()


# Define a custom callback to print metrics after training
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        test_loss, test_accuracy, test_recall, test_precision, test_f1 = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test Recall: {test_recall * 100:.2f}%")
        print(f"Test Precision: {test_precision * 100:.2f}%")
        print(f"Test F1 Score: {test_f1 * 100:.2f}%")
        
# Train the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[MetricsCallback()])

model.save('harth_model.h5')
