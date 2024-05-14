import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Path to directory containing CSV files
directory = '/Users/anastasiskiout/Documents/Πανεπηστήμιο/Εξόρυξη Δεδομένων/Υλοποιητικό Project/harth'

def plot_subplots(label_x, label_y, dir):
    directory = dir

    # List all CSV files in the directory
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    
    group1 = csv_files[:11]
    group2 = csv_files[11:]

    fig1, axes1 = plt.subplots(4, 3, figsize=(20, 16))
    axes1 = axes1.flatten()
    
    # Iterate over each CSV file
    for i, file in enumerate(group1):
            file_name = os.path.basename(file)
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Convert 'timestamp' column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Plot 'label_x' against 'label_y' (THE FIRST GROUP)
            sns.lineplot(x=label_x, y=label_y, data=df, ax=axes1[i], color='blue', linewidth=1)
            axes1[i].set_title(f'{file_name}', fontsize=12)
            axes1[i].set_xlabel(label_x, fontsize=10)
            axes1[i].set_ylabel(label_y, fontsize=10)
            axes1[i].set_xticks([])
            axes1[i].grid(True)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.6)

    fig2, axes2 = plt.subplots(4, 3, figsize=(20, 16))
    axes2 = axes2.flatten()
    for i, file in enumerate(group2):
            file_name = os.path.basename(file)
            # Read the CSV file
            df = pd.read_csv(file)
            print(file)
            
            # Convert 'timestamp' column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Plot 'label_x' against 'label_y' (THE SECOND GROUP)
            sns.lineplot(x=label_x, y=label_y, data=df, ax=axes2[i], color='blue', linewidth=1)
            axes2[i].set_title(f'{file_name}', fontsize=12)
            axes2[i].set_xlabel(label_x, fontsize=10)
            axes2[i].set_ylabel(label_y, fontsize=10)
            axes2[i].set_xticks([])
            axes2[i].grid(True)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.6)
    plt.show()
    
plot_subplots('timestamp', 'back_z', directory)