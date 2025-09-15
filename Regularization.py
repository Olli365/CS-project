import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif  # ANOVA F-test for classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
import os
from chirp.inference import tf_examples

tfrecord_dir  = '/mnt/d/Uni/'

def list_files_in_folder(folder_path):
    """
    Returns a list of all files in the given folder path.

    Parameters:
    folder_path (str): The path to the folder.

    Returns:
    List[str]: A list of file names in the folder.
    """
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    # List all files in the folder (excluding directories)
    files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    return files

"""
def read_embeddings_to_dataframe(tfrecord_files):
    
    #Read the embeddings from the TFRecord files and return them as a DataFrame.
    
    #Parameters:
    #tfrecord_files (list): List of paths to the TFRecord files.
    
    #Returns:
    #pd.DataFrame: DataFrame containing the embeddings and filenames.
 
    # Initialize empty lists to store filenames and embeddings
    filenames = []
    embeddings = []

    # Loop through the list of TFRecord files
    for tfrecord_file in tfrecord_files:
        # Create a TFRecordDataset from the current TFRecord file
        ds = tf.data.TFRecordDataset(tfrecord_file)

        # Use the example parser from tf_examples to parse the embeddings
        parser = tf_examples.get_example_parser()
        ds = ds.map(parser)

        # Iterate through the dataset and extract filenames and embeddings
        for ex in ds.as_numpy_iterator():
            filename = ex['filename'].decode("utf-8")  # Decode the byte string
            embedding = ex['embedding'].flatten()  # Flatten the embedding for easier handling
            filenames.append(filename)
            embeddings.append(embedding)

    # Convert the embeddings list to a DataFrame
    df = pd.DataFrame(embeddings)
    
    # Add the filenames as a separate column
    df['filename'] = filenames

    # Reorder columns to have 'filename' first
    df = df[['filename'] + [col for col in df.columns if col != 'filename']]

    return df
"""
def read_embeddings_to_dataframe(tfrecord_files):
    """
    Read embeddings from a list of TFRecord files using an optimized pipeline and
    return a DataFrame containing embeddings and filenames.
    
    Parameters:
    tfrecord_files (list): List of paths to the TFRecord files.
    
    Returns:
    pd.DataFrame: DataFrame containing the embeddings and filenames.
    """
    # Create a dataset of file paths
    ds_files = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    
    # Use interleave to read from multiple files concurrently
    ds = ds_files.interleave(
        lambda file: tf.data.TFRecordDataset(file),
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    # Get the parser and apply it in parallel
    parser = tf_examples.get_example_parser()
    ds = ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Prefetch to improve pipeline throughput
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    # Extract filenames and embeddings from the dataset
    filenames = []
    embeddings = []
    for ex in ds.as_numpy_iterator():
        filename = ex['filename'].decode("utf-8")  # decode the byte string
        embedding = ex['embedding'].flatten()        # flatten for easier handling
        filenames.append(filename)
        embeddings.append(embedding)
    
    # Convert the embeddings list to a DataFrame and add filenames as a column
    df = pd.DataFrame(embeddings)
    df['filename'] = filenames
    
    # Reorder columns to have 'filename' first
    df = df[['filename'] + [col for col in df.columns if col != 'filename']]
    
    return df

def process_in_batches(tfrecord_files, batch_size):
    """
    Process TFRecord files in batches and return a single DataFrame with all the embeddings.
    
    Parameters:
    tfrecord_files (list): List of paths to the TFRecord files.
    batch_size (int): Number of files to process in each batch.
    
    Returns:
    pd.DataFrame: DataFrame containing the embeddings and filenames from all batches.
    """
    # Initialize an empty list to store the dataframes from each batch
    df_list = []
    
    # Loop over the files in batches
    for i in range(0, len(tfrecord_files), batch_size):
        # Select the current batch of files
        batch_files = tfrecord_files[i:i + batch_size]
        
        print(f"Processing batch {i//batch_size + 1}")
        
        # Process the current batch and get a DataFrame
        batch_df = read_embeddings_to_dataframe(batch_files)
        
        # Append the batch dataframe to the list
        df_list.append(batch_df)
    
    # Concatenate all batch dataframes into one
        final_df = pd.concat(df_list, ignore_index=True)
        
    return final_df



file_list = list_files_in_folder(tfrecord_dir)
if 'reduced_feature_embeddings.csv' in file_list:
    file_list.remove('reduced_feature_embeddings.csv')
if 'config.json' in file_list:
    file_list.remove('config.json')
    
tfrecord_files = [os.path.join(tfrecord_dir, f) for f in file_list]

embeddings_df = process_in_batches(tfrecord_files, batch_size=5)
df = embeddings_df
def extract_metadata_from_filename(file):
    # Split the filename using '_' as the delimiter
    parts = file.split('_')
    
    # Extract the first part (before the first '_')
    first_part = parts[0] if len(parts) > 0 else ""
    
    # Return only the first letter of the first part
    return first_part[0] if first_part else ""


# Applying the function to each filename in the DataFrame
df['class_type'] = df['filename'].apply(extract_metadata_from_filename)


# If using a Jupyter Notebook in VS Code, uncomment the following line:
# %matplotlib widget

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import pandas as pd

# If running as a script, enable interactive mode:
plt.ion()

# Assuming embeddings_df is already defined and contains your data
# Extract feature columns (exclude non-numeric columns like 'filename' and 'class_type')
feature_columns = [col for col in embeddings_df.columns if col not in ['filename', 'class_type']]
X = embeddings_df[feature_columns]

# Reduce dimensions to 3 using PCA
pca = PCA(n_components=3)
pca_features = pca.fit_transform(X)

# Add the PCA components to the DataFrame
embeddings_df['pca1'] = pca_features[:, 0]
embeddings_df['pca2'] = pca_features[:, 1]
embeddings_df['pca3'] = pca_features[:, 2]

# Define a mapping of class types to specific colors
color_mapping = {
    'D': 'red',
    'R': 'blue',
    'H': 'green'
}
colors = embeddings_df['class_type'].map(color_mapping)

# Create an interactive 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(embeddings_df['pca1'], embeddings_df['pca2'], embeddings_df['pca3'], 
                     c=colors, alpha=0.7)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Reduction with Custom Class Colors')

# Create a legend for the class types
patches = [mpatches.Patch(color=color, label=cls) for cls, color in color_mapping.items()]
ax.legend(handles=patches)

plt.show()

# If running as a script, you might want to keep the window open:
input("Press [enter] to close the plot...")

