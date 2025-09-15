import os
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from sklearn.decomposition import IncrementalPCA
from chirp.inference import tf_examples

# Directory containing your TFRecord files
tfrecord_dir = '/mnt/d/Uni/maldives_embeddings'

def list_files_in_folder(folder_path):
    """
    Returns a list of all files in the given folder path.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files

def read_embeddings_from_file(tfrecord_file):
    """
    Reads embeddings and corresponding class information from a single TFRecord file.
    
    Returns:
      embeddings: np.array of shape (num_examples, embedding_dim)
      class_types: list of class type strings (extracted from filename)
    """
    ds = tf.data.TFRecordDataset(tfrecord_file)
    parser = tf_examples.get_example_parser()
    ds = ds.map(parser)
    
    embeddings = []
    class_types = []
    
    # Process each example in the file
    for ex in ds.as_numpy_iterator():
        embedding = ex['embedding'].flatten()  # flatten the embedding
        filename = ex['filename'].decode("utf-8")
        # Extract class type: first character of the first part of the filename (split by '_')
        class_type = filename.split('_')[0][0] if filename else ""
        embeddings.append(embedding)
        class_types.append(class_type)
    
    # Convert list of embeddings into a numpy array
    embeddings = np.array(embeddings)
    return embeddings, class_types

# Get list of files and filter out non-TFRecord files if needed
file_list = list_files_in_folder(tfrecord_dir)
for ignore in ['reduced_feature_embeddings.csv', 'config.json']:
    if ignore in file_list:
        file_list.remove(ignore)
tfrecord_files = [os.path.join(tfrecord_dir, f) for f in file_list]

# Initialize IncrementalPCA for global dimensionality reduction to 3 components.
ipca = IncrementalPCA(n_components=3)

print("First pass: Fitting IncrementalPCA...")
# First pass: Fit the incremental PCA model on embeddings from each file.
for idx, tfrecord_file in enumerate(tfrecord_files):
    print(f"Fitting file {idx+1}/{len(tfrecord_files)}: {tfrecord_file}")
    embeddings, _ = read_embeddings_from_file(tfrecord_file)
    # partial_fit on this batch of embeddings
    ipca.partial_fit(embeddings)
    # Free up memory from this file
    del embeddings
    gc.collect()

print("Second pass: Transforming embeddings and plotting...")
# Prepare a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define a mapping of class types to specific colors
color_mapping = {
    'D': 'red',
    'R': 'blue',
    'H': 'green'
}

# Second pass: Transform and immediately plot each file's embeddings.
for idx, tfrecord_file in enumerate(tfrecord_files):
    print(f"Processing file {idx+1}/{len(tfrecord_files)}: {tfrecord_file}")
    embeddings, class_types = read_embeddings_from_file(tfrecord_file)
    # Transform the embeddings using the fitted ipca
    reduced_embeddings = ipca.transform(embeddings)
    # Map each class type to a color
    colors = [color_mapping.get(cls, 'black') for cls in class_types]
    # Plot the reduced points
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2],
               c=colors, alpha=0.7)
    # Free memory for the next iteration
    del embeddings, reduced_embeddings, class_types
    gc.collect()

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Reduction of Embeddings')

# Create a legend for the class types
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=color, label=cls) for cls, color in color_mapping.items()]
ax.legend(handles=patches)

plt.show()
input("Press [enter] to close the plot")
