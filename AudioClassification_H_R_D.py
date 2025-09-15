import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

# Import the example parser from your chirp library.
from chirp.inference import tf_examples

# -------------------------------
# 1. List and filter TFRecord files
# -------------------------------
def list_files_in_folder(folder_path):
    """Returns a list of files (not directories) in the given folder."""
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

tfrecord_dir = '/mnt/d/Uni/maldives_embeddings/'  # Adjust this path if needed.
file_list = list_files_in_folder(tfrecord_dir)

# Exclude files you don't want.
for exclude in ['reduced_feature_embeddings.csv', 'config.json']:
    if exclude in file_list:
        file_list.remove(exclude)

# Since your embedding files have no extension but begin with "embeddings-"
tfrecord_files = [os.path.join(tfrecord_dir, f) for f in file_list if f.startswith("embeddings-")]
print("All files in the directory:", file_list)
print(f"Found {len(tfrecord_files)} TFRecord files after filtering.")

# -------------------------------
# 2. Split the file list into training and validation sets
# -------------------------------
num_files = len(tfrecord_files)
split_index = int(num_files * 0.8)
train_files = tfrecord_files[:split_index]
val_files = tfrecord_files[split_index:]
print(f"Using {len(train_files)} files for training and {len(val_files)} files for validation.")

def build_dataset(file_list):
    """Builds a tf.data.Dataset from a list of TFRecord files using interleave."""
    files_ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = files_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Use the example parser from your chirp library.
    parser = tf_examples.get_example_parser()
    ds = ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

train_ds = build_dataset(train_files)
val_ds = build_dataset(val_files)

# -------------------------------
# 3. Define label extraction and filtering (ignoring examples not starting with D, R, or H)
# -------------------------------
# Map:
#   'D' (degraded)   to 0.0,
#   'R' (restored)   to 0.5,
#   'H' (healthy)    to 1.0.
label_mapping = {'D': 0.0, 'R': 0.5, 'H': 1.0}

def get_label(filename):
    """
    Extracts the label from the filename.
    Converts the tensor to a numpy bytes object, decodes it, and returns:
      0.0 if the first letter is 'D'
      0.5 if 'R'
      1.0 if 'H'
      -1 otherwise.
    """
    s = filename.numpy().decode('utf-8')
    parts = s.split('_')
    first_letter = parts[0][0] if parts and parts[0] else ""
    return label_mapping.get(first_letter, -1)

def parse_and_extract(example):
    # Flatten the embedding to a 1-D tensor.
    embedding = tf.reshape(example['embedding'], [-1])
    # Set the expected shape to [163840] based on observed output.
    embedding.set_shape([163840])
    # Use tf.py_function to compute the label from the filename.
    label = tf.py_function(func=get_label, inp=[example['filename']], Tout=tf.float32)
    label.set_shape([])
    return embedding, label

def prepare_dataset(ds):
    ds = ds.map(parse_and_extract, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Filter out examples with label -1 (i.e., those not starting with D, R, or H)
    ds = ds.filter(lambda emb, lab: tf.not_equal(lab, -1))
    return ds

train_ds = prepare_dataset(train_ds)
val_ds = prepare_dataset(val_ds)

# -------------------------------
# 4. Prepare the dataset for training
# -------------------------------
batch_size = 32
train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# 5. Build a Keras regression model for ordinal prediction
# -------------------------------
# The model takes a 163840-dimensional input and outputs a single value in [0,1].
model = Sequential([
    tf.keras.layers.Input(shape=(163840,)),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid ensures output is between 0 and 1.
])

# Compile with a regression loss (mean squared error) and monitor MAE.
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
model.summary()

# -------------------------------
# 6. Train the model and print validation metrics
# -------------------------------
# We simply call model.fit() with the training and validation datasets.
history = model.fit(train_ds, epochs=10, validation_data=val_ds)
