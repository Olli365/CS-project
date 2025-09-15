import os
import glob
import tensorflow as tf
import wave
import concurrent.futures
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input, MaxPooling2D, Flatten
import random

# ---------------------------------------------------------
# 1. Define dataset directories and assign dataset IDs
# ---------------------------------------------------------
australia_dir = '/mnt/f/mars_global_acoustic_study/australia_acoustics/'
indonesia_dir = '/mnt/f/mars_global_acoustic_study/indonesia_acoustics/raw_audio/'
maldives_dir  = '/mnt/f/mars_global_acoustic_study/maldives_acoustics/'
mexico_dir    = '/mnt/f/mars_global_acoustic_study/mexico_acoustics/'

# Dataset IDs: Australia = 1, Indonesia = 2, Maldives = 3, Mexico = 4


# ---------------------------------------------------------
# 2. Control variable for balancing:
#    Total samples per dataset (50/50 split for H and D).
# ---------------------------------------------------------
samples_per_class = 2000  # e.g., 500 H and 500 D per dataset

# ---------------------------------------------------------
# 3. Helper: Check if a WAV file is exactly 60 seconds long.
# ---------------------------------------------------------
def is_sixty_seconds_file(path):
    try:
        with wave.open(path, 'rb') as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            return frames == sr * 60
    except Exception:
        return False
    
# ---------------------------------------------------------
# 4. Label extraction functions (Python version)
# ---------------------------------------------------------
import os

def extract_label_from_folder_py(file_path):
    # For datasets with subfolders.
    # Example file path: .../Degraded_Moth32/20230207_050000.WAV
    # We split on os.sep and take the parent folder (e.g. "Degraded_Moth32")
    parts = file_path.split(os.sep)
    if len(parts) < 2:
        return -1
    folder = parts[-2]
    # Use the first character of the folder name.
    if folder.startswith("H"):
        return 0
    elif folder.startswith("D"):
        return 1
    else:
        return -1

def extract_label_from_filename_py(file_path):
    # For Indonesia files.
    # Example file path: '/mnt/f/mars_global_acoustic_study/indonesia_acoustics/raw_audio/ind_D1_20220829_120000.WAV'
    # Remove the prefix if it exists.
    prefix = '/mnt/f/mars_global_acoustic_study/indonesia_acoustics/raw_audio/'
    if file_path.startswith(prefix):
        file_path = file_path[len(prefix):]
    # Now file_path might look like "ind_D1_20220829_120000.WAV"
    parts = file_path.split('_')
    if len(parts) < 2:
        return -1
    # Take the first character of the second token, e.g., "D1" -> "D"
    if parts[1][0] == "H":
        return 0
    elif parts[1][0] == "D":
        return 1
    else:
        return -1
    
# ---------------------------------------------------------
# 5. Build a single file list for all datasets.
# ---------------------------------------------------------
import glob
import concurrent.futures
import random

def check_file(f, ds_id):
    if is_sixty_seconds_file(f):
        return f  # Return the file path if valid.
    return None

def get_files_for_dataset(root_dir, ds_id, samples_per_class=50):
    pos_files = []
    neg_files = []
    # Build the glob pattern (assumes a subfolder structure).
    if ds_id == 2:
        pattern = os.path.join(root_dir, "*.WAV")
    else:
        pattern = os.path.join(root_dir, "*", "*.WAV")
    files = glob.glob(pattern)
    print(f"Dataset {ds_id}: Total files found by glob: {len(files)}")
    # Shuffle the file list for random processing.
    random.shuffle(files)
    
    # Loop over the files until we have enough for both classes.
    for f in files:
        # Stop if we've collected enough files for both classes.
        if len(pos_files) >= samples_per_class and len(neg_files) >= samples_per_class:
            break
        
        # Check if the file is exactly 60 seconds long.
        if not is_sixty_seconds_file(f):
            continue
        
        # Extract the label depending on the dataset.
        if ds_id == 2:
            label = extract_label_from_filename_py(f)
        else:
            label = extract_label_from_folder_py(f)
        
        # Only add files with a valid label.
        if label == 1 and len(pos_files) < samples_per_class:
            pos_files.append((f, label, float(ds_id)))
        elif label == 0 and len(neg_files) < samples_per_class:
            neg_files.append((f, label, float(ds_id)))
    
    files_found = pos_files + neg_files
    random.shuffle(files_found)
    print(f"Dataset {ds_id}: Collected {len(files_found)} files (target was {2 * samples_per_class}).")
    return files_found

files_australia = get_files_for_dataset(australia_dir, ds_id=1, samples_per_class=samples_per_class)
files_indonesia = get_files_for_dataset(indonesia_dir, ds_id=2, samples_per_class=samples_per_class)
files_maldives  = get_files_for_dataset(maldives_dir,  ds_id=3, samples_per_class=samples_per_class)
files_mexico    = get_files_for_dataset(mexico_dir,    ds_id=4, samples_per_class=samples_per_class)

# Combine file lists.
all_files = files_australia + files_indonesia + files_maldives + files_mexico
print(f"Total valid files: {len(all_files)}")
# all_files is a list of tuples (file_path, ds_value)
pos_files = [item for item in all_files if item[1] == 1]
neg_files = [item for item in all_files if item[1] == 0]
x = [item for item in all_files if item[1] == -1]
print("Positive files count:", len(pos_files))
print("Negative files count:", len(neg_files))

# ---------------------------------------------------------
# 6. Create a tf.data.Dataset from the file list and extract labels.
# ---------------------------------------------------------
# Now, all_files is a list of tuples: (file_path, label, dataset_id)
file_paths = [f for f, label, ds in all_files]
labels = [label for f, label, ds in all_files]
ds_ids = [ds for f, label, ds in all_files]

# Create the dataset from a tuple of file_paths, ds_ids, and labels.
ds_all = tf.data.Dataset.from_tensor_slices((file_paths, ds_ids, labels))

def restructure(file_path, dataset_id, label):
    # Return a tuple in the format: ((file_path, dataset_id), label)
    return ((file_path, tf.cast(dataset_id, tf.float32)), label)

ds_all = ds_all.map(restructure, num_parallel_calls=tf.data.AUTOTUNE)


# ---------------------------------------------------------
# 7. Balance the dataset per location.
#    We first split by dataset ID, then select samples_per_class per label.
# ---------------------------------------------------------
def filter_by_ds(ds, ds_id):
    # The second element of inputs (index 1) is the dataset ID.
    return ds.filter(lambda inputs, label: tf.equal(inputs[1], tf.cast(ds_id, tf.float32)))

def balance_ds(ds):
    # Use samples_per_class directly for each label.
    samples_per_label = samples_per_class  
    balanced = None
    for label_val in [0, 1]:
        ds_label = ds.filter(lambda inputs, label: tf.equal(label, label_val))
        ds_label = ds_label.shuffle(10000).take(samples_per_label)
        if balanced is None:
            balanced = ds_label
        else:
            balanced = balanced.concatenate(ds_label)
    return balanced.shuffle(10000)

balanced_datasets = []
for ds_id in [1, 2, 3, 4]:
    ds_filtered = filter_by_ds(ds_all, ds_id)
    balanced_subset = balance_ds(ds_filtered)
    balanced_datasets.append(balanced_subset)  # Append the balanced subset

# Concatenate all balanced datasets into one.
balanced_data = balanced_datasets[0]
for ds in balanced_datasets[1:]:
    balanced_data = balanced_data.concatenate(ds)


# ---------------------------------------------------------
# 8. Audio loading and preprocessing (including extra features)
# ---------------------------------------------------------
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    
    target_sr = 16000
    orig_length = tf.shape(wav)[0]
    new_length = tf.cast(tf.cast(orig_length, tf.float32) *
                         (tf.cast(target_sr, tf.float32) / tf.cast(sample_rate, tf.float32)),
                         tf.int32)
    
    # Treat the audio as a 1-row “image” for resizing.
    wav_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(wav, 0), 0), -1)
    wav_resized = tf.image.resize(wav_expanded, size=[1, new_length], method='bilinear')
    wav_resized = tf.squeeze(wav_resized, axis=[0, 1, 3])
    return wav_resized

def preprocess(inputs, label):
    # inputs is a tuple: (file_path, dataset_id)
    file_path, dataset_id = inputs
    # Ensure dataset_id is a float.
    dataset_id = tf.cast(dataset_id, tf.float32)
    
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    padding_amount = tf.maximum(48000 - tf.shape(wav)[0], 0)
    wav = tf.concat([tf.zeros([padding_amount], dtype=tf.float32), wav], axis=0)
    
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    # Ensure a fixed shape: (1491, 257, 1)
    spectrogram.set_shape((1491, 257, 1))
    
    # Extra features:
    filename = tf.strings.split(file_path, os.sep)[-1]
    is_indonesia = tf.strings.regex_full_match(filename, r'^ind.*')
    
    def extract_time():
        parts = tf.strings.split(filename, '_')
        time_str = tf.strings.substr(parts[1], 0, 4)
        time_num = tf.strings.to_number(time_str, out_type=tf.float32)
        return time_num / 2359.0  # Normalize
    
    time_feature = tf.cond(is_indonesia,
                           lambda: tf.constant(0.0, dtype=tf.float32),
                           extract_time)
    dataset_feature = dataset_id / 4.0
    extra_features = tf.stack([time_feature, dataset_feature])
    
    # Return as a tuple in the correct order.
    return (spectrogram, extra_features), label

batch_size = 16
# After you have your balanced_data from all concatenations and after preprocessing
balanced_data = balanced_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
balanced_data = balanced_data.shuffle(buffer_size=1000)
balanced_data = balanced_data.batch(batch_size)
# Remove the earlier cache() call

# Now take the fixed subset, cache it, and repeat it
balanced_data = balanced_data.take(1000).cache().repeat()
balanced_data = balanced_data.prefetch(tf.data.AUTOTUNE)

# Split into training and testing datasets
train = balanced_data.take(800)
test = balanced_data.skip(800).take(200)

# ---------------------------------------------------------
# 10. Build a multi-input model (spectrogram + extra features)
# ---------------------------------------------------------
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, concatenate


# Spectrogram input
spect_input = Input(shape=(1491, 257, 1), name='spectrogram')

# Block 1: 7x7 kernel for large receptive field
x = Conv2D(64, (7, 7), padding='same')(spect_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.4)(x)

# Block 2: 3x3 kernel
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.4)(x)

# Block 3: 3x3 kernel with increased filters
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.6)(x)

# Global pooling to collapse spatial dimensions
x = GlobalAveragePooling2D()(x)

# Extra features branch (if you have 2 extra normalized features)
extra_input = Input(shape=(2,), name='extra_features')
y = Dense(16, activation='relu')(extra_input)

# Combine the convolutional features with extra features
combined = concatenate([x, y])
combined = Dense(64, activation='relu')(combined)
combined = Dropout(0.6)(combined)
output = Dense(1, activation='sigmoid')(combined)

# Use AdamW with weight decay for improved regularization
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.00005
)

model = Model(inputs=[spect_input, extra_input], outputs=output)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------------------------------------
# 11. Train the model
# ---------------------------------------------------------
hist = model.fit(train, epochs=8, validation_data=test)


import matplotlib.pyplot as plt

plt.plot(hist.history['loss'], label='Training loss')
plt.plot(hist.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(hist.history['accuracy'], label='Training loss')
plt.plot(hist.history['val_accuracy'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()