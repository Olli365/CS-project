import os
from matplotlib import pyplot as plt
import tensorflow as tf 
import librosa
import glob
import wave

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization



# If you have a complete model saved:
model = tf.saved_model.load('/home/os/aqoustics/Aqoustics-Surfperch/kaggle/')



def load_wav_16k_mono(filename):
    # Read and decode the WAV file, automatically retrieving the sample rate
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)  # shape: [orig_length]
    
    target_sr = 16000  # desired sample rate
    
    # Compute new length using the sample rate from the file
    orig_length = tf.shape(wav)[0]
    new_length = tf.cast(
        tf.cast(orig_length, tf.float32) *
        (tf.cast(target_sr, tf.float32) / tf.cast(sample_rate, tf.float32)),
        tf.int32
    )
    
    # Expand dims to create a 4D tensor [batch, height, width, channels]
    # Here we treat the audio as a 1-row image where width = number of samples.
    wav_expanded = tf.expand_dims(wav, 0)         # shape: [1, orig_length]
    wav_expanded = tf.expand_dims(wav_expanded, 0)  # shape: [1, 1, orig_length]
    wav_expanded = tf.expand_dims(wav_expanded, -1) # shape: [1, 1, orig_length, 1]
    
    # Resize the "width" from orig_length to new_length; height remains 1.
    wav_resized = tf.image.resize(wav_expanded, size=[1, new_length], method='bilinear')
    
    # Remove extra dimensions: batch, height, and channel.
    wav_resized = tf.squeeze(wav_resized, axis=[0, 1, 3])  # final shape: [new_length]
    
    return wav_resized




audio = "/mnt/f/mars_global_acoustic_study/maldives_acoustics/"



def is_sixty_seconds_file(path):
    try:
        with wave.open(path, 'rb') as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            # Check if duration is exactly 60 seconds.
            return frames == sr * 60
    except Exception as e:
        return False

# Get all healthy files using glob.
healthy_files = glob.glob(audio + "H*/*.WAV")
# Filter healthy files for only those that are exactly 60 seconds.
sixty_second_healthy_files = [f for f in healthy_files if is_sixty_seconds_file(f)]
# Create the tf.data.Dataset from the filtered list.
pos = tf.data.Dataset.from_tensor_slices(sixty_second_healthy_files)

# Do the same for degraded files.
degraded_files = glob.glob(audio + "D*/*.WAV")
sixty_second_degraded_files = [f for f in degraded_files if is_sixty_seconds_file(f)]
neg = tf.data.Dataset.from_tensor_slices(sixty_second_degraded_files)
# Then attach labels
positives = pos.map(lambda file: (file, tf.constant(1, dtype=tf.int32)))
negatives = neg.map(lambda file: (file, tf.constant(0, dtype=tf.int32)))
data = positives.concatenate(negatives)


max_files = 1000

lengths = []
processed_files = 0

# Iterate through each subfolder in the audio directory
for folder in os.listdir(audio):
    # Only process folders that start with 'H' (Healthy)
    if folder.startswith('H'):
        folder_path = os.path.join(audio, folder)
        # Process only .wav files (case-insensitive)
        for file in os.listdir(folder_path):
            if file.lower().endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                try:
                    tensor_wave = load_wav_16k_mono(file_path)
                    lengths.append(len(tensor_wave))
                    processed_files += 1
                except Exception as e:
                    print(f"Skipping file {file_path} due to error: {e}")
                # Break out of loop if we've processed the maximum number of files
                if processed_files >= max_files:
                    break
        if processed_files >= max_files:
            break

print(f"Processed {processed_files} healthy audio files.")

tf.math.reduce_mean(lengths)
tf.math.reduce_min(lengths)
tf.math.reduce_max(lengths)

def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    padding_amount = 48000 - tf.shape(wav)[0]
    # Ensure padding_amount is non-negative
    padding_amount = tf.maximum(padding_amount, 0)
    zero_padding = tf.zeros([padding_amount], dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], axis=0)
    # ...
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    # Force static shape if you know what it should be:
    spectrogram.set_shape((1491, 257, 1))
    return spectrogram, label


filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()

spectrogram, label = preprocess(filepath, label)



data = data.map(preprocess)
#data = data.apply(tf.data.experimental.ignore_errors())
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(10)
data = data.prefetch(8)


train = data.take(3269)
test = data.skip(3269).take(1401)

samples, labels = train.as_numpy_iterator().next()


"""
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257, 1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
"""

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257, 1)),
    BatchNormalization(),
    Conv2D(16, (3,3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

model.summary()

hist = model.fit(train, epochs=4, validation_data=test)