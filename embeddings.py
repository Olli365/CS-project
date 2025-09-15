from etils import epath
from ml_collections import config_dict
import os
import tensorflow as tf
import tqdm
from chirp.inference import colab_utils
colab_utils.initialize(use_tf_gpu=True, disable_warnings=True)

from chirp import audio_utils
from chirp.inference import embed_lib
from chirp.inference import tf_examples
import librosa
import soundfile as sf
from audioread import NoBackendError
from chunk import Chunk
import aifc
import wave
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

australia_dir = '/mnt/f/mars_global_acoustic_study/australia_acoustics'
indonesia_dir = '/mnt/f/mars_global_acoustic_study/indonesia_acoustics/raw_audio'
maldives_dir = '/mnt/f/mars_global_acoustic_study/maldives_acoustics/'
mexico_dir = '/mnt/f/mars_global_acoustic_study/mexico_acoustics/'

test_dir = '/mnt/f/mars_global_acoustic_study/test'

# Recursive function to gather all .wav files from subdirectories with progress bar
def gather_audio_files(root_dir):
    audio_files = []
    total_files = sum([len(files) for _, _, files in os.walk(root_dir)])  # Estimate total files
    with tqdm.tqdm(total=total_files, desc="Gathering audio files") as pbar:
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.WAV'):
                    full_path = os.path.join(dirpath, file)
                    audio_files.append(full_path)
                    pbar.update(1)
    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files:
        print(f)
    return audio_files

config = config_dict.ConfigDict()
config.embed_fn_config = config_dict.ConfigDict()
config.embed_fn_config.model_config = config_dict.ConfigDict()

perch_model_path = '/home/os/aqoustics/Aqoustics-Surfperch/kaggle'


config.source_file_patterns = gather_audio_files(maldives_dir)
config.output_dir = '/mnt/d/Uni/'


model_choice = 'perch'
if model_choice == 'perch':
    config.embed_fn_config.model_key = 'taxonomy_model_tf'
    config.embed_fn_config.model_config.window_size_s = 5.0
    config.embed_fn_config.model_config.hop_size_s = 5.0
    config.embed_fn_config.model_config.sample_rate = 32000 
    config.embed_fn_config.model_config.model_path = perch_model_path
    
    
# Only write embeddings to reduce size.
config.embed_fn_config.write_embeddings = True
config.embed_fn_config.write_logits = False
config.embed_fn_config.write_separated_audio = False
config.embed_fn_config.write_raw_audio = False


# Number of parent directories to include in the filename.
config.embed_fn_config.file_id_depth = 1
config.tf_record_shards = 100
# Set up the embedding function, including loading models.
embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
print(f'\n\nLoading model(s) from: ', perch_model_path)
embed_fn.setup()


output_dir = epath.Path(config.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
embed_lib.maybe_write_config(config, output_dir)


# Create SourceInfos.
# 3m 56s for 97232 files!
source_infos = embed_lib.create_source_infos(
    config.source_file_patterns,
    num_shards_per_file=config.get('num_shards_per_file', -1),
    shard_len_s=config.get('shard_len_s', -1))
print(f'Found {len(source_infos)} source infos.')


def safe_load_audio(filepath: str, sample_rate: int):
    try:
        # Load the audio file as is
        audio, sr = librosa.load(filepath, sr=sample_rate)
        return audio
    except (librosa.util.exceptions.ParameterError, sf.LibsndfileError, EOFError, aifc.Error) as e:
        print(f"Skipping file {filepath}: {str(e)}")
        return None
    except Exception as e:
        # Catch any other exceptions that might occur
        print(f"Unexpected error for file {filepath}: {str(e)}")
        return None


sample_audio_file = config.source_file_patterns[0]  # Take the first audio file from the gathered list
sample_audio = safe_load_audio(sample_audio_file, config.embed_fn_config.model_config.sample_rate)


if sample_audio is not None:
    time_axis = np.linspace(0, len(sample_audio) / config.embed_fn_config.model_config.sample_rate, num=len(sample_audio))
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, sample_audio)
    plt.title("Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.show()
    
    
# Set up the audio loader function for the main loop
audio_loader = lambda fp, offset: safe_load_audio(fp, config.embed_fn_config.model_config.sample_rate)

# Initialize counters for successful, failed, and skipped files
succ, fail, skipped = 0, 0, 0

# Initialize audio_iterator to None
audio_iterator = None

try:
    # Use source_infos for audio iterator
    audio_iterator = audio_utils.multi_load_audio_window(
        filepaths=[s.filepath for s in source_infos],
        offsets=[s.shard_num * s.shard_len_s for s in source_infos],
        audio_loader=audio_loader,
    )

    # Adding progress bar for processing source_infos
    with tqdm.tqdm(total=len(source_infos), desc="Processing embeddings") as pbar:
        with tf_examples.EmbeddingsTFRecordMultiWriter(
            output_dir=output_dir, num_files=config.get('tf_record_shards', 1)) as file_writer:
            
            for source_info, audio in zip(source_infos, audio_iterator):
                file_id = source_info.file_id(config.embed_fn_config.file_id_depth)
                offset_s = source_info.shard_num * source_info.shard_len_s
                if audio is None:
                    skipped += 1
                    pbar.update(1)
                    continue
                example = embed_fn.audio_to_example(file_id, offset_s, audio)
                if example is None:
                    fail += 1
                    pbar.update(1)
                    continue
                file_writer.write(example.SerializeToString())
                succ += 1
                pbar.update(1)
            file_writer.flush()
finally:
    # Only delete audio_iterator if it was defined
    if audio_iterator is not None:
        del (audio_iterator)

# Print summary of processing results
print(f'\n\nSuccessfully processed {succ} source_infos.')
print(f'Failed to process {fail} source_infos.')
print(f'Skipped {skipped} files due to errors.')