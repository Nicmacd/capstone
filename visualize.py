import librosa, librosa.display
from pydub import AudioSegment
from visualize_mel import plot_spectrogram
from visualize_mfcc import create_mfcc
from scipy.io import wavfile
import scipy.signal as signal
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil

initialize = True

def make_directories(dir_name):
    os.mkdir(dir_name)

if initialize:
    make_directories("data")
    make_directories("data/spectogramData")
    make_directories("data/spectogramData/test")
    make_directories("data/spectogramData/validate")
    make_directories("data/spectogramData/train")
    make_directories("data/mfccData")
    make_directories("data/mfccData/test")
    make_directories("data/mfccData/validate")
    make_directories("data/mfccData/train")
    make_directories("data/mfccData/test/images")
    make_directories("data/mfccData/test/labels")
    make_directories("data/mfccData/train/images")
    make_directories("data/mfccData/train/labels")
    make_directories("data/audioData")
    make_directories("data/audioData/segmentedFiles")
    make_directories("data/audioData/test")
    make_directories("data/audioData/validate")
    make_directories("data/audioData/train")
    make_directories("data/melData")
    make_directories("data/melData/test")
    make_directories("data/melData/validate")
    make_directories("data/melData/train")

audio_folder = "./watkinsSpottedSeal/1971/"
# audio_folder = "orca"
segmented_folder = "./data/audioData/segmentedFiles/"
spectoFolderPath = "./data/spectogramData/"
train_folder = "./data/audioData/train/"
test_folder = "./data/audioData/test/"
mfcc_folder = "./data/mfccData/"
mfcc_train_imgs_folder = "./data/mfccData/train/images/"
mfcc_train_lbls_folder = "./data/mfccData/train/labels/"
mfcc_test_imgs_folder = "./data/mfccData/test/images/"
mfcc_test_lbls_folder = "./data/mfccData/test/labels/"
mel_train_folder = "./data/melData/train/"
mel_test_folder = "./data/melData/test/"

# noisyFolderPath ="./data/audioData/addedNoise/"
# cutnoisyFolderPath = "./data/audioData/cutFiles/"
# filePath = "C:/Users/conbo/PycharmProjects/capstone/capstone/audioData/watkinsSpottedSeal/1971/SpottedSeal_0006.wav"

def segmentAudio(input_file, output_folder, file_name, segment_length = 2000):
    # read the input audio wav file
    audio = AudioSegment.from_file(f'{audio_folder}/{input_file}', format="wav")

    # Calculate the number of segments
    num_segments = len(audio) // segment_length

    if num_segments == 0:
        raise ValueError("ERROR: Audio file is smaller that desired segment")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split the audio into segments and save each segment
    # starts at 0 for i
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        segment = audio[start_time:end_time]
        segment.export(os.path.join(output_folder, file_name + f"_{i + 1:04d}.wav"), format="wav")

def create_spectogram(folderPath, audioFile, destination_path, data_type):
    sample_rate, samples = wavfile.read(folderPath + audioFile)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    ###Plot as Mel Spectrogram, uses Librosa Module
    libSignal, sr = librosa.load(folderPath + audioFile)
    plot_spectrogram(destination_path, audioFile, libSignal, sr)

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [msec]')

    (file_name, file_type) = audioFile.split(".")

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(spectoFolderPath + data_type + file_name + '_spectogram.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def create_labels(data_dir, label_dir):

    for name in os.listdir(data_dir):
        name.split(".", 1)
        #txtfile_name = os.path.splitext(file_name)[0]
        txtfile = os.path.join(label_dir, name + '.txt')

        contents, excess = name.split("_", 1)

        with open(txtfile, 'w') as txt_file:
            txt_file.write(contents)




# loop through and add blank noise to original audio files
for audio_file in os.listdir(audio_folder):
    # naming convention animalType_000#.wav
    file_name, type = audio_file.split(".")
    animal_type, audio_number = audio_file.split("_")

    try:
        segmentAudio(audio_file, segmented_folder, file_name)
        print("Audio file successfully split into 2-second segments.")
    except ValueError as e:
        print(f"Error: {e}")

wav_files = [file for file in os.listdir(segmented_folder)]

# split segmented data into train and test sets
train_files, test_files = train_test_split(wav_files, test_size=0.2, random_state=42)

# Copy the segmented train and test data into their folders
for audio_file in train_files:
    src_path = os.path.join(segmented_folder, audio_file)
    dest_path = os.path.join(train_folder, audio_file)
    shutil.copy(src_path, dest_path)  # Use shutil.move if you want to move instead of copy

for audio_file in test_files:
    src_path = os.path.join(segmented_folder, audio_file)
    dest_path = os.path.join(test_folder, audio_file)
    shutil.copy(src_path, dest_path)  # Use shutil.move if you want to move instead of copy

for segmented_train in os.listdir(train_folder):
    create_spectogram(train_folder, segmented_train, mel_train_folder, "train/")
    create_mfcc(train_folder, segmented_train, mfcc_train_imgs_folder)

for segmented_test in os.listdir(test_folder):
    create_spectogram(test_folder, segmented_test, mel_test_folder, "test/")
    create_mfcc(test_folder, segmented_test, mfcc_test_imgs_folder)


# add labels here
create_labels(mfcc_train_imgs_folder, mfcc_train_lbls_folder)
create_labels(mfcc_test_imgs_folder, mfcc_test_lbls_folder)