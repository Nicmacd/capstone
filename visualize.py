from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import numpy as np

initialize = False

def make_directories(dir_name):
    os.mkdir(dir_name)

if initialize:
    make_directories("data")
    make_directories("data/spectogramData")
    make_directories("data/spectogramData/test")
    make_directories("data/spectogramData/validate")
    make_directories("data/spectogramData/train")
    make_directories("data/audioData")
    make_directories("data/audioData/addedNoise")
    make_directories("data/audioData/test")
    make_directories("data/audioData/validate")
    make_directories("data/audioData/train")

audioFolderPath = "/Users/eloisecallaghan/PycharmProjects/capstone/capstone/watkinsSpottedSeal/1971/"
spectoFolderPath ="/Users/eloisecallaghan/PycharmProjects/capstone/capstone/data/spectogramData/"
noisyFolderPath ="/Users/eloisecallaghan/PycharmProjects/capstone/capstone/data/audioData/addedNoise/"

# filePath = "C:/Users/conbo/PycharmProjects/capstone/capstone/audioData/watkinsSpottedSeal/1971/71012016.wav"

# add noise to the .wav audio files
def addBlankNoise(inputFile, outputFile, noise_level = 0, windowSize = 5.0):
    # Read the input wav file
    sample_rate, data = wavfile.read(inputFile)
    # Generate random noise
    noise = np.random.normal(0, noise_level, len(data))
    # Add noise to the original data
    noisyData = np.concatenate((noise, data, noise))
    # Ensure the data is within the valid range for a 16-bit PCM wav file
    noisyData = np.clip(noisyData, -32767, 32767)

    # working on trying to find the peak noise value
    # peakAmplitude = np.max(np.abs(noisyData))
    # peakIndex = np.where(noisyData == peakAmplitude)[0]
    #
    # # Get the sample index of the maximum peak
    # peakSample = noisyData[peakIndex]
    #
    # # Set the window size around the peak (in seconds)
    # halfWindow = windowSize / 2
    # start_index = max(0, peakSample - int(sample_rate * halfWindow))
    # end_index = min(len(data), peakSample + int(sample_rate * halfWindow))
    #
    # # Cut the audio around the peak
    # cut_audio = data[start_index:end_index]
    #
    # # Save the cut audio to a new file
    # output_file = 'cut_audio_around_peak.wav'
    # wavfile.write(output_file, sample_rate, cut_audio.astype(np.int16))

    # Save the noisy data to a new wav file
    wavfile.write(noisyFolderPath + outputFile, sample_rate, noisyData.astype(np.int16))


# loop through and add blank noise to original audio files
for audioFile in os.listdir(audioFolderPath):
    addBlankNoise(audioFolderPath + audioFile, "noiseAdded" + audioFile, 0)

# loop through and create and save spectrogram for each audio file
for noisyAudioFile in os.listdir(noisyFolderPath):
    sample_rate, samples = wavfile.read(noisyFolderPath + noisyAudioFile)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [msec]')

    (file_name, file_type) = noisyAudioFile.split(".")

    plt.savefig(spectoFolderPath + file_name + '_spectogram.png')
    plt.show()