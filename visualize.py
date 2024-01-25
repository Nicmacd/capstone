import librosa, librosa.display
from visualize_mel import plot_spectrogram
from visualize_mfcc import plot_mfcc
from scipy.io import wavfile
import scipy.signal as signal
import os
import numpy as np
import matplotlib.pyplot as plt

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
    make_directories("data/audioData/cutFiles")
    make_directories("data/audioData/test")
    make_directories("data/audioData/validate")
    make_directories("data/audioData/train")

audioFolderPath = "./watkinsSpottedSeal/1971/"
spectoFolderPath ="./data/spectogramData/"
noisyFolderPath ="./data/audioData/addedNoise/"
cutnoisyFolderPath = "./data/audioData/cutFiles/"
# filePath = "C:/Users/conbo/PycharmProjects/capstone/capstone/audioData/watkinsSpottedSeal/1971/71012016.wav"

# add noise to the .wav audio files
def addBlankNoise(inputFile, outputFile, noise_level = 0, windowSize = 5.0):
    # Read the input wav file
    sample_rate, data = wavfile.read(inputFile)
    # Generate random noise
    # length = len(data)
    length = int((windowSize/2)*sample_rate) # 5 seconds I am not sure I am confused about the units
    ###I think that length is just the number of samples, so time elapsed would be length/sample rate, 2.5 seconds will be 2.5 * SR
    ###Are we sure that we want noise vs just adding silence?
    noise = np.random.normal(0, noise_level, length)
    # Add noise to the original data
    noisyData = np.concatenate((noise, data, noise))
    # Ensure the data is within the valid range for a 16-bit PCM wav file
    noisyData = np.clip(noisyData, -32767, 32767)

    #working on trying to find the peak noise value
    peakAmplitude = noisyData.max()
    peakIndex = np.where(noisyData == peakAmplitude)[0]

    # Get the sample index of the maximum peak
    #peakSample = noisyData[peakIndex]

    # Set the window size around the peak (in seconds)
    halfWindow = windowSize / 2

    # need to figure out timing unit here
    # not sure about the sampling rate need to figure out what that is actually
    # Idea is:
    # start index ... 2.5 sec ... max audio recording ... 2.5 sec ... end index
    # getting negative number from the starting index maybe not enough time is being added

    ###Sample rate is like how much time goes by in between each value in the audio file, so it looks good below
    ###Difference in indexes would be seconds*sample rate, eg if the SR is two samples a second 2.5 seconds of audio would be 5 samples

    difference = int(sample_rate * halfWindow)
    startIndex = int(peakIndex - difference)
    endIndex = int(peakIndex + difference)

    # Cut the audio around the peak
    cut_audio = noisyData[startIndex:endIndex]

    # Save the cut audio to a new file
    output_file = 'cut_audio_around_peak.wav'
    wavfile.write(cutnoisyFolderPath + outputFile + output_file, sample_rate, cut_audio.astype(np.int16))

    # Save the noisy data to a new wav file
    wavfile.write(noisyFolderPath + outputFile, sample_rate, noisyData.astype(np.int16))


# loop through and add blank noise to original audio files
for audioFile in os.listdir(audioFolderPath):
    addBlankNoise(audioFolderPath + audioFile, "noiseAdded" + audioFile, 0)

# loop through and create and save spectrogram for each audio file
for noisyAudioFile in os.listdir(noisyFolderPath):
    sample_rate, samples = wavfile.read(noisyFolderPath + noisyAudioFile)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    ###Plot as Mel Spectrogram, uses Librosa Module
    libSignal, sr = librosa.load(noisyFolderPath + noisyAudioFile)
    plot_spectrogram(libSignal, sr)

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [msec]')

    (file_name, file_type) = noisyAudioFile.split(".")

    plt.savefig(spectoFolderPath + file_name + '_spectogram.png')
    plt.show()

    # plot mfcc's
    plot_mfcc(noisyFolderPath + noisyAudioFile)


for cutnoisyAudioFile in os.listdir(cutnoisyFolderPath):
    sample_rate, samples = wavfile.read(cutnoisyFolderPath + cutnoisyAudioFile)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [msec]')

    (file_name, other_name, file_type) = cutnoisyAudioFile.split(".")

    plt.savefig(spectoFolderPath + file_name + '_cut_spectogram.png')
    plt.show()
