from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

initialize = False

def make_directories(dir_name):
    os.mkdir(dir_name)

audioFolderPath = "/Users/eloisecallaghan/PycharmProjects/capstone/venv/capstone/data/audioData/watkinsSpottedSeal/1971/"
spectoFolderPath ="/Users/eloisecallaghan/PycharmProjects/capstone/venv/capstone/data/spectogramData/"
# filePath = "C:/Users/conbo/PycharmProjects/capstone/capstone/audioData/watkinsSpottedSeal/1971/71012016.wav"


if initialize:
    make_directories("data")
    make_directories("data/spectogramData")
    make_directories("data/spectogramData/test")
    make_directories("data/spectogramData/validate")
    make_directories("data/spectogramData/train")
    make_directories("data/audioData/test")
    make_directories("data/audioData/validate")
    make_directories("data/audioData/train")

# loop through and create and save spectrogram for each audio file
for audioFile in os.listdir(audioFolderPath):
    sample_rate, samples = wavfile.read(audioFolderPath + audioFile)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [msec]')

    (file_name, file_type) = audioFile.split(".")

    plt.savefig(spectoFolderPath + file_name + '_spectogram.png')
    plt.show()