import librosa, librosa.display
from visualize_mel import plot_spectrogram
#from peak_extraction import extract_peaks
from scipy.io import wavfile


#extract peaks
sr, signal = wavfile.read("C:/Users/conbo/PycharmProjects/capstoneDirectory/capstone/watkinsSpottedSeal/1971/71012011.wav")




#display spectrograms
signal, sr = librosa.load("C:/Users/conbo/PycharmProjects/capstoneDirectory/capstone/watkinsSpottedSeal/1971/71012011.wav")
# this is the number of samples in a window per fft
n_fft = 2048
# The amount of samples we are shifting after each fft
hop_length = 512
plot_spectrogram(signal, sr, n_fft, hop_length)