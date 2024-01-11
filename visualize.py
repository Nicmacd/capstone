from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile


sample_rate, samples = wavfile.read("C:/Users/conbo/PycharmProjects/capstone/capstone/watkinsSpottedSeal/1971/71012016.wav")
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()