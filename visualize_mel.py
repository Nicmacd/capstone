import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa, librosa.display
import numpy as np
from scipy.signal import find_peaks

def plot_spectrogram(signal, sr, n_fft, hl):
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hl, n_fft=n_fft)
    #mel_signal = mel_signal[86:108]
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    power_sums = []
    power_threshhold = -5
    for time_index in power_to_db:
        power_sum = 0
        for frequency in range(0, len(time_index)):
            if(time_index[frequency] > power_threshhold):
                power_sum = 1
        power_sums.append(power_sum)
    print(power_sums)
    plt.plot(power_sums)
    plt.title('Data Plot')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.show()


    peaks, _ = find_peaks(power_sums, prominence=5, height=0, distance = 10)
    peaktimes = []
    for peak in peaks:
        peaktimes.append(((peak * hl) / sr))

    plt.figure(figsize=(8, 7))
    librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hl)
    plt.colorbar(label='dB')

    for peak in peaktimes:
        plt.axvline(x=peak, color='r', linestyle='--', label=f'Line at {peak} seconds')

    plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.show()