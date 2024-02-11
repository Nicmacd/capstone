import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa, librosa.display
import numpy as np
from scipy.signal import find_peaks

def plot_spectrogram(destination_folder, audio_file, signal, sr, n_fft = 2048, hl = 512, peakIndex = None):
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hl, n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(8, 7))
    librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hl)
    plt.colorbar(label='dB')
    if(peakIndex):
        peak = peakIndex/sr
        plt.axvline(x=peak, color='r', linestyle='--', label=f'Line at {peak} seconds')
    # plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    # plt.xlabel('Time', fontdict=dict(size=15))
    # plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(destination_folder + audio_file + '_mel.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()