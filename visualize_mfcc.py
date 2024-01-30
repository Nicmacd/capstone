import librosa
import librosa.display
#import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

def plot_mfcc(audio_file):
    # load audio file
    signal, sr = librosa.load(audio_file)
    #num_samples = signal.shape

    # extract mfcc's

    mfccs = librosa.feature.mfcc(y=signal, sr=sr)
    #num_mfcc = mfcc.shape

    # visualize mfcc's
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(format='%+2f')
    plt.show()

