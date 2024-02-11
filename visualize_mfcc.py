import librosa
import librosa.display
#import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

def plot_mfcc(source_folder, audio_file, destination_folder):
    # load audio file
    signal, sr = librosa.load(source_folder + audio_file)
    #num_samples = signal.shape

    # extract mfcc's

    mfccs = librosa.feature.mfcc(y=signal, sr=sr)
    #num_mfcc = mfcc.shape

    # visualize mfcc's
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(format='%+2f')

    # make it so that you are only getting the "image" section do not what to feed extra information into the model
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(destination_folder + audio_file + '_spectogram.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

