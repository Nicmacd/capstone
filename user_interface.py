import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa, librosa.display
import numpy as np
from scipy.signal import find_peaks
import os
import torch
from torch import tensor
from torchvision import transforms
import cv2
import pygame
import torch.nn as nn
import torch.nn.functional as F
import threading

data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

label_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

file_path = None

class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)

        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(32)

        # Fully Connected Layer
        self.fc1 = nn.Linear(32 * 28 * 28, 64)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)

        # Output Layer
        # TODO change to 4
        self.fc2 = nn.Linear(64, 12)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.batchnorm2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.batchnorm3(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def plot_spectrogram(signal, sr, n_fft = 2048, hl = 512, peakIndex = None):
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hl, n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(8, 7))
    librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hl)
    plt.colorbar(label='dB')
    if(peakIndex):
        peak = peakIndex/sr
        plt.axvline(x=peak, color='r', linestyle='--', label=f'Line at {peak} seconds')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("./mel.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def create_mfcc(signal, sr):
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
    plt.savefig("./mfcc.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

def start_audio_thread():
    threading.Thread(target=play_audio).start()

def play_audio():
    if(file_path == None):
        error_label.config(text="ERROR: Must Select an Audio File Before Playing")
        return
    
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

    # Load the audio file into the mixer
    pygame.mixer.music.load(file_path)
    signal, sr = librosa.load(file_path)

    start_point = start_slider.get()
    end_point = end_slider.get()
    play_time = (end_point - start_point)*1000
    pygame.mixer.music.play()
    pygame.mixer.music.set_pos(start_point)
    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        clock.tick(10)
        if pygame.mixer.music.get_pos() >= play_time:
            pygame.mixer.music.stop()
            break

    pygame.mixer.music.stop()

def open_file():
    global file_path 
    file_path = filedialog.askopenfilename()
    #Set Segment Stuff
    start_point = start_slider.get()
    end_point = end_slider.get()
    if end_point <= start_point:
        error_label.config(text="End point must be greater than start point")
        return
    else:
        error_label.config(text="")

    signal, sr = librosa.load(file_path)
    file_length = librosa.get_duration(y=signal, sr=sr)
    start_slider.config(to=file_length)
    end_slider.config(to=file_length)

    start_sample = int(start_point * sr)
    end_sample = int(end_point * sr)
    segment_signal = signal[start_sample:end_sample]

    # Pass File to Visualization
    plot_spectrogram(segment_signal, sr)
    create_mfcc(segment_signal, sr)
    # Update visualization_display with the visualization result
    visualization_result = Image.open("./mel.png")
    resized_img = visualization_result.resize((200,200), Image.LANCZOS)
    img = ImageTk.PhotoImage(resized_img)
    visualization_display.configure(image=img)
    visualization_display.image = img

    visualization_result = Image.open("./mfcc.png")
    resized_img = visualization_result.resize((200,200), Image.LANCZOS)
    img = ImageTk.PhotoImage(resized_img)
    cepstrum_display.configure(image=img)
    cepstrum_display.image = img

def refresh_file():
    if(file_path == None):
        error_label.config(text="ERROR: Must Select an Audio File Before Refreshing")
        return

    #Set Segment Stuff
    start_point = start_slider.get()
    end_point = end_slider.get()
    if end_point <= start_point:
        error_label.config(text="End point must be greater than start point")
        return
    else:
        error_label.config(text="")

    signal, sr = librosa.load(file_path)
    file_length = librosa.get_duration(y=signal, sr=sr)
    start_slider.config(to=file_length)
    end_slider.config(to=file_length)

    start_sample = int(start_point * sr)
    end_sample = int(end_point * sr)
    segment_signal = signal[start_sample:end_sample]

    # Pass File to Visualization
    plot_spectrogram(segment_signal, sr)
    create_mfcc(segment_signal, sr)
    # Update visualization_display with the visualization result
    visualization_result = Image.open("./mel.png")
    resized_img = visualization_result.resize((200,200), Image.LANCZOS)
    img = ImageTk.PhotoImage(resized_img)
    visualization_display.configure(image=img)
    visualization_display.image = img

    visualization_result = Image.open("./mfcc.png")
    resized_img = visualization_result.resize((200,200), Image.LANCZOS)
    img = ImageTk.PhotoImage(resized_img)
    cepstrum_display.configure(image=img)
    cepstrum_display.image = img

def process_segment():
    try:
        image = cv2.imread("./mfcc.png", cv2.IMREAD_COLOR)
    except:
        error_label.config(text="Failed to Load Image for Inference, ensure on has been generated")
        return
    data = []
    image = data_transform(image)
    data.append(image)
    data = torch.stack(data)
    try:
        model = CNN(data[0].shape)
    except:
        error_label.config(text="Failed to load model parameters")

    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    input = data[0].unsqueeze(0)
    output = model(input)[0]
    value, indices = torch.max(output, 0)
    result_label.config(text=label_names[indices])


#################################### Model ###################################################
label_names = ["BlueWhale", "BowheadWhale", "FalseKillerWhale", "FinFinbackWhale", "GrayWhale", "HumpbackWhale", "LongFinnedPilotWhale", "MelonHeadedWhale", "MinkeWhale", "ShortFinnedPilotWhale", "SouthernRightWhale", "SpermWhale"]  # Replace with your actual label names
model_path = "./model_12Whales.pth"
if(not os.path.exists(model_path)):
    print("Model Not Found, please check path")
    exit()

##################################### GUI #####################################################
root = tk.Tk()
root.title("Marine Mammal Identification")

file_label = tk.Label(root, text="Select Audio File:")
file_label.pack()
file_button = tk.Button(root, text="Open File", command=open_file)
file_button.pack()

#Visualization
frame = tk.Frame(root)
frame.pack()
visualization_label = tk.Label(root, text="< Mel Spectrogram  |  Mel Frequency Cepstral Coeffecients >")
visualization_label.pack()

visualization_display = tk.Label(frame)
cepstrum_display = tk.Label(frame)

visualization_display.grid(row=0, column=0, padx=5, pady=5)
cepstrum_display.grid(row=0, column=1, padx=5, pady=5)


segment_label = tk.Label(root, text="Select Segment start point and end point in seconds (Note: Model was trained on 1 second segments):")
segment_label.pack()

# Start point slider
start_frame = tk.Frame(root)
start_frame.pack()
start_label = tk.Label(start_frame, text="Start:")
start_label.pack(side="left")
start_slider = tk.Scale(start_frame, from_=0, to=100, orient="horizontal")
start_slider.pack(side="left")


# End point slider
end_frame = tk.Frame(root)
end_frame.pack()
end_label = tk.Label(end_frame, text="End:")
end_label.pack(side="left")
end_slider = tk.Scale(end_frame, from_=0, to=100, orient="horizontal")
end_slider.pack(side="left")

refresh_button = tk.Button(root, text="Refresh", command=refresh_file)
refresh_button.pack()

audio_button = tk.Button(root, text="Play Audio", command=start_audio_thread)
audio_button.pack()

process_button = tk.Button(root, text="Process Segment", command=process_segment)
process_button.pack()

result_label = tk.Label(root, fg="green")
result_label.pack()

error_label = tk.Label(root, fg="red")
error_label.pack()

output_display = tk.Label(root)
output_display.pack()

pygame.mixer.init()
root.mainloop()