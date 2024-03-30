# Marine Mammal Classification UI Docs 

## Overview
The python script "user_interface.py" provides a graphical interface for a user to leverage the trained marine mammal classification model. The interface accepts an audio file, extracts a segment of it, visualizes it, and uses it for inference. 

## Requirements
#### Python modules
- Python
-  tkinter
-   PIL
-   matplotlib.pyplot
-   scipy
-   librosa
-   numpy
-   torch
-   cv2
-   pygame

#### Files
- Source Code (user_interface.py)
- Trained model (model_12Whales.pth) 
- Input (Any .wav audio signal)

## Getting Started
To get started, simply run the user_interface.py application with python, then update the sliders to select a valid segment of audio, then click "Open File", select your audio file, view the visualizations and listen to the audio playback, then click process result to view the model inference displayed in green text. 

## Features
### Audio Manipulation 
The user has the ability to manipulate and analyze audio data (ex. looking at specific segment of uploaded file).
### Visualization
The interface provides visual representation of data to help with users understanding.
### Inference
Audio Files can be passed through a trained model for inference.  
### Audio Playback
The interface can play the selected segment of audio  