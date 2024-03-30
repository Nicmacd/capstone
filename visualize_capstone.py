# import tkinter as tk
# import pygame
# from tkinter import messagebox
# from PIL import Image, ImageTk  # Make sure to install the Pillow library using: pip install Pillow
#
# class AudioPlayerApp:
#     def __init__(self, master):
#         self.master = master
#         self.master.title("Audio Player")
#         self.master.configure(bg="#ADD8E6")  # Set the background color to light blue
#         self.current_index = 0
#
#         # List of audio file paths and corresponding answers
#         self.audio_files = [
#                         {"file": "audio/blue.wav", "answer_text": "Blue Whale", "answer_image": "images/blue_whale.jpeg", "model_guess": "Model guess: Blue Whale"},
#                         {"file": "audio/bowhead.wav", "answer_text": "Bowhead Whale", "answer_image": "images/bowhead_whale.jpeg", "model_guess": "Model guess: Bow Headed Whale"},
#                         {"file": "audio/falseKiller.wav", "answer_text": "False Killer Whale", "answer_image": "images/false_killer.jpeg", "model_guess": "Model guess: Long Finned Whale"},
#                         {"file": "audio/finbackedWhale.wav", "answer_text": "Finback Whale", "answer_image": "images/finback.png", "model_guess": "Model guess: Finback Whale"},
#                         {"file": "audio/gray.wav", "answer_text": "Gray Whale", "answer_image": "images/gray.jpeg", "model_guess": "Model guess: Minke Whale"},
#                         {"file": "audio/humpback.wav", "answer_text": "Humpback Whale",  "answer_image": "images/humpback.jpeg", "model_guess": "Model guess: Humpback Whale"},
#                         {"file": "audio/longfinnedPilot.wav", "answer_text": "Long Finned Pilot Whale",  "answer_image": "images/750x500-long-finned-pilot-whale.png", "model_guess": "Model guess: Long Finned Whale"},
#                         {"file": "audio/Melon.wav", "answer_text": "Melon Headed Whale",  "answer_image": "images/melonHeaded.jpeg"},
#                         {"file": "audio/Minke.wav", "answer_text": "Minke Whale",  "answer_image": "images/minke-whale-vancouver-1200x600.jpg"},
#                         {"file": "audio/shortfinned.wav", "answer_text": "Short Finned Pilot Whale",  "answer_image": "images/ShortFinnedPilotWhale.jpeg"},
#                         {"file": "audio/Southern.wav", "answer_text": "Southern Right Whale",  "answer_image": "images/SouthernRightWhale.jpg"},
#                         {"file": "audio/Sperm.wav", "answer_text": "Sperm Whale",  "answer_image": "images/Sperm-whale.jpeg"},
#                         # Add more audio file and answer pairs as needed
#         ]
#
#         # Initialize pygame mixer
#         pygame.mixer.init()
#
#         # Create GUI components
#         self.create_buttons()
#
#     def create_buttons(self):
#         # Create a title label
#         title_label = tk.Label(self.master, text="Marine Mammal Protection from Human Induced Noises Model Demo", font=("Helvetica", 16, "bold"), bg="#ADD8E6")
#         title_label.grid(row=0, column=0, columnspan=3, pady=10, sticky="ew")  # Use sticky="ew" to center the title
#
#         for index, audio_info in enumerate(self.audio_files):
#             play_button = tk.Button(self.master, text=f"Play {index + 1}", command=lambda i=index: self.play_audio(i), fg="dark blue", highlightbackground="#ADD8E6")
#             play_button.grid(row=index + 1, column=0, padx=10, pady=10, sticky="ew")
#
#             answer_button = tk.Button(self.master, text=f"Answer {index + 1}", command=lambda i=index: self.show_answer(i), fg="dark blue", highlightbackground="#ADD8E6")
#             answer_button.grid(row=index + 1, column=1, padx=10, pady=10, sticky="ew")
#
#             model_guess_button = tk.Button(self.master, text=f"Model Guess {index + 1}", command=lambda i=index: self.show_model_guess(i), fg="dark blue", highlightbackground="#ADD8E6")
#             model_guess_button.grid(row=index + 1, column=2, padx=10, pady=10, sticky="ew")
#
#         self.master.columnconfigure(0, weight=1)
#         self.master.columnconfigure(1, weight=1)
#         self.master.columnconfigure(2, weight=1)
#
#     def play_audio(self, index):
#         if pygame.mixer.music.get_busy():
#             pygame.mixer.music.stop()
#
#         audio_file = self.audio_files[index]["file"]
#         pygame.mixer.music.load(audio_file)
#         pygame.mixer.music.play()
#
#     def show_answer(self, index):
#         answer_text = self.audio_files[index]["answer_text"]
#         answer_image_path = self.audio_files[index]["answer_image"]
#
#         # Create a new window to display the answer
#         answer_window = tk.Toplevel(self.master)
#         answer_window.title("Answer")
#
#         # Show answer text
#         answer_text_label = tk.Label(answer_window, text=answer_text, font=("Helvetica", 20, "bold"))
#         answer_text_label.pack(pady=10)
#
#         # Show answer image
#         if answer_image_path:
#             answer_image = Image.open(answer_image_path)
#             answer_photo = ImageTk.PhotoImage(answer_image)
#             answer_image_label = tk.Label(answer_window, image=answer_photo)
#             answer_image_label.image = answer_photo
#             answer_image_label.pack(pady=10)
#
#     def show_model_guess(self, index):
#         model_guess = self.audio_files[index]["model_guess"]
#         messagebox.showinfo("Model Guess", model_guess)
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = AudioPlayerApp(root)
#     root.mainloop()

import tkinter as tk
import pygame
from tkinter import messagebox
from PIL import Image, ImageTk  # Make sure to install the Pillow library using: pip install Pillow

class AudioPlayerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Player")
        self.master.configure(bg="#ADD8E6")  # Set the background color to light blue
        self.current_index = 0

        # List of audio file paths and corresponding answers
        self.audio_files = [
            {"file": "audio/bowhead.wav", "answer_text": "Bowhead Whale", "answer_image": "images/bowhead_whale.jpeg"},
            {"file": "audio/Southern.wav", "answer_text": "Southern Right Whale", "answer_image": "images/SouthernRightWhale.jpg"},
            {"file": "audio/gray.wav", "answer_text": "Gray Whale", "answer_image": "images/gray.jpeg"},
            {"file": "audio/Minke.wav", "answer_text": "Northern Right Whale", "answer_image": "images/minke-whale-vancouver-1200x600.jpg"},
            {"file": "audio/blue.wav", "answer_text": "Blue Whale", "answer_image": "images/blue_whale.jpeg"},
            {"file": "audio/finbackedWhale.wav", "answer_text": "Finback Whale", "answer_image": "images/finback.png"},
            {"file": "audio/humpback.wav", "answer_text": "Humpback Whale", "answer_image": "images/humpback.jpeg"},
            {"file": "audio/Sperm.wav", "answer_text": "Sperm Whale", "answer_image": "images/Sperm-whale.jpeg"},
            {"file": "audio/Melon.wav", "answer_text": "Melon Headed Whale",  "answer_image": "images/melonHeaded.jpeg"},
            {"file": "audio/longfinnedPilot.wav", "answer_text": "Long Finned Pilot Whale", "answer_image": "images/750x500-long-finned-pilot-whale.png"},
            {"file": "audio/shortfinned.wav", "answer_text": "Short Finned Pilot Whale", "answer_image": "images/ShortFinnedPilotWhale.jpeg"},
            {"file": "audio/falseKiller.wav", "answer_text": "False Killer Whale", "answer_image": "images/false_killer.jpeg"},
            # Add more audio file and answer pairs as needed
        ]

        # self.audio_files = [
        #     {"file": "capstone/data/audioData/demo/seal_CC12L_502.wav", "answer_text": "seal",
        #      "answer_image": "capstone/images/spotted_seal.jpeg"},
        #     {"file": "capstone/data/audioData/demo/orca_BE7A_032.wav", "answer_text": "orca",
        #      "answer_image": "capstone/images/orca_image.jpeg"},
        #     {"file": "capstone/data/audioData/demo/dolphin_BD19D_246.wav", "answer_text": "dolphin",
        #      "answer_image": "capstone/images/dolphin.jpeg"},
        #     {"file": "capstone/data/audioData/demo/whale_BA2A_4050.wav", "answer_text": "whale",
        #      "answer_image": "capstone/images/Sperm-whale.jpeg"},
        # ]

        # Initialize pygame mixer
        pygame.mixer.init()

        # Create GUI components
        self.create_buttons()

    def create_buttons(self):
        # Create a title label
        title_label = tk.Label(self.master, text="Marine Mammal Protection from Human Induced Noises Model Demo", font=("Helvetica", 16, "bold"), bg="#ADD8E6")
        title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")  # Use sticky="ew" to center the title

        for index, audio_info in enumerate(self.audio_files):
            play_button = tk.Button(self.master, text=f"Play {index + 1}", command=lambda i=index: self.play_audio(i), fg="dark blue", highlightbackground="#ADD8E6")
            play_button.grid(row=index + 1, column=0, padx=10, pady=10, sticky="ew")

            answer_button = tk.Button(self.master, text=f"Answer {index + 1}", command=lambda i=index: self.show_answer(i), fg="dark blue", highlightbackground="#ADD8E6")
            answer_button.grid(row=index + 1, column=1, padx=10, pady=10, sticky="ew")

        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)
    def play_audio(self, index):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        audio_file = self.audio_files[index]["file"]
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

    def show_answer(self, index):
        answer_text = self.audio_files[index]["answer_text"]
        answer_image_path = self.audio_files[index]["answer_image"]

        # Create a new window to display the answer
        answer_window = tk.Toplevel(self.master)
        answer_window.title("Answer")

        # Show answer text
        answer_text_label = tk.Label(answer_window, text=answer_text, font=("Helvetica", 20, "bold"))
        answer_text_label.pack(pady=10)

        # Show answer image
        if answer_image_path:
            answer_image = Image.open(answer_image_path)
            answer_photo = ImageTk.PhotoImage(answer_image)
            answer_image_label = tk.Label(answer_window, image=answer_photo)
            answer_image_label.image = answer_photo
            answer_image_label.pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioPlayerApp(root)
    root.mainloop()
