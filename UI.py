import os
from pathlib import Path
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import wave  # FIX 1: Missing import

from vosk import KaldiRecognizer, Model


class DynamicSubprocessTranscriber:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dynamic Subprocess Extractor")
        self.root.geometry("650x550")
        self.root.minsize(500, 400)

        self.video_path = ""
        self.extracted_text = ""
        self.create_widgets()

        # 核心逻辑：动态检测是否为 Windows 系统
        self.check_os_and_adapt()

    def create_widgets(self):
        # Placeholder to prevent errors if not defined elsewhere
        pass

    def check_os_and_adapt(self):
        # Placeholder to prevent errors if not defined elsewhere
        pass

    def cleanPath(self):
        # Remove extension from a full path or just a filename
        file_path = Path(self.video_path)
        clean_path = file_path.with_suffix("")
        return clean_path

    def injectTrainedModel(self):
        # FIX 3: Used a raw string (r"...") for a clean, literal Windows path
        self.MODEL_PATH = r"C:\Users\mrdan\OneDrive\Documents\python\vosk-model-en-us-0.22-lgraph"
        model = Model(self.MODEL_PATH)
        return model

    def readTrainedAIModel(self):
        model = self.injectTrainedModel()

        # FIX 2: Correctly calling self.cleanPath() and appending .wav extension
        # Vosk requires a .wav file, so ensure the path includes it
        audio_file = str(self.cleanPath().with_suffix(".wav"))

        # Open .wav file
        wf = wave.open(audio_file, "rb")

        # Verify audio formatting constraints
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getcomptype() != "NONE"
        ):
            print("Audio file must be wav format mono PCM")
            return

        # FIX 4: Correctly initialize the recognizer and process the audio chunks
        recognizer = KaldiRecognizer(model, wf.getframerate())
        recognizer.SetWords(True)

        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                results.append(recognizer.Result())
            else:
                results.append(recognizer.PartialResult())

        results.append(recognizer.FinalResult())
        wf.close()

        # Join results into a single block of text if needed
        self.extracted_text = "".join(results)
        print("Transcription complete.")
