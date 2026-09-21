import os
import json
import wave
import pyaudio
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer

# OPTIONAL: Explicitly link ffmpeg if pydub can't find it automatically
# AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"

class AudioTranscriber:
    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        self.MODEL_PATH = model_path
        self.model = Model(self.MODEL_PATH)
        self.CONFIDENCE_THRESHOLD = confidence_threshold

    def _prepare_audio(self, source_path: str) -> str:
        """Converts any audio file to 16kHz, 16-bit, mono WAV for Vosk."""
        print(f"\n🔄 Converting: {os.path.basename(source_path)}...")
        audio = AudioSegment.from_file(source_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        temp_path = f"temp_{os.path.basename(source_path)}.wav"
        audio.export(temp_path, format="wav")
        return temp_path

    def _format_transcript_text(self, word_timeline: list) -> str:
        """Formats text, flagging low-confidence words with [brackets*]."""
        formatted_words = []
        for item in word_timeline:
            word = item["word"]
            # Flag if the confidence score is strictly less than the threshold
            if item.get("conf", 1.0) < self.CONFIDENCE_THRESHOLD:
                formatted_words.append(f"[{word}*]")
            else:
                formatted_words.append(word)
        return " ".join(formatted_words)

    def _save_output(self, word_timeline: list, format_choice: str, original_filepath: str):
        """Saves transcribed segments, highlights, or timestamps to a file."""
        base_path, _ = os.path.splitext(original_filepath)
        
        # Build clean string with brackets highlighting weaker guesses
        flagged_text = self._format_transcript_text(word_timeline)
        # Standard unflagged string
        clean_text = " ".join([item["word"] for item in word_timeline])
        
        # Filter out low confidence segments for granular inspection
        low_conf_items = [
            item for item in word_timeline 
            if item.get("conf", 1.0) < self.CONFIDENCE_THRESHOLD
        ]

        if format_choice == "txt":
            output_file = f"{base_path}_transcript.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(flagged_text)
            print(f"💾 Saved flagged text transcript to: {output_file}")
            
        elif format_choice == "json":
            output_file = f"{base_path}_transcript.json"
            payload = {
                "filename": os.path.basename(original_filepath),
                "full_text": clean_text,
                "flagged_text": flagged_text,
                "confidence_threshold_used": self.CONFIDENCE_THRESHOLD,
                "low_confidence_count": len(low_conf_items),
                "low_confidence_words": low_conf_items,
                "word_timestamps": word_timeline
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            print(f"💾 Saved JSON transcript with filter analytics to: {output_file}")

    def stream_microphone(self, save_format: str = "json"):
        sample_rate = 16000
        recognizer = KaldiRecognizer(self.model, sample_rate)
        recognizer.SetWords(True)
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=8192)
        stream.start_stream()
        print("🎤 Listening... Press Ctrl+C to stop.")
        
        all_words = []
        try:
            while True:
                data = stream.read(4096, exception_on_overflow=False)
                if not data: break
                if recognizer.AcceptWaveform(data):
                    res = json.loads(recognizer.Result())
                    if "result" in res:
                        print(self._format_transcript_text(res["result"]))
                        all_words.extend(res["result"])
        except KeyboardInterrupt:
            print("\n🛑 Stopped listening.")
        finally:
            stream.stop_stream(); stream.close(); p.terminate()
            
            final_res = json.loads(recognizer.FinalResult())
            if "result" in final_res:
                print(self._format_transcript_text(final_res["result"]))
                all_words.extend(final_res["result"])
                
            if all_words and save_format in ["txt", "json"]:
                self._save_output(all_words, save_format, "live_stream_output")

    def process_single_file(self, file_path: str, save_format: str = "json"):
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        temp_file = self._prepare_audio(file_path)
        all_words = []
        
        try:
            with wave.open(temp_file, "rb") as wf:
                recognizer = KaldiRecognizer(self.model, wf.getframerate())
                recognizer.SetWords(True)
                
                print("📝 Transcribing and filtering confidence scores...")
                while True:
                    data = wf.readframes(4000)
                    if not data: break
                    if recognizer.AcceptWaveform(data):
                        res = json.loads(recognizer.Result())
                        if "result" in res:
                            all_words.extend(res["result"])
                
                final = json.loads(recognizer.FinalResult())
                if "result" in final:
                    all_words.extend(final["result"])
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        if all_words:
            print(f"\n📊 Summary: Found {len([w for w in all_words if w.get('conf', 1.0) < self.CONFIDENCE_THRESHOLD])} low-confidence words.")
            self._save_output(all_words, save_format, file_path)
        else:
            print("⚠️ No speech recognized in this file.")

    def process_folder(self, folder_path: str, save_format: str = "json"):
        if not os.path.isdir(folder_path):
            print(f"❌ Folder path does not exist: {folder_path}")
            return
            
        supported_extensions = ('.mp3', '.wav', '.m4a', '.flac', '.ogg')
        files_to_process = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if f.lower().endswith(supported_extensions)
        ]
        
        if not files_to_process:
            print("📁 No valid audio files found in the folder.")
            return
            
        print(f"📚 Found {len(files_to_process)} audio files to process.")
        for idx, file_path in enumerate(files_to_process, 1):
            print(f"\n🚀 [File {idx}/{len(files_to_process)}]")
            self.process_single_file(file_path, save_format=save_format)
        print("\n✅ Batch processing complete!")

if __name__ == "__main__":
    MODEL_DIR = "path_to_your_vosk_model_folder"
    
    # Initialize transcriber with a custom threshold limit (0.7 = 70% confidence)
    transcriber = AudioTranscriber(MODEL_DIR, confidence_threshold=0.7)
    
    print("Select Operation:")
    print("1) Live microphone streaming")
    print("2) Single audio file transcription")
    print("3) Batch process an entire folder")
    choice = input("Enter option (1/2/3): ").strip()
    
    save_choice = input("Save format? (json/txt): ").strip().lower()
    if save_choice not in ["json", "txt"]:
        save_choice = "json"
        
    if choice == "1":
        transcriber.stream_microphone(save_format=save_choice)
    elif choice == "2":
        path = input("Enter audio file path: ").strip()
        transcriber.process_single_file(path, save_format=save_choice)
    elif choice == "3":
        folder_path = input("Enter folder directory path: ").strip()
        transcriber.process_folder(folder_path, save_format=save_choice)
