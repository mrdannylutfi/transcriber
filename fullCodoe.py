import os
import json
import wave
import pyaudio
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from transformers import pipeline

# OPTIONAL: Explicitly link ffmpeg if pydub can't find it automatically
# AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"

class AudioTranscriber:
    def __init__(self, model_path: str, confidence_threshold: float = 0.7, use_nn_punc: bool = True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model path not found: {model_path}")
        self.MODEL_PATH = model_path
        self.model = Model(self.MODEL_PATH)
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        
        # Initialize a standard Hugging Face Neural Network pipeline for restoration
        self.punc_pipeline = None
        if use_nn_punc:
            print("🧠 Loading standard Neural Network for punctuation & casing...")
            try:
                # Utilizing a highly efficient sequence-to-sequence punctuation model
                self.punc_pipeline = pipeline("text2text-generation", model="felfel/t5-punctuation-restoration")
            except Exception as e:
                print(f"⚠️ Could not load NN model ({e}). Falling back to timing-based formatting.")

    def _prepare_audio(self, source_path: str) -> str:
        """Converts any audio file to 16kHz, 16-bit, mono WAV for Vosk."""
        print(f"\n🔄 Converting: {os.path.basename(source_path)}...")
        audio = AudioSegment.from_file(source_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        temp_path = f"temp_{os.path.basename(source_path)}.wav"
        audio.export(temp_path, format="wav")
        return temp_path

    def _apply_nn_formatting(self, raw_text: str) -> str:
        """Passes text through the standard neural network to recover punctuation/casing."""
        if self.punc_pipeline and raw_text.strip():
            try:
                result = self.punc_pipeline(raw_text, max_length=512)
                return result[0]['generated_text']
            except Exception:
                pass
        return raw_text.capitalize()

    def _generate_report_data(self, word_timeline: list) -> dict:
        """Processes raw timestamps into punctuated strings alongside Markdown/HTML highlights."""
        raw_text = " ".join([item["word"] for item in word_timeline])
        
        # 1. Recover structure via Neural Network
        punctuated_text = self._apply_nn_formatting(raw_text)
        punc_words = punctuated_text.split()
        
        markdown_words = []
        html_words = []
        low_conf_items = []
        
        # 2. Map structural predictions back against Vosk data to retain highlighting integrity
        for i, item in enumerate(word_timeline):
            conf = item.get("conf", 1.0)
            
            # Use punctuated word if available, else standard fallback
            display_word = punc_words[i] if i < len(punc_words) else item["word"]
            
            if conf < self.CONFIDENCE_THRESHOLD:
                clean_word = display_word.rstrip(",.?!")
                trailing = display_word[len(clean_word):]
                
                markdown_words.append(f"`{clean_word}`{trailing}")
                html_words.append(f"<mark class='low-conf' title='Confidence: {conf:.2f}'>{clean_word}</mark>{trailing}")
                low_conf_items.append(item)
            else:
                markdown_words.append(display_word)
                html_words.append(display_word)
                
        return {
            "clean_text": punctuated_text,
            "markdown_text": " ".join(markdown_words),
            "html_text": " ".join(html_words),
            "low_confidence_words": low_conf_items
        }

    def _generate_html_report(self, report_data: dict, original_filepath: str):
        """Generates a beautifully styled, standalone HTML report file."""
        base_path, _ = os.path.splitext(original_filepath)
        output_file = f"{base_path}_report.html"
        
        filename = os.path.basename(original_filepath)
        low_count = len(report_data["low_confidence_words"])
        total_words = len(report_data["html_text"].split())
        accuracy_rate = ((total_words - low_count) / max(total_words, 1)) * 100

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Transcription Report - {filename}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 40px auto; padding: 0 20px; background-color: #f8f9fa; }}
        .card {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 24px; }}
        h1 {{ color: #1a73e8; margin-top: 0; border-bottom: 2px solid #e8eaed; padding-bottom: 12px; }}
        h2 {{ color: #3c4043; font-size: 1.3rem; margin-top: 0; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
        .metric {{ background: #f1f3f4; padding: 16px; border-radius: 6px; text-align: center; }}
        .metric-val {{ font-size: 1.5rem; font-weight: bold; color: #1a73e8; margin-bottom: 4px; }}
        .metric-label {{ font-size: 0.85rem; color: #5f6368; text-transform: uppercase; letter-spacing: 0.5px; }}
        .transcript-box {{ font-size: 1.1rem; color: #202124; white-space: pre-wrap; background: #fff; padding: 20px; border-left: 4px solid #1a73e8; }}
        mark.low-conf {{ background-color: #fce8e6; color: #c5221f; border-bottom: 2px dotted #c5221f; padding: 0 2px; cursor: help; border-radius: 3px; }}
        .badge {{ display: inline-block; padding: 2px 6px; font-size: 0.8rem; font-weight: bold; border-radius: 4px; background: #eed; margin: 2px; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🎙️ Audio Transcription Audit</h1>
        <div class="meta-grid">
            <div class="metric"><div class="metric-val">{filename}</div><div class="metric-label">Source File</div></div>
            <div class="metric"><div class="metric-val">{accuracy_rate:.1f}%</div><div class="metric-label">Confidence Score Rating</div></div>
            <div class="metric"><div class="metric-val">{low_count}</div><div class="metric-label">Flagged Words (< {self.CONFIDENCE_THRESHOLD})</div></div>
        </div>
    </div>

    <div class="card">
        <h2>Interactive Document View</h2>
        <div class="transcript-box">
{report_data["html_text"]}
        </div>
    </div>
    
    <div class="card">
        <h2>Flagged Terms Registry</h2>
        {f"<p style='color: #137333;'>🎉 Excellent clarity! No words dropped below the threshold setting.</p>" if not low_count else "".join([f"<span class='badge' title='Timestamp: {w.get('start',0)}s - {w.get('end',0)}s (Confidence: {w.get('conf',0):.2f})'>{w['word']} ({w.get('conf',0):.2f})</span>" for w in report_data["low_confidence_words"]])}
    </div>
</body>
</html>"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"📊 Visual HTML Dashboard exported safely to: {output_file}")

    def _save_output(self, word_timeline: list, format_choice: str, original_filepath: str):
        """Saves outputs across preferred data format arrays."""
        base_path, _ = os.path.splitext(original_filepath)
        report_data = self._generate_report_data(word_timeline)
        
        # Always build the HTML visualization report file alongside standard formats
        self._generate_html_report(report_data, original_filepath)

        if format_choice == "txt":
            output_file = f"{base_path}_transcript.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_data["markdown_text"])
            print(f"💾 Saved Markdown text document to: {output_file}")
            
        elif format_choice == "json":
            output_file = f"{base_path}_transcript.json"
            payload = {
                "filename": os.path.basename(original_filepath),
                "full_text": report_data["clean_text"],
                "markdown_text": report_data["markdown_text"],
                "html_text": report_data["html_text"],
                "confidence_threshold_used": self.CONFIDENCE_THRESHOLD,
                "low_confidence_count": len(report_data["low_confidence_words"]),
                "low_confidence_words": report_data["low_confidence_words"],
                "word_timestamps": word_timeline
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            print(f"💾 Saved JSON tracking arrays to: {output_file}")

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
                        all_words.extend(res["result"])
        except KeyboardInterrupt:
            print("\n🛑 Stopped listening.")
        finally:
            stream.stop_stream(); stream.close(); p.terminate()
            final_res = json.loads(recognizer.FinalResult())
            if "result" in final_res:

if all_words:self._save_output(all_words, save_format, "live_stream_output.wav")def process_single_file(self, file_path: str, save_format: str = "json"):if not os.path.exists(file_path):print(f"❌ File not found: {file_path}")returntemp_file = self._prepare_audio(file_path)all_words = []try:with wave.open(temp_file, "rb") as wf:recognizer = KaldiRecognizer(self.model, wf.getframerate())recognizer.SetWords(True)print("📝 Transcribing and passing text layers through pipeline execution...")while True:data = wf.readframes(4000)if not data: break

if recognizer.AcceptWaveform(data):res = json.loads(recognizer.Result())if "result" in res:all_words.extend(res["result"])final = json.loads(recognizer.FinalResult())if "result" in final:all_words.extend(final["result"])finally:if os.path.exists(temp_file):os.remove(temp_file)if all_words:self._save_output(all_words, save_format, file_path)else:print("⚠️ No speech recognized in this file.")def process_folder(self, folder_path: str, save_format: str = "json"):if not os.path.isdir(folder_path):print(f"❌ Folder path does not exist: {folder_path}")
returnsupported_extensions = ('.mp3', '.wav', '.m4a', '.flac', '.ogg')files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]if not files:print("📁 No valid audio files found in the folder.")returnprint(f"📚 Found {len(files)} files to batch process.")for idx, file_path in enumerate(files, 1):print(f"\n🚀 [File {idx}/{len(files)}]")self.process_single_file(file_path, save_format=save_format)print("\n✅ Batch execution complete!")

if name == "main":VOSK_MODEL_DIR = "path_to_your_vosk_model_folder"transcriber = AudioTranscriber(VOSK_MODEL_DIR, confidence_threshold=0.7)print("Select Operation:\n1) Microphone Stream\n2) Single Audio File\n3) Folder Batch Loop")choice = input("Enter selection (1/2/3): ").strip()save_choice = input("Save format? (json/txt): ").strip().lower()if choice == "1":transcriber.stream_microphone(save_format=save_choice)elif choice == "2":transcriber.process_single_file(input("Enter file path: ").strip(), save_format=save_choice)elif choice == "3":transcriber.process_folder(input("Enter folder path: ").strip(), save_format=save_choice)
