# -*- coding: utf-8 -*-
import sys
import os
import threading
import subprocess
"""
if sys.version_info < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox
    import ttk
else:
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

class FfmpegSubExtractorApp(object):
    def __init__(self, root):
        self.root = root
        self.root.title("FFmpeg Text Extractor (Fixed Path)")
        self.root.geometry("650x550")
        self.root.minsize(500, 400)
        
        self.video_path = ""
        self.extracted_text = ""
        self.create_widgets()

    def create_widgets(self):
        file_frame = ttk.LabelFrame(self.root, text=" 1. Select Video File ")
        file_frame.pack(fill="x", padx=15, pady=10, ipady=5)

        self.btn_browse = ttk.Button(file_frame, text="Browse Video", command=self.browse_video)
        self.btn_browse.pack(side="left", padx=5, pady=5)

        self.lbl_file_path = ttk.Label(file_frame, text="No file selected", font=("Arial", 9))
        self.lbl_file_path.pack(side="left", padx=10, fill="x", expand=True)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=15, pady=5)

        self.btn_transcribe = ttk.Button(control_frame, text="Extract Text via FFmpeg", command=self.start_extraction_thread)
        self.btn_transcribe.pack(side="left", padx=5)
        self.btn_transcribe.config(state="disabled")

        self.btn_export = ttk.Button(control_frame, text="Export to .txt", command=self.export_to_txt)
        self.btn_export.pack(side="left", padx=5)
        self.btn_export.config(state="disabled")

        self.lbl_status = ttk.Label(self.root, text="Status: Ready", font=("Arial", 10))
        self.lbl_status.pack(anchor="w", padx=20, pady=5)

        text_frame = ttk.LabelFrame(self.root, text=" 2. Extracted Text Output ")
        text_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.txt_output = tk.Text(text_frame, wrap="word", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, command=self.txt_output.yview)
        self.txt_output.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.txt_output.pack(side="left", fill="both", expand=True)

    def browse_video(self):
        """Robust file selection that handles tuples, strings, and empty selections."""
        file_data = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.srt"), ("All Files", "*.*")]
        )
        
        # Handle tuple returns or empty values from certain OS/Tkinter combinations
        if isinstance(file_data, tuple):
            selected_file = file_data[0] if file_data else ""
        else:
            selected_file = file_data

        if selected_file and selected_file != "()":
            if sys.version_info < 3 and isinstance(selected_file, unicode):
                self.video_path = selected_file.encode('utf-8')
            else:
                self.video_path = str(selected_file)
                
            display_name = os.path.basename(self.video_path)
            self.lbl_file_path.config(text=display_name)
            self.btn_transcribe.config(state="normal")
            self.btn_export.config(state="disabled")
            self.txt_output.delete("1.0", "end")
            self.extracted_text = ""
        else:
            self.lbl_file_path.config(text="No file selected")

    def start_extraction_thread(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("Error", "Invalid file path selected.")
            return
            
        self.btn_browse.config(state="disabled")
        self.btn_transcribe.config(state="disabled")
        self.btn_export.config(state="disabled")
        self.lbl_status.config(text="Status: FFmpeg is processing streams...")
        
        task_thread = threading.Thread(target=self.process_ffmpeg)
        task_thread.daemon = True
        task_thread.start()

    def process_ffmpeg(self):
        try:
            ffmpeg_cmd = ["ffmpeg", "-y", "-i", self.video_path, "-an", "-vn", "-c:s", "srt", "-f", "srt", "-"]
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            if sys.version_info >= (3, 5):
                res = subprocess.run(ffmpeg_cmd, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout_data, stderr_data = res.stdout, res.stderr
            else:
                p = subprocess.Popen(ffmpeg_cmd, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout_data, stderr_data = p.communicate()
            
            text_result = stdout_data.decode('utf-8', 'ignore')
            error_log = stderr_data.decode('utf-8', 'ignore')

            if not text_result.strip():
                text_result = "[FFmpeg Report] No embedded text subtitle stream found.\n\n" + error_log

            self.root.after(0, self.on_success, text_result)

        except Exception as e:
            self.root.after(0, self.on_failure, str(e))

    def on_success(self, text):
        self.extracted_text = text
        self.txt_output.insert("end", text)
        self.lbl_status.config(text="Status: FFmpeg extraction completed.")
        self.btn_browse.config(state="normal")
        self.btn_transcribe.config(state="normal")
        self.btn_export.config(state="normal")

    def on_failure(self, error_message):
        self.lbl_status.config(text="Status: Execution failed.")
        messagebox.showerror("Process Error", error_message)
        self.btn_browse.config(state="normal")
        self.btn_transcribe.config(state="normal")

    def export_to_txt(self):
        if not self.extracted_text.strip():
            return
        save_path = filedialog.asksaveasfilename(
            title="Save Output Text",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if isinstance(save_path, tuple):
            save_path = save_path[0] if save_path else ""
            
        if save_path:
            if sys.version_info < 3 and isinstance(save_path, unicode):
                save_path = save_path.encode('utf-8')
            try:
                mode = "wb" if sys.version_info < 3 else "w"
                with open(save_path, mode) as f:
                    content = self.extracted_text.encode('utf-8') if sys.version_info < 3 else self.extracted_text
                    f.write(content)
                messagebox.showinfo("Success", "File exported successfully.")
            except Exception as e:
                messagebox.showerror("Export Failed", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = FfmpegSubExtractorApp(root)
    root.mainloop()
