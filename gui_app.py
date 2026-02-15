"""
Movinet Action Recognition - GUI Application
Real-time action recognition with webcam/video file support
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
import time
from typing import Optional
import numpy as np

# Import our classifier
from movinet_classifier import MovinetClassifier


class MovinetGUI:
    """GUI Application for Movinet Action Recognition"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎬 Movinet Action Recognition")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a2e")
        
        # Model
        self.classifier: Optional[MovinetClassifier] = None
        self.model_loaded = False
        
        # Video
        self.video_capture = None
        self.current_frame = None
        self.is_processing = False
        self.is_streaming = False
        
        # UI Setup
        self.setup_ui()
        
        # Load model in background
        self.load_model_thread()
    
    def setup_ui(self):
        """Setup UI components"""
        # Title
        title = tk.Label(
            self.root,
            text="🎬 Movinet Action Recognition",
            font=("Arial", 24, "bold"),
            bg="#1a1a2e",
            fg="#e94560"
        )
        title.pack(pady=20)
        
        # Model Status Frame
        status_frame = tk.Frame(self.root, bg="#16213e", padx=20, pady=10)
        status_frame.pack(fill=tk.X, padx=20)
        
        self.status_label = tk.Label(
            status_frame,
            text="⏳ Loading model...",
            font=("Arial", 12),
            bg="#16213e",
            fg="#ffffff"
        )
        self.status_label.pack(side=tk.LEFT)
        
        self.gpu_label = tk.Label(
            status_frame,
            text="GPU: Checking...",
            font=("Arial", 10),
            bg="#16213e",
            fg="#0f3460"
        )
        self.gpu_label.pack(side=tk.RIGHT)
        
        # Control Frame
        control_frame = tk.Frame(self.root, bg="#1a1a2e", padx=20, pady=10)
        control_frame.pack(fill=tk.X, padx=20)
        
        # Model selection
        tk.Label(control_frame, text="Model:", bg="#1a1a2e", fg="white").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="a0")
        model_combo = ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=["a0", "a1", "a2", "a3"],
            width=5,
            state="readonly"
        )
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Fine-tuned model path
        self.pretrained_var = tk.StringVar(value="")
        tk.Button(
            control_frame,
            text="📂 Fine-tuned Model",
            command=self.select_pretrained,
            bg="#16213e",
            fg="white",
            padx=10,
            pady=3
        ).pack(side=tk.LEFT, padx=5)
        
        # Streaming toggle
        self.streaming_var = tk.BooleanVar(value=False)
        streaming_check = tk.Checkbutton(
            control_frame,
            text="Streaming Mode",
            variable=self.streaming_var,
            bg="#1a1a2e",
            fg="white",
            selectcolor="#0f3460"
        )
        streaming_check.pack(side=tk.LEFT, padx=20)
        
        # Buttons
        btn_frame = tk.Frame(control_frame, bg="#1a1a2e")
        btn_frame.pack(side=tk.RIGHT)
        
        self.load_btn = tk.Button(
            btn_frame,
            text="📁 Load Video",
            command=self.load_video,
            bg="#0f3460",
            fg="white",
            padx=15,
            pady=5
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.webcam_btn = tk.Button(
            btn_frame,
            text="📷 Webcam",
            command=self.toggle_webcam,
            bg="#0f3460",
            fg="white",
            padx=15,
            pady=5
        )
        self.webcam_btn.pack(side=tk.LEFT, padx=5)
        
        # Video Display
        self.video_label = tk.Label(
            self.root,
            text="No video loaded",
            bg="#16213e",
            fg="gray",
            font=("Arial", 14)
        )
        self.video_label.pack(pady=20)
        
        # Results Frame
        results_frame = tk.LabelFrame(
            self.root,
            text="🎯 Predictions",
            bg="#16213e",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Results text
        self.results_text = tk.Text(
            results_frame,
            height=10,
            bg="#0f3460",
            fg="white",
            font=("Consolas", 11),
            state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
        # Info bar
        info_frame = tk.Frame(self.root, bg="#16213e", padx=20, pady=5)
        info_frame.pack(fill=tk.X, padx=20)
        
        self.info_label = tk.Label(
            info_frame,
            text="Ready",
            bg="#16213e",
            fg="#888888"
        )
        self.info_label.pack()
    
    def select_pretrained(self):
        file_path = filedialog.askopenfilename(
            title="Select Fine-tuned Model",
            filetypes=[
                ("PyTorch models", "*.pth *.pt"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.pretrained_var.set(file_path)
            self.info_label.config(text=f"Selected: {file_path}")
            self.load_model_thread()
    
    def load_model_thread(self):
        def load():
            try:
                model_id = self.model_var.get()
                use_streaming = self.streaming_var.get()
                pretrained_path = self.pretrained_var.get() if self.pretrained_var.get() else ""
                
                self.classifier = MovinetClassifier(
                    model_id=model_id,
                    use_streaming=use_streaming,
                    pretrained_path=pretrained_path if pretrained_path else None
                )
                self.model_loaded = True
                
                device_name = str(self.classifier.device)
                gpu_text = f"Device: {device_name}"
                
                if pretrained_path:
                    if self.classifier.custom_classes:
                        gpu_text += f" | Classes: {', '.join(self.classifier.custom_classes)}"
                
                self.root.after(0, self.update_status, "✅ Model loaded!", gpu_text)
                
            except Exception as e:
                self.root.after(0, self.update_status, f"❌ Error: {str(e)}", "")
                self.root.after(0, messagebox.showerror, "Error", str(e))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def update_status(self, status: str, gpu_text: str = ""):
        """Update status labels"""
        self.status_label.config(text=status)
        if gpu_text:
            self.gpu_label.config(text=gpu_text)
    
    def load_video(self):
        """Load video file"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please wait for model to load!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.process_video(file_path)
    
    def toggle_webcam(self):
        """Toggle webcam"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please wait for model to load!")
            return
        
        if self.is_streaming:
            self.stop_streaming()
        else:
            self.start_webcam()
    
    def start_webcam(self):
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Cannot access webcam!")
            return
        
        self.is_streaming = True
        
        if self.classifier:
            try:
                self.classifier.init_streaming(buffer_size=8)
                print(f"Streaming initialized, use_streaming={self.classifier.use_streaming}")
            except Exception as e:
                print(f"Init streaming error: {e}")
        
        self.webcam_btn.config(text="⏹ Stop", bg="#e94560")
        self.info_label.config(text="Webcam active")
        
        self.process_webcam()
    
    def stop_streaming(self):
        """Stop streaming"""
        self.is_streaming = False
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        if self.classifier and hasattr(self.classifier, 'reset_stream'):
            self.classifier.reset_stream()
        
        self.webcam_btn.config(text="📷 Webcam", bg="#0f3460")
        self.video_label.config(text="Streaming stopped")
        self.info_label.config(text="Ready")
    
    def process_webcam(self):
        if not self.is_streaming:
            return
        
        ret, frame = self.video_capture.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (640, 480))
            from PIL import Image, ImageTk
            img = Image.fromarray(frame_small)
            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image="", text="")
            self.video_label.imgtk = photo
            self.video_label.configure(image=photo)
            
            if self.classifier and hasattr(self.classifier, 'use_streaming') and self.classifier.use_streaming:
                try:
                    print("Calling process_stream_frame...")
                    results = self.classifier.process_stream_frame(frame, top_k=3)
                    print(f"Results: {results}")
                    self.update_results(results)
                except Exception as e:
                    print(f"Prediction error: {e}")
        
        if self.is_streaming:
            self.root.after(33, self.process_webcam)
    
    def process_video(self, video_path: str):
        """Process video file"""
        self.info_label.config(text="Processing video...")
        
        def process():
            try:
                # Get predictions
                results = self.classifier.predict(video_path, top_k=5)
                self.root.after(0, self.update_results, results)
                self.root.after(0, self.info_label.config, {"text": "Done!"})
                
            except Exception as e:
                self.root.after(0, messagebox.showerror, "Error", str(e))
                self.root.after(0, self.info_label.config, {"text": "Error"})
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def update_results(self, results):
        """Update results display"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, "🎯 Top Predictions:\n\n")
        
        for i, (label, prob) in enumerate(results, 1):
            bar = "█" * int(prob * 20)
            self.results_text.insert(
                tk.END,
                f"{i}. {label:25s} {prob:.2%} {bar}\n"
            )
        
        self.results_text.config(state=tk.DISABLED)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = MovinetGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
