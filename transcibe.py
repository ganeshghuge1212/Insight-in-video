# ================= TRANSCRIBE =================
def transcribe_audio(audio_path, progress_bar, status_text):
    status_text.text("📝 Transcribing audio (this may take a few minutes)...")
    progress_bar.progress(25)

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, task="translate", fp16=False)

    progress_bar.progress(40)
    return result["text"].strip()