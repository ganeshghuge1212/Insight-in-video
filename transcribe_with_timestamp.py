# ================= TRANSCRIBE WITH TIMESTAMPS (ADDED) =================
@st.cache_resource
def load_whisper_model_cached():
    return whisper.load_model("base")

def transcribe_audio_with_timestamps(audio_path, progress_bar, status_text):
    status_text.text("🕒 Transcribing with timestamps...")
    progress_bar.progress(30)

    model = load_whisper_model_cached()
    result = model.transcribe(audio_path, task="translate", fp16=False)

    lines = []
    for seg in result.get("segments", []):
        start = time.strftime("%M:%S", time.gmtime(seg["start"]))
        text = seg["text"].strip()
        lines.append(f"{start}||{text}")

    progress_bar.progress(40)
    return "\n".join(lines)