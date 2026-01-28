import os
import glob
import shutil
import time
import yt_dlp
import whisper
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= MODELS =================
whisper_model = whisper.load_model("base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ================= AUDIO =================
def get_audio_from_youtube(url):
    outtmpl = os.path.join(AUDIO_DIR, "temp.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    files = glob.glob(os.path.join(AUDIO_DIR, "temp.*"))
    if not files:
        raise RuntimeError("Audio download failed")

    audio_path = os.path.join(OUTPUT_DIR, "audio.mp3")
    shutil.copy(files[0], audio_path)
    return audio_path

# ================= TRANSCRIBE =================
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path, fp16=False)
    return result["text"]

# ================= NLP =================
def summarize(text):
    return text[:1500]  # placeholder (LLM call here)

def generate_keypoints(text):
    return text[:800]   # placeholder

def generate_mindmap(text):
    return "Main Topic\n  Sub Topic\n    Detail"

def mindmap_to_edges(text):
    edges = []
    stack = []

    for line in text.splitlines():
        indent = len(line) - len(line.lstrip())
        level = indent // 2
        node = line.strip()

        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            edges.append({"parent": stack[-1][1], "child": node})

        stack.append((level, node))
    return edges

# ================= Q&A =================
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings
