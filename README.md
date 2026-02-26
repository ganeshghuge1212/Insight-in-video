# 🎬 Insight in Video

**AI-Powered YouTube Video Analysis — Transcribe, Summarize, and Explore any video in seconds.**

Insight in Video is a Streamlit web app that takes a YouTube URL and automatically generates a full transcript, executive summary, key takeaways, an interactive timeline, a visual mind map, and a Q&A interface — all powered by OpenAI Whisper and Google Gemma.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Full Transcript** | Transcribes and translates audio to English using OpenAI Whisper |
| ⏱ **Video Timeline** | Automatically segments the video into major thematic blocks with timestamps |
| 📋 **Executive Summary** | Concise paragraph-style summary of the video's main argument |
| 📌 **Key Takeaways** | Numbered list of the most important points |
| 🧠 **Interactive Mind Map** | Hierarchical knowledge tree rendered with ECharts (click to expand/collapse) |
| 💬 **Q&A Chat** | Ask questions about the video — answered using semantic search + Gemma |

All analysis tasks run **in parallel in the background** so you don't have to wait for each one sequentially.

---

## 🖼️ Demo

> Paste a YouTube link → Click **Generate Insights** → Explore results across 6 tabs.

---

## 🛠️ Tech Stack

- **Frontend/UI** — [Streamlit](https://streamlit.io/)
- **Audio Download** — [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- **Transcription** — [OpenAI Whisper](https://github.com/openai/whisper) (`base` model)
- **LLM** — [Google Gemma 3 27B](https://ai.google.dev/) via `google-genai`
- **Embeddings & Semantic Search** — [SentenceTransformers](https://www.sbert.net/) + [FAISS](https://github.com/facebookresearch/faiss)
- **Visualization** — [streamlit-echarts](https://github.com/andfanilo/streamlit-echarts)
- **Audio processing** — ffmpeg

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- `ffmpeg` installed on your system

**Install ffmpeg:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/insight-in-video.git
   cd insight-in-video
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**

   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   > Get your free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)

4. **Run the app**
   ```bash
   streamlit run App.py
   ```

5. Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
insight-in-video/
├── App.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed to Git)
├── .gitignore           # Excludes .env, data/, output/
├── data/
│   └── audio/           # Temporary audio downloads (auto-created)
└── output/              # Processed output files (auto-created)
```

---

## ⚙️ Configuration

You can change the LLM model used for generation by editing this line in `App.py`:

```python
MODEL_NAME = "gemma-3-27b-it"
```

Any model available via the `google-genai` client can be used here.

---

## 🔒 Security

- Your API key is loaded from `.env` and **never hardcoded**.
- Make sure `.env` is listed in your `.gitignore` before pushing to GitHub.

---

## 📦 Requirements

```
streamlit
python-dotenv
yt-dlp
openai-whisper
google-genai
streamlit-echarts
sentence-transformers
faiss-cpu
numpy
ffmpeg-python
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)
