# ================= AUDIO (FIXED FOR 403 ERROR) =================
def get_audio_from_youtube(url, progress_bar, status_text):
    status_text.text("📥 Downloading audio from YouTube...")
    progress_bar.progress(10)

    outtmpl = os.path.join(AUDIO_DIR, "temp.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
                "skip": ["hls", "dash"]
            }
        },
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-us,en;q=0.5",
            "Sec-Fetch-Mode": "navigate"
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e):
            status_text.text("🔄 Retrying with alternative method...")
            ydl_opts["extractor_args"]["youtube"]["player_client"] = ["android"]
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
        else:
            raise e

    files = glob.glob(os.path.join(AUDIO_DIR, "temp.*"))
    if not files:
        raise Exception("No audio file was downloaded. Please check the URL.")
    
    audio_path = os.path.join(OUTPUT_DIR, "audio.mp3")
    shutil.copy(files[0], audio_path)

    progress_bar.progress(20)
    return audio_path


# ================= CLEAN =================
def clean_outputs():
    for folder in [AUDIO_DIR, OUTPUT_DIR]:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except:
                pass