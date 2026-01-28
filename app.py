import os
import glob
import shutil
from typing import List, Dict
import streamlit as st
import yt_dlp
import whisper
from google import genai
from streamlit_echarts import st_echarts
import time
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDH1MUpdMI8MsVLNNEp3dt2jIjARUX9kzE")
MODEL_NAME = "gemma-3-27b-it"
client = genai.Client(api_key=API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load embedding model for Q&A
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# ================= CLEAN =================
def clean_outputs():
    for folder in [AUDIO_DIR, OUTPUT_DIR]:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except:
                pass

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

# ================= TRANSCRIBE =================
def transcribe_audio(audio_path, progress_bar, status_text):
    status_text.text("📝 Transcribing audio (this may take a few minutes)...")
    progress_bar.progress(25)
   
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, task="translate", fp16=False)
    progress_bar.progress(40)
    return result["text"].strip()

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

# ================= SECONDS TO HH:MM:SS CONVERTER =================
def seconds_to_hhmmss(seconds: float):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

# ================= IMPROVED: YOUTUBE CHAPTERS GENERATOR (ADAPTIVE) =================
def generate_youtube_chapters(segments, progress_bar, status_text):
    """
    Generate YouTube chapters with adaptive timing based on video length.
    Longer videos automatically get more chapters with appropriate spacing.
    """
    status_text.text("📘 Generating YouTube chapters...")
    progress_bar.progress(42)
    
    if not segments:
        return "00:00:00 Introduction"
    
    # Calculate total video duration
    total_duration = segments[-1]["end"]
    
    # Dynamic chunk size based on video length
    # This ensures appropriate number of chapters for any video length
    if total_duration < 600:  # < 10 minutes
        MIN_LENGTH = 90
        IDEAL_LENGTH = 150
        MAX_LENGTH = 240
    elif total_duration < 1800:  # < 30 minutes
        MIN_LENGTH = 120
        IDEAL_LENGTH = 210
        MAX_LENGTH = 360
    elif total_duration < 3600:  # < 1 hour
        MIN_LENGTH = 180
        IDEAL_LENGTH = 300
        MAX_LENGTH = 480
    elif total_duration < 7200:  # < 2 hours
        MIN_LENGTH = 240
        IDEAL_LENGTH = 420
        MAX_LENGTH = 600
    else:  # 2+ hours
        MIN_LENGTH = 300
        IDEAL_LENGTH = 600
        MAX_LENGTH = 900
    
    chapters = []
    buffer_text = ""
    start_time = segments[0]["start"]
    
    for seg in segments:
        buffer_text += " " + seg["text"]
        current_duration = seg["end"] - start_time
        word_count = len(buffer_text.split())
        
        # Create chapter if we meet conditions
        should_create = False
        
        # Force break at maximum length
        if current_duration >= MAX_LENGTH:
            should_create = True
        
        # Natural break at ideal length with enough content
        elif current_duration >= IDEAL_LENGTH and word_count >= 50:
            should_create = True
        
        # Natural sentence break after minimum length
        elif current_duration >= MIN_LENGTH and word_count >= 40:
            if buffer_text.strip().endswith(('.', '!', '?')):
                # Check if next segment starts a new thought
                next_text = seg["text"].strip()
                if next_text and next_text[0].isupper():
                    should_create = True
        
        if should_create:
            title = generate_chapter_title(buffer_text)
            chapters.append(f"{seconds_to_hhmmss(start_time)} {title}")
            start_time = seg["end"]
            buffer_text = ""
    
    # Handle remaining content (final chapter)
    if buffer_text.strip() and len(buffer_text.split()) >= 30:
        title = generate_chapter_title(buffer_text)
        chapters.append(f"{seconds_to_hhmmss(start_time)} {title}")
    
    progress_bar.progress(48)
    
    # Ensure at least one chapter
    if not chapters:
        chapters.append("00:00:00 Introduction")
    
    return "\n".join(chapters)


def generate_chapter_title(text_chunk):
    """
    Generate a concise chapter title from a text chunk.
    """
    prompt = f"""
You are generating YouTube video chapters.

Rules:
- Title must be short (4–8 words)
- Clear topic name
- No punctuation
- No emojis
- No numbering
- No markdown
- Title case
- Capture the MAIN topic discussed

Transcript section: {text_chunk[:1000]}

Return ONLY the chapter title.
"""
    
    try:
        title = call_gemma_with_retry(prompt)
        title = title.replace("\n", " ").strip()
        
        # Clean up unwanted characters
        title = title.replace('"', '').replace("'", '').replace('*', '')
        
        # Ensure reasonable length
        words = title.split()
        if len(words) > 8:
            title = ' '.join(words[:8])
        
        return title if title else "Topic Discussion"
    except:
        return "Topic Discussion"

# ================= GEMMA CALL WITH RETRY =================
def call_gemma_with_retry(prompt, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            return res.text.strip()
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    wait = delay * (attempt + 1)
                    st.warning(f"⏳ Model busy, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise Exception("❌ Model overloaded. Try again later.")
            else:
                raise e

# ================= SUMMARY (INFORMATION-DENSE) =================
def summarize(text, progress_bar, status_text):
    status_text.text("📄 Generating summary...")
    progress_bar.progress(45)
   
    prompt = f"""
You are an expert knowledge distillation system.

GOAL: Produce an efficient and meaningful summary of the video content.

RULES:
- Summary length must NOT depend on transcript or video length
- Include ONLY high-information content
- Remove repetition, examples, stories, fillers
- Preserve core concepts, definitions, reasoning, conclusions
- Do NOT add new information
- Do NOT distort speaker intent

QUALITY STANDARD: The reader should understand the video without watching it.

FORMAT:
Title:
Summary:

FINAL CHECK: If removing a sentence reduces understanding, keep it. Otherwise, remove it.

Transcript: {text}
"""
   
    result = call_gemma_with_retry(prompt)
    progress_bar.progress(50)
    return result

# ================= KEY POINTS (NUMBERED LIST) =================
def generate_keypoints(text, progress_bar, status_text):
    status_text.text("📌 Extracting key points...")
    progress_bar.progress(55)
   
    prompt = f"""
You are an expert content analyst.

TASK: Extract ONLY the MAIN and MOST IMPORTANT points from the content.

CRITICAL RULES:
- Do NOT limit the number of points
- Include a point ONLY if it is essential to understanding the content
- Remove:
  - Repetitions
  - Examples
  - Side stories
  - Minor details
- Keep:
  - Core ideas
  - Definitions
  - Key arguments
  - Final conclusions
- One clear, concise sentence per point
- Do NOT explain the points
- Do NOT add new information

QUALITY CHECK: Each point should answer: "Would understanding the video suffer if this point was removed?"
If yes → keep it. If no → discard it.

FORMAT (IMPORTANT - Use numbered list):
1. Point 1
2. Point 2
3. Point 3
(continue as needed)

Transcript: {text}
"""
   
    result = call_gemma_with_retry(prompt)
    progress_bar.progress(60)
    return result

# ================= MINDMAP (IMPROVED) =================
def generate_mindmap(text, progress_bar, status_text):
    status_text.text("🧠 Creating mind map...")
    progress_bar.progress(70)
   
    topic_prompt = f"""
Read this transcript and identify the MAIN TOPIC in 2-6 words.
Be specific and accurate. Use ONLY information from the transcript.

Transcript (first part): {text[:1500]}

Answer with ONLY the topic name, nothing else.
"""
   
    try:
        main_topic = call_gemma_with_retry(topic_prompt).strip()
        main_topic = main_topic.replace('*', '').replace('#', '').replace('-', '').strip()
    except:
        main_topic = "Video Content"
   
    prompt = f"""
You are creating a hierarchical mind map from a video transcript.

MAIN TOPIC: {main_topic}

STRICT FORMATTING RULES:
1. Use EXACTLY 2 spaces for each indentation level
2. NO markdown symbols (no *, -, #, •, etc.)
3. NO numbering (no 1., 2., etc.)
4. Plain text only
5. Each line = one concept only

STRUCTURE RULES:
- Level 0: Main topic (no indentation)
- Level 1: Major subtopics (2 spaces)
- Level 2: Supporting details (4 spaces)
- Maximum 3 levels deep
- 4-8 subtopics at level 1
- 2-5 details per subtopic at level 2

CONTENT RULES:
- Extract ACTUAL concepts from the transcript
- Use short phrases (2-6 words)
- Be specific, not generic
- NO explanations or full sentences

EXACT FORMAT EXAMPLE:
Main Topic
  First Subtopic
    Detail A
    Detail B
  Second Subtopic
    Detail C
    Detail D
  Third Subtopic
    Detail E

Now create the mind map for this transcript: {text}

Remember: EXACTLY 2 spaces per level, NO symbols, plain text only.
"""
   
    result = call_gemma_with_retry(prompt)
    progress_bar.progress(75)
   
    cleaned = []
    for line in result.strip().split('\n'):
        clean_line = line
        for symbol in ['*', '#', '-', '•', '○', '●', '►', '▸', '→', '├', '└', '│', '`']:
            clean_line = clean_line.replace(symbol, '')
       
        import re
        clean_line = re.sub(r'^\s*\d+[\.\)]\s*', '', clean_line)
        clean_line = re.sub(r'^\s*[a-zA-Z][\.\)]\s*', '', clean_line)
        clean_line = clean_line.strip()
       
        if clean_line and len(clean_line) > 1:
            original_indent = len(line) - len(line.lstrip())
            normalized_indent = (original_indent // 2) * 2
            normalized_indent = min(normalized_indent, 4)
            cleaned.append(' ' * normalized_indent + clean_line)
   
    final_result = '\n'.join(cleaned)
    progress_bar.progress(80)
    return final_result

# ================= MINDMAP → EDGES (IMPROVED) =================
def mindmap_to_edges(text):
    edges = []
    stack = []
    lines = text.strip().split('\n')
   
    for line_num, line in enumerate(lines):
        if not line.strip():
            continue
       
        indent = len(line) - len(line.lstrip())
        level = indent // 2
        node = line.strip()
       
        if not node:
            continue
       
        while stack and stack[-1][0] >= level:
            stack.pop()
       
        if stack:
            parent_node = stack[-1][1]
            edges.append({"parent": parent_node, "child": node})
       
        stack.append((level, node))
   
    return edges

# ================= Q&A: TEXT CHUNKING =================
def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []
    start = 0
   
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
   
    return chunks

# ================= Q&A: BUILD FAISS INDEX =================
def build_faiss_index(text_chunks):
    embeddings = embed_model.encode(text_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
   
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
   
    return index, embeddings

# ================= Q&A: ANSWER QUESTION =================
def answer_question(question, index, chunks, top_k=3):
    # Embed question
    q_embedding = embed_model.encode([question])
    q_embedding = np.array(q_embedding).astype("float32")
   
    # Search
    distances, indices = index.search(q_embedding, top_k)
   
    # Get relevant chunks
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    context = "\n\n".join(relevant_chunks)
   
    # Generate answer using Gemma
    prompt = f"""
You are a helpful assistant answering questions based on video content.

Context from video:
{context}

Question: {question}

Instructions:
- Answer ONLY based on the context provided
- Be concise and accurate
- If the context doesn't contain enough information, say so
- Do not make up information

Answer:
"""
   
    try:
        answer = call_gemma_with_retry(prompt)
        return answer, relevant_chunks
    except:
        return "Sorry, I couldn't generate an answer. Please try again.", relevant_chunks

# ================= THEME DETECTION =================
def get_theme():
    try:
        return st.get_option("theme.base") or "light"
    except:
        return "light"

# ================= TREE BUILDER (ENHANCED) =================
def build_enhanced_tree(edges: List[Dict], is_dark_mode=False) -> Dict:
    if not edges:
        return {
            "name": "Mind Map Error",
            "children": [
                {"name": "No structure generated", "value": 0}
            ],
            "itemStyle": {"color": "#999"},
            "label": {"color": "#666"}
        }
   
    children_map, all_nodes, child_nodes = {}, set(), set()
   
    for e in edges:
        p, c = e["parent"], e["child"]
        all_nodes.add(p)
        all_nodes.add(c)
        child_nodes.add(c)
        children_map.setdefault(p, []).append(c)
   
    if is_dark_mode:
        colors = ["#a78bfa", "#60a5fa", "#34d399", "#fbbf24", "#f87171", "#f472b6"]
        text_color = "#e5e7eb"
        border_color = "#374151"
    else:
        colors = ["#7c3aed", "#2563eb", "#059669", "#d97706", "#dc2626", "#db2777"]
        text_color = "#1f2937"
        border_color = "#e5e7eb"
   
    def make_node(name, depth=0):
        childs = children_map.get(name, [])
        color = colors[min(depth, len(colors)-1)]
       
        return {
            "name": name,
            "value": len(childs),
            "children": [make_node(c, depth+1) for c in childs],
            "itemStyle": {
                "color": color,
                "borderColor": border_color,
                "borderWidth": 2
            },
            "label": {
                "fontSize": max(16 - depth*2, 11),
                "fontWeight": "bold" if depth == 0 else "normal",
                "color": text_color
            },
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 20,
                    "shadowColor": color,
                    "borderWidth": 3,
                },
                "label": {"color": text_color}
            },
            "collapsed": depth >= 1
        }
   
    roots = [n for n in all_nodes if n not in child_nodes]
   
    if len(roots) == 1:
        return make_node(roots[0])
    else:
        return {
            "name": "Video Content",
            "children": [make_node(r, 1) for r in roots],
            "itemStyle": {"color": colors[0], "borderColor": border_color},
            "label": {"color": text_color, "fontWeight": "bold"},
            "collapsed": False
        }

# ================= TREE OPTIONS (MORE SPACIOUS) =================
def get_tree_options(tree_data, is_dark_mode=False):
    tooltip_bg = "rgba(0, 0, 0, 0.9)" if is_dark_mode else "rgba(255, 255, 255, 0.95)"
    tooltip_text = "#e5e7eb" if is_dark_mode else "#1f2937"
    tooltip_border = "#4b5563" if is_dark_mode else "#d1d5db"
    line_color = "#6b7280" if is_dark_mode else "#9ca3af"

    return {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}",
            "backgroundColor": tooltip_bg,
            "borderColor": tooltip_border,
            "textStyle": {"color": tooltip_text}
        },
        "series": [{
            "type": "tree",
            "data": [tree_data],

            # 📌 POSITIONING
            "top": "10%",
            "left": "10%",
            "right": "20%",
            "bottom": "10%",
            "orient": "LR",

            # 📌 VERY IMPORTANT
            "expandAndCollapse": True,
            # "initialTreeDepth": 0,   # 🔥 ONLY ROOT NODE SHOWN

            "symbolSize": 14,

            "label": {
                "position": "right",
                "align": "left",
                "fontSize": 14,
                "overflow": "break",
                "width": 200
            },

            "leaves": {
                "label": {
                    "position": "right",
                    "align": "left"
                }
            },

            "lineStyle": {
                "color": line_color,
                "width": 2.5,
                "curveness": 0.4
            },

            # 📌 NODE INTERACTION
            "emphasis": {
                "focus": "descendant"
            }
        }]
    }

# ================= CUSTOM CSS =================
def apply_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .content-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ================= YOUTUBE STYLE TRANSCRIPT UI (ADDED) =================
def render_youtube_style_transcript(ts_text):
    st.markdown("""
    <style>
    .transcript-container {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px;
        background: rgba(255, 255, 255, 0.05);
    }
    .transcript-line {
        display: flex;
        margin-bottom: 12px;
        align-items: flex-start;
    }
    .timestamp {
        color: #1976d2;
        font-weight: bold;
        margin-right: 12px;
        min-width: 60px;
        font-size: 14px;
    }
    .transcript-text {
        flex: 1;
        line-height: 1.5;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)
   
    st.markdown('<div class="transcript-container">', unsafe_allow_html=True)
    for row in ts_text.split("\n"):
        if "||" in row:
            t, txt = row.split("||", 1)
            st.markdown(
                f'<div class="transcript-line">'
                f'<div class="timestamp">{t}</div>'
                f'<div class="transcript-text">{txt}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Insight in Video", layout="wide", page_icon="🎬")
apply_custom_css()

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 Insight in Video</h1>
    <p>Transform YouTube videos into comprehensive study materials</p>
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown("### 📎 Enter Video Details")
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_btn = st.button("🚀 Generate Study Materials", use_container_width=True)

if generate_btn and url:
    # Clear all previous session state data
    for key in list(st.session_state.keys()):
        if key != "active_tab":
            del st.session_state[key]
   
    clean_outputs()
    progress = st.progress(0)
    status = st.empty()
   
    try:
        audio = get_audio_from_youtube(url, progress, status)
        transcript = transcribe_audio(audio, progress, status)
       
        # Generate chapters using improved function
        model = load_whisper_model_cached()
        whisper_result = model.transcribe(audio, task="translate", fp16=False)
        chapters = generate_youtube_chapters(
            whisper_result["segments"], progress, status
        )
        st.session_state["youtube_chapters"] = chapters
       
        st.session_state["transcript"] = transcript
        st.session_state["summary"] = summarize(transcript, progress, status)
        st.session_state["keypoints"] = generate_keypoints(transcript, progress, status)
       
        mindmap_raw = generate_mindmap(transcript, progress, status)
        st.session_state["mindmap"] = mindmap_raw
        edges = mindmap_to_edges(mindmap_raw)
        st.session_state["edges_count"] = len(edges)
        is_dark = get_theme() == "dark"
        st.session_state["tree"] = build_enhanced_tree(edges, is_dark)
       
        # Build Q&A index
        status.text("🔍 Building Q&A index...")
        progress.progress(85)
        full_text = (
            transcript + "\n\n" +
            st.session_state["summary"] + "\n\n" +
            st.session_state["keypoints"]
        )
        chunks = chunk_text(full_text)
        faiss_index, _ = build_faiss_index(chunks)
        st.session_state["chunks"] = chunks
        st.session_state["faiss_index"] = faiss_index
       
        progress.progress(100)
        status.text("✅ All materials generated successfully!")
        time.sleep(1)
        status.empty()
        progress.empty()
       
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 **Troubleshooting tips:**\n- Make sure yt-dlp is updated: `pip install --upgrade yt-dlp`\n- Try a different video\n- Check if the video is available in your region")

# Display results with horizontal navigation
if "transcript" in st.session_state:
    st.markdown("---")
    st.markdown("### 📚 Generated Materials")
   
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "transcript"
   
    col1, col2, col3, col4, col5, col6 = st.columns(6)
   
    with col1:
        if st.button("📝 Transcript", use_container_width=True):
            st.session_state.active_tab = "transcript"
   
    with col2:
        if st.button("⏱ YouTube Chapters", use_container_width=True):
            st.session_state.active_tab = "yt_transcript"
   
    with col3:
        if st.button("📄 Summary", use_container_width=True):
            st.session_state.active_tab = "summary"
   
    with col4:
        if st.button("📌 Key Points", use_container_width=True):
            st.session_state.active_tab = "keypoints"
   
    with col5:
        if st.button("🧠 Mind Map", use_container_width=True):
            st.session_state.active_tab = "mindmap"
   
    with col6:
        if st.button("💬 Q&A", use_container_width=True):
            st.session_state.active_tab = "qa"
   
    st.markdown("<br>", unsafe_allow_html=True)
   
    # Display active tab
    if st.session_state.active_tab == "transcript":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("#### 📝 Transcript")
        st.write(st.session_state["transcript"])
        st.markdown('</div>', unsafe_allow_html=True)
   
    elif st.session_state.active_tab == "yt_transcript":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("#### ⏱ YouTube Chapters")
        st.info("📌 Copy & paste directly into YouTube description")
        st.code(
            st.session_state["youtube_chapters"],
            language="text"
        )
        st.markdown('</div>', unsafe_allow_html=True)
   
    elif st.session_state.active_tab == "summary":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("#### 📄 Summary")
        st.write(st.session_state["summary"])
        st.markdown('</div>', unsafe_allow_html=True)
   
    elif st.session_state.active_tab == "keypoints":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("#### 📌 Key Points")
        st.write(st.session_state["keypoints"])
        st.markdown('</div>', unsafe_allow_html=True)
   
    elif st.session_state.active_tab == "mindmap":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Mind Map")
        is_dark = get_theme() == "dark"
        options = get_tree_options(st.session_state["tree"], is_dark)
        st_echarts(options, height="600px", key="mindmap")
        st.markdown('</div>', unsafe_allow_html=True)
   
    elif st.session_state.active_tab == "qa":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("#### 💬 Ask Questions About This Video")
       
        question = st.text_input("Enter your question:", key="qa_input")
       
        if st.button("Get Answer", key="qa_button"):
            if question.strip():
                with st.spinner("🔍 Searching for answer..."):
                    answer, sources = answer_question(
                        question,
                        st.session_state["faiss_index"],
                        st.session_state["chunks"]
                    )
                   
                    st.markdown("**Answer:**")
                    st.write(answer)
                   
                    with st.expander("📚 View relevant sections"):
                        for i, chunk in enumerate(sources, 1):
                            st.markdown(f"**Section {i}:**")
                            st.write(chunk)
                            st.markdown("---")
            else:
                st.warning("Please enter a question.")
       
        st.markdown('</div>', unsafe_allow_html=True)