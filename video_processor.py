import os
import pickle

# ✅ ABSOLUTE BASE PATH (CRITICAL)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

from core_processing import (
    get_audio_from_youtube,
    transcribe_audio,
    summarize,
    generate_keypoints,
    generate_mindmap,
    mindmap_to_edges,
    chunk_text,
    build_faiss_index
)

def process_video(job_id, url):
    print(f"🎬 Processing job {job_id}")

    # ✅ USE ABSOLUTE OUTPUT DIRECTORY
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    # 🚀 PURE BACKGROUND PROCESSING (NO STREAMLIT)
    audio = get_audio_from_youtube(url)
    transcript = transcribe_audio(audio)

    summary = summarize(transcript)
    keypoints = generate_keypoints(transcript)
    mindmap = generate_mindmap(transcript)

    edges = mindmap_to_edges(mindmap)

    full_text = transcript + "\n" + summary + "\n" + keypoints
    chunks = chunk_text(full_text)
    faiss_index, _ = build_faiss_index(chunks)

    # ✅ ABSOLUTE RESULT PATH
    result_path = os.path.join(output_dir, "results.pkl")

    with open(result_path, "wb") as f:
        pickle.dump(
            {
                "transcript": transcript,
                "summary": summary,
                "keypoints": keypoints,
                "mindmap": mindmap,
                "edges": edges
            },
            f
        )

    # ✅ DEBUG CONFIRMATION (VERY IMPORTANT)
    print("📂 Result saved at:", result_path)
    print(f"✅ Job {job_id} finished successfully")
