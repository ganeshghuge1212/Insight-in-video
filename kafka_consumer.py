import json
import traceback
from kafka import KafkaConsumer
from video_processor import process_video

consumer = KafkaConsumer(
    "video_jobs",
    bootstrap_servers="localhost:9092",
    group_id="video_worker_v1",          # ✅ fixed consumer group
    auto_offset_reset="latest",          # ✅ ignore old bad messages
    enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

print("🚀 Kafka consumer started and waiting for jobs...")

for message in consumer:
    job = message.value
    print("📩 Received job:", job)

    # ✅ Safe extraction (NO KeyError)
    job_id = job.get("job_id")
    url = job.get("youtube_url")

    # ✅ Validate message
    if not job_id or not url:
        print("⚠️ Invalid job message, skipping:", job)
        continue

    try:
        print(f"🎬 Processing job_id={job_id}")
        process_video(job_id, url)
        print(f"✅ Job {job_id} completed successfully")

    except Exception as e:
        print(f"❌ Job {job_id} failed")
        traceback.print_exc()
