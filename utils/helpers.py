import os
import uuid

def generate_job_id():
    return str(uuid.uuid4())

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
