# GenAI Video Insight Engine

An AI-powered system that converts YouTube videos into structured knowledge by generating transcriptions, summaries, key points, mind maps, and interactive Q&A. The project helps users learn faster by extracting essential information without watching entire videos.

## Problem Statement
Educational and informational videos are often long and time-consuming. Users struggle to quickly find key insights, revise concepts, or ask questions based on video content.

## Solution
This system automatically processes YouTube videos and transforms them into concise, searchable, and interactive learning material using Generative AI and NLP techniques.

## Features
- YouTube video audio extraction
- Automatic speech-to-text transcription
- AI-generated summaries and key points
- Concept-based mind map generation
- Question & Answer system for interactive learning
- Fast knowledge retrieval using vector search

## Tech Stack
- Python
- Streamlit
- Whisper (Speech-to-Text)
- LLMs (Gemma)
- FAISS Vector Database
- Kafka (for data pipeline)
- NLP & Embeddings

## Project Structure
- `app12.py` – Streamlit application
- `main_12.py` – Core orchestration logic
- `audio.py / transcribe.py` – Audio processing
- `chunking.py` – Text chunking
- `embedding.py` – Vector embedding generation
- `FAISS_Vector_Database.py` – Vector storage and retrieval
- `kafka_producer.py / kafka_consumer.py` – Kafka integration
- `utils/` – Helper functions

## How to Run
```bash
pip install -r requirements.txt
streamlit run app12.py
