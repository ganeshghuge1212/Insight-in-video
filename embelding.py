from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    return embedder.encode(chunks)
