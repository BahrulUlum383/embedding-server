from sentence_transformers import SentenceTransformer

def load_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")
