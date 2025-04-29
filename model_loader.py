from sentence_transformers import SentenceTransformer

def load_model():
    model_path = "./models/bge-small-en-v1.5"
    return SentenceTransformer(model_path)
