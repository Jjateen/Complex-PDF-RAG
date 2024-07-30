import os

class Config:
    MODEL = os.environ.get('MODEL', "llama3")
    EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', "all-MiniLM-L6-v2")
    HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE = os.environ.get('HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE', "cuda")
