from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Use the exact same model we used for ingestion
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
CHROMA_DB_DIR = "/data/chroma"

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'}
)

print("Connecting to Vector DB...")
vector_store = Chroma(
    persist_directory=CHROMA_DB_DIR, 
    embedding_function=embeddings
)

# Fetch all metadata from the database
collection = vector_store.get()
sources = [meta.get('source', 'Unknown') for meta in collection['metadatas']]

# Count the unique sources
unique_sources = set(sources)

print("\n=== DATABASE CONTENTS ===")
print(f"Total Text Chunks: {len(sources)}")
print("Unique Sources Found:")
for source in unique_sources:
    chunk_count = sources.count(source)
    print(f" - {source} ({chunk_count} chunks)")
print("=========================\n")
