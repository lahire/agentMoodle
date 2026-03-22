import os
import logging
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration paths
DATA_DIR = "/data/corpus"      
CHROMA_DB_DIR = "/data/chroma" 

# Using the E5 multilingual large model
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

def ingest_pdfs():
    logging.info(f"Checking for PDFs in {DATA_DIR}...")
    
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        logging.warning("No PDFs found. Please place your math PDFs in the 'data/corpus' directory.")
        return

    # 1. Load the PDFs using PyMuPDF for better Spanish character and layout retention
    logging.info("Loading PDFs with PyMuPDF...")
    loader = DirectoryLoader(
        DATA_DIR, 
        glob="**/*.pdf", 
        loader_cls=PyMuPDFLoader
    )
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} pages from the corpus.")

    # 2. Split the text into manageable chunks
    logging.info("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Apply the E5 required "passage: " prefix to every chunk
    logging.info("Applying E5 'passage: ' prefix to chunks...")
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"
        
    logging.info(f"Prepared {len(chunks)} text chunks for embedding.")

    # 4. Initialize Local Embeddings
    logging.info(f"Initializing local embedding model: {EMBEDDING_MODEL_NAME}...")
    # model_kwargs={'device': 'cpu'} ensures it runs smoothly if you don't have a GPU exposed to Docker
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} 
    )

    # 5. Create and persist the Vector Database
    logging.info("Generating embeddings and storing in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    if hasattr(vector_store, 'persist'):
        vector_store.persist()
        
    logging.info(f"Ingestion complete! Database saved to {CHROMA_DB_DIR}.")

if __name__ == "__main__":
    ingest_pdfs()
