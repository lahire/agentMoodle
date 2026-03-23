import os
import shutil
import logging
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DATA_DIR = "/data/corpus"      
CHROMA_DB_DIR = "/data/chroma" 

# Using the powerful large model
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

def ingest_data():
    all_documents = []

    # 1. Load the PDFs
    logging.info(f"Checking for PDFs in {DATA_DIR}...")
    try:
        # Load all .pdf files using PyMuPDF
        pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        pdf_docs = pdf_loader.load()
        all_documents.extend(pdf_docs)
        logging.info(f"Loaded {len(pdf_docs)} pages from PDFs.")
    except Exception as e:
        logging.error(f"Error loading PDFs: {e}")

    # 2. Load the Manual Transcripts (.txt files)
    logging.info(f"Checking for TXT transcripts in {DATA_DIR}...")
    try:
        # Load all .txt files using standard TextLoader
        txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
        txt_docs = txt_loader.load()
        all_documents.extend(txt_docs)
        logging.info(f"Loaded {len(txt_docs)} manual text documents.")
    except Exception as e:
        logging.error(f"Error loading TXT files: {e}")

    # Safety check
    if not all_documents:
        logging.warning("No PDFs or TXT files found. Please add content to 'data/corpus'.")
        return

    # 3. Split the text into manageable chunks
    logging.info("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_documents(all_documents)
    
    # 4. Apply the E5 required "passage: " prefix to every chunk
    logging.info("Applying E5 'passage: ' prefix to chunks...")
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"
        
    logging.info(f"Prepared {len(chunks)} text chunks for embedding.")

    # 5. Initialize Local Embeddings
    logging.info(f"Initializing local embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} 
    )

    # 6. Clean up the old database to prevent ghost data
    if os.path.exists(CHROMA_DB_DIR):
        logging.info("Cleaning up old database...")
        shutil.rmtree(CHROMA_DB_DIR)

    # 7. Create and persist the Vector Database
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
    ingest_data()
