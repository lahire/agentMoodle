import os
import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Import the database logic
from database import init_db, get_db, SessionLocal, UserInteraction

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Math Tutor Agent API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change this to ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Configuration ---
CHROMA_DB_DIR = "/data/chroma"
# Note: Keeping the "-base" model we discussed to prevent Apple Silicon crashes
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" 
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
LLM_MODEL = "llama3" 
PROMPT_FILE_PATH = "/app/system_prompt.txt"
FALLBACK_PHRASE = "No puedo responder a esto basándome en el material proporcionado."

# --- Global Variables ---
embeddings = None
vector_store = None
llm = None

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    user_hash: str
    question: str
    confidence_level: str

class SourceDocument(BaseModel):
    content: str
    source: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    init_db()
    global embeddings, vector_store, llm
    logging.info("Initializing models and connecting to Vector DB...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings
    )
    llm = Ollama(base_url=OLLAMA_HOST, model=LLM_MODEL)
    logging.info("Ready to accept queries!")

# --- Background Task for Pedagogical Logging ---
def log_interaction_task(
    user_hash: str, 
    question: str, 
    confidence_level: str, 
    sources_str: str, 
    answered_successfully: bool, 
    question_length: int
):
    """Runs after the API responds to extract the topic and save to Postgres."""
    # 1. Quick Zero-Shot prompt to extract the math topic using the local LLM
    topic_prompt = f"Extrae el tema matemático principal de esta pregunta en 1 a 3 palabras. Responde SOLO con el tema, sin explicaciones ni puntuación extra.\nPregunta: {question}\nTema:"
    
    try:
        topic_extracted = llm.invoke(topic_prompt).strip()
    except Exception as e:
        logging.error(f"Failed to extract topic: {e}")
        topic_extracted = "Desconocido"

    # 2. Open a fresh database session and save
    db = SessionLocal()
    try:
        interaction = UserInteraction(
            user_hash=user_hash,
            question=question,
            confidence_level=confidence_level,
            sources_referenced=sources_str,
            answered_successfully=answered_successfully,
            question_length=question_length,
            topic_extracted=topic_extracted
        )
        db.add(interaction)
        db.commit()
        logging.info(f"Pedagogical data logged -> Topic: {topic_extracted} | Success: {answered_successfully}")
    except Exception as e:
        logging.error(f"Failed to log interaction to database: {e}")
        db.rollback()
    finally:
        db.close()

# --- The Core RAG Endpoint ---
@app.post("/ask", response_model=QueryResponse)
async def ask_math_question(request: QueryRequest, background_tasks: BackgroundTasks):
    if not vector_store:
        raise HTTPException(status_code=500, detail="Database not initialized.")

    e5_query = f"query: {request.question}"
    docs = vector_store.similarity_search(e5_query, k=3)
    
    # Calculate initial metrics
    question_length = len(request.question.split())
    
    if not docs:
        # If no docs found, log the failure in the background and return fallback
        background_tasks.add_task(
            log_interaction_task, request.user_hash, request.question, 
            request.confidence_level, "Ninguna", False, question_length
        )
        return QueryResponse(answer=FALLBACK_PHRASE, sources=[])

    context_text = "\n\n---\n\n".join([doc.page_content.replace("passage: ", "") for doc in docs])
    
    try:
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as file:
            template_string = file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="El archivo de plantilla del prompt no fue encontrado.")

    prompt_template = PromptTemplate(
        input_variables=["confidence", "context", "question"],
        template=template_string
    )
    
    final_prompt = prompt_template.format(
        confidence=request.confidence_level,
        context=context_text,
        question=request.question
    )

    answer = llm.invoke(final_prompt)

    # Format the sources and extract pedagogy metrics
    sources = []
    source_names = []
    
    for doc in docs:
        src = doc.metadata.get('source', 'Desconocido')
        pg = doc.metadata.get('page', 0)
        
        sources.append(SourceDocument(
            content=doc.page_content.replace("passage: ", "")[:200] + "...", 
            source=src,
            page=pg
        ))
        
        # Clean up the filename for the database log (e.g., "calculus.pdf (p.12)")
        filename = os.path.basename(src)
        source_names.append(f"{filename} (p.{pg})")

    sources_str = ", ".join(list(set(source_names))) 
    answered_successfully = FALLBACK_PHRASE not in answer

    # Dispatch the background task to handle LLM topic extraction and DB commit
    background_tasks.add_task(
        log_interaction_task, 
        request.user_hash, 
        request.question, 
        request.confidence_level, 
        sources_str, 
        answered_successfully, 
        question_length
    )

    return QueryResponse(answer=answer, sources=sources)
