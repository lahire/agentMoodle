import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Fetch the connection string
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://moodleuser:moodlepassword@db:5432/moodledb")

# 2. Setup the SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 3. Define the Schema with Pedagogical Tracking
class UserInteraction(Base):
    __tablename__ = "user_interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_hash = Column(String, index=True, nullable=False) 
    timestamp = Column(DateTime, default=datetime.utcnow)  

    # Core Data
    question = Column(Text, nullable=False)                
    confidence_level = Column(String, nullable=False)      
    
    # Pedagogical Metrics
    sources_referenced = Column(String, nullable=True)     
    answered_successfully = Column(Boolean, default=True)  
    question_length = Column(Integer, nullable=False)      
    topic_extracted = Column(String, nullable=True)        

# 4. Helper function to create the tables on startup
def init_db():
    logging.info("Initializing PostgreSQL tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables verified/created successfully.")
    except Exception as e:
        logging.error(f"Failed to connect to PostgreSQL: {e}")

# 5. Dependency function for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
