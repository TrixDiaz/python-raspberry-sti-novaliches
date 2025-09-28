from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

# Database connection string
DATABASE_URL = "postgresql://neondb_owner:npg_VgHuhCp2Jx6P@ep-wispy-firefly-adswi51k-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# Create engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

class MotionDetection(Base):
    __tablename__ = "motion_detection"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    motion_data = Column(Text, nullable=False)
    confidence = Column(Text, nullable=False)
    captured_photo = Column(Text, nullable=False)
    device_serial_number = Column(String, nullable=False)
    device_model = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_motion_detection(motion_data, confidence, captured_photo_path, device_serial="SNABC123", device_model="RPI3"):
    """Save motion detection data to database"""
    db = SessionLocal()
    try:
        motion_record = MotionDetection(
            motion_data=motion_data,
            confidence=confidence,
            captured_photo=captured_photo_path,
            device_serial_number=device_serial,
            device_model=device_model
        )
        db.add(motion_record)
        db.commit()
        db.refresh(motion_record)
        return motion_record
    except Exception as e:
        db.rollback()
        print(f"Error saving motion detection: {e}")
        return None
    finally:
        db.close()
