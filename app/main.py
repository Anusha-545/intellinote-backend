from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
import os
import io
import uuid
import hashlib
import time
from groq import Groq
import bcrypt
import jwt
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Try to import PyPDF2 as alternative to PyMuPDF
try:
    import PyPDF2
    PDF_PARSER = "PyPDF2"
except ImportError:
    try:
        import fitz  # PyMuPDF
        PDF_PARSER = "PyMuPDF"
    except ImportError:
        raise ImportError("Please install either PyPDF2 or PyMuPDF: pip install PyPDF2 pymupdf")

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Intellinote Forge API",
    description="PDF Text Extraction and AI Summarization using Groq AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# MongoDB setup
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise ValueError("MONGO_URL environment variable is not set")

client = AsyncIOMotorClient(MONGO_URL)
db = client.intellinote_forge

# Collections
users_collection = db.users
documents_collection = db.documents
responses_collection = db.responses

# Groq AI setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

groq_client = Groq(api_key=GROQ_API_KEY)

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "secret123")
JWT_ALGORITHM = "HS256"

# Pydantic Models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime

class AIPrompt(BaseModel):
    text: str
    document_id: Optional[str] = None
    prompt_type: Optional[str] = "summarize"  # "summarize", "question", "analyze"

class AIResponse(BaseModel):
    response: str
    document_id: Optional[str] = None
    processing_time: float
    model_used: str

class SummaryResponse(BaseModel):
    summary: str
    key_points: List[str]
    document_id: str
    processing_time: float

# Utility Functions
async def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

async def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

async def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = await verify_token(token)
    
    user = await users_collection.find_one({"_id": payload["user_id"]})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

# PDF Text Extraction Functions
async def extract_text_with_pypdf2(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2"""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        
        return text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to extract text from PDF with PyPDF2: {str(e)}"
        )

async def extract_text_with_pymupdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF (fitz)"""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        
        pdf_document.close()
        return text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to extract text from PDF with PyMuPDF: {str(e)}"
        )

async def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF file using available library"""
    if PDF_PARSER == "PyPDF2":
        return await extract_text_with_pypdf2(pdf_bytes)
    elif PDF_PARSER == "PyMuPDF":
        return await extract_text_with_pymupdf(pdf_bytes)
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No PDF parser available"
        )

# Document Hashing for duplicate detection
async def create_document_hash(document_bytes: bytes) -> str:
    """Create a hash for document to detect duplicates"""
    return hashlib.sha256(document_bytes).hexdigest()

# AI Processing Functions
async def process_with_groq_ai(text: str, prompt_type: str = "summarize") -> str:
    """Send text to Groq AI for processing"""
    try:
        # Different prompts based on request type
        prompts = {
            "summarize": f"""Please provide a comprehensive summary of the following text. 
            Include key points, main arguments, and important details. Keep it concise but thorough.
            
            Text: {text}
            
            Summary:""",
            
            "question": f"""Based on the following text, answer any questions that might be asked about it.
            Provide clear, accurate information extracted from the text.
            
            Text: {text}
            
            Answer:""",
            
            "analyze": f"""Analyze the following text. Provide insights about:
            1. Main themes and topics
            2. Writing style and tone
            3. Key arguments or points
            4. Any notable patterns or structures
            
            Text: {text}
            
            Analysis:"""
        }
        
        prompt = prompts.get(prompt_type, prompts["summarize"])
        
        # Truncate text if too long (Groq has token limits)
        max_chars = 12000  # Adjust based on your needs
        if len(text) > max_chars:
            text = text[:max_chars] + "... [text truncated due to length]"
        
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",  # You can change model as needed
            messages=[
                {"role": "system", "content": "You are an expert document analyzer and summarizer. Provide accurate, concise, and helpful responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI processing failed: {str(e)}"
        )

async def extract_key_points_from_summary(summary: str) -> List[str]:
    """Extract key points from a summary"""
    try:
        key_points_prompt = f"""Extract 3-5 key points from the following summary. Return them as a numbered list:
        
        Summary: {summary}
        
        Key Points:"""
        
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are an expert at extracting key information from text."},
                {"role": "user", "content": key_points_prompt}
            ],
            temperature=0.5,
            max_tokens=512,
            top_p=1,
            stream=False
        )
        
        response = completion.choices[0].message.content
        
        # Parse the response into a list of key points
        lines = response.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if line and any(line.startswith(str(i)) for i in range(1, 10)):
                # Remove numbering and bullet points
                for prefix in [f"{i}.", f"{i})", f"-"]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                if line:
                    key_points.append(line)
        
        return key_points[:5] if key_points else ["No key points extracted"]
        
    except Exception as e:
        print(f"Error extracting key points: {str(e)}")
        return ["Key points extraction failed"]

# Routes

@app.get("/")
async def root():
    return {
        "message": "Welcome to Intellinote Forge API",
        "version": "1.0.0",
        "pdf_parser": PDF_PARSER,
        "endpoints": {
            "auth": ["/register", "/login"],
            "documents": ["/upload/pdf", "/documents", "/documents/{document_id}"],
            "ai": ["/ask/ai", "/summarize"]
        }
    }

# Authentication Routes
@app.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """Register a new user"""
    # Check if user already exists
    existing_user = await users_collection.find_one({
        "$or": [
            {"email": user_data.email},
            {"username": user_data.username}
        ]
    })
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )
    
    # Hash password
    hashed_password = await hash_password(user_data.password)
    
    # Create user document
    user_id = str(uuid.uuid4())
    user_doc = {
        "_id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "password": hashed_password,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    await users_collection.insert_one(user_doc)
    
    return UserResponse(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        created_at=user_doc["created_at"]
    )

@app.post("/login")
async def login(login_data: UserLogin):
    """Login user and return JWT token"""
    # Find user
    user = await users_collection.find_one({"email": login_data.email})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not await verify_password(login_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create token
    token_data = {"user_id": user["_id"], "email": user["email"]}
    access_token = await create_access_token(token_data)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["_id"],
            "username": user["username"],
            "email": user["email"]
        }
    }

# Document Routes
@app.post("/upload/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    document_name: Optional[str] = None,
    summarize: Optional[bool] = False,
    current_user: dict = Depends(get_current_user)
):
    """Upload PDF, extract text, and optionally process with AI"""
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    # Read file
    contents = await file.read()
    
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty"
        )
    
    # Create document hash
    document_hash = await create_document_hash(contents)
    
    # Check if document already exists for this user
    existing_document = await documents_collection.find_one({
        "userId": current_user["_id"],
        "documentHash": document_hash
    })
    
    document_id = str(uuid.uuid4())
    extracted_text = ""
    ai_response = None
    key_points = []
    
    if existing_document:
        # Use existing document
        document_id = existing_document["_id"]
        extracted_text = existing_document["documentInfo"][0]["documentContent"]
        is_duplicate = True
        
        # If summarize is requested, use existing text
        if summarize and extracted_text:
            start_time = time.time()
            ai_response = await process_with_groq_ai(extracted_text, "summarize")
            key_points = await extract_key_points_from_summary(ai_response)
            processing_time = time.time() - start_time
            
            # Save response
            await save_ai_response(current_user["_id"], document_id, ai_response, key_points, processing_time)
    else:
        # Extract text from PDF
        extracted_text = await extract_text_from_pdf(contents)
        
        if not extracted_text or len(extracted_text.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the PDF"
            )
        
        # Prepare document info
        doc_info = {
            "documentName": document_name or file.filename,
            "documentContent": extracted_text,
            "originalFilename": file.filename,
            "fileSize": len(contents),
            "uploadedAt": datetime.utcnow()
        }
        
        # Save to database
        document_doc = {
            "_id": document_id,
            "userId": current_user["_id"],
            "documentHash": document_hash,
            "documentInfo": [doc_info],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await documents_collection.insert_one(document_doc)
        is_duplicate = False
        
        # If summarize is requested
        if summarize:
            start_time = time.time()
            ai_response = await process_with_groq_ai(extracted_text, "summarize")
            key_points = await extract_key_points_from_summary(ai_response)
            processing_time = time.time() - start_time
            
            # Save response
            await save_ai_response(current_user["_id"], document_id, ai_response, key_points, processing_time)
    
    response_data = {
        "message": "PDF uploaded successfully",
        "document_id": document_id,
        "extracted_text_length": len(extracted_text),
        "is_duplicate": is_duplicate,
        "file_name": file.filename,
        "upload_time": datetime.utcnow().isoformat()
    }
    
    if ai_response:
        response_data.update({
            "summary": ai_response,
            "key_points": key_points,
            "has_summary": True
        })
    
    return response_data

async def save_ai_response(user_id: str, document_id: str, ai_response: str, key_points: List[str], processing_time: float):
    """Save AI response to database"""
    response_id = str(uuid.uuid4())
    response_entry = {
        "response_id": response_id,
        "document_id": document_id,
        "ai_response": ai_response,
        "key_points": key_points,
        "processing_time": processing_time,
        "timestamp": datetime.utcnow()
    }
    
    # Update responses collection
    await responses_collection.update_one(
        {"userId": user_id},
        {
            "$push": {
                "summarizations": response_entry
            },
            "$setOnInsert": {
                "userId": user_id,
                "created_at": datetime.utcnow()
            }
        },
        upsert=True
    )

@app.get("/documents")
async def get_user_documents(current_user: dict = Depends(get_current_user)):
    """Get all documents for the current user"""
    documents = await documents_collection.find(
        {"userId": current_user["_id"]}
    ).sort("created_at", -1).to_list(length=100)
    
    simplified_docs = []
    for doc in documents:
        simplified_docs.append({
            "id": doc["_id"],
            "document_name": doc["documentInfo"][0]["documentName"],
            "original_filename": doc["documentInfo"][0]["originalFilename"],
            "upload_date": doc["created_at"],
            "text_length": len(doc["documentInfo"][0]["documentContent"]),
            "file_size": doc["documentInfo"][0]["fileSize"]
        })
    
    return {"documents": simplified_docs, "count": len(simplified_docs)}

@app.get("/documents/{document_id}")
async def get_document(document_id: str, current_user: dict = Depends(get_current_user)):
    """Get specific document details"""
    document = await documents_collection.find_one({
        "_id": document_id,
        "userId": current_user["_id"]
    })
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return {
        "id": document["_id"],
        "document_info": {
            "name": document["documentInfo"][0]["documentName"],
            "original_filename": document["documentInfo"][0]["originalFilename"],
            "content_preview": document["documentInfo"][0]["documentContent"][:500] + "..." if len(document["documentInfo"][0]["documentContent"]) > 500 else document["documentInfo"][0]["documentContent"],
            "full_content_length": len(document["documentInfo"][0]["documentContent"]),
            "file_size": document["documentInfo"][0]["fileSize"],
            "upload_date": document["documentInfo"][0]["uploadedAt"]
        },
        "created_at": document["created_at"]
    }

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a document"""
    result = await documents_collection.delete_one({
        "_id": document_id,
        "userId": current_user["_id"]
    })
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Also delete related responses
    await responses_collection.update_one(
        {"userId": current_user["_id"]},
        {"$pull": {"summarizations": {"document_id": document_id}}}
    )
    
    return {"message": "Document deleted successfully"}

# AI Processing Routes
@app.post("/ask/ai", response_model=AIResponse)
async def ask_ai(
    prompt: AIPrompt,
    current_user: dict = Depends(get_current_user)
):
    """Send text to Groq AI and get response"""
    start_time = time.time()
    
    # Get text (either from prompt or from document)
    if prompt.document_id:
        # Fetch document text
        document = await documents_collection.find_one({
            "_id": prompt.document_id,
            "userId": current_user["_id"]
        })
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        text_to_process = document["documentInfo"][0]["documentContent"]
    else:
        text_to_process = prompt.text
    
    # Validate text
    if not text_to_process or len(text_to_process.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text provided for AI processing"
        )
    
    # Process with Groq AI
    ai_response = await process_with_groq_ai(text_to_process, prompt.prompt_type)
    
    processing_time = time.time() - start_time
    
    # Save response to database
    await save_ai_response(
        user_id=current_user["_id"],
        document_id=prompt.document_id,
        ai_response=ai_response,
        key_points=[],
        processing_time=processing_time
    )
    
    return AIResponse(
        response=ai_response,
        document_id=prompt.document_id,
        processing_time=round(processing_time, 2),
        model_used="openai/gpt-oss-120b"
    )

    

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_document(
    document_id: str,
    detailed: Optional[bool] = False,
    current_user: dict = Depends(get_current_user)
):
    """Specialized endpoint for document summarization"""
    start_time = time.time()
    
    # Fetch document
    document = await documents_collection.find_one({
        "_id": document_id,
        "userId": current_user["_id"]
    })
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    text_to_summarize = document["documentInfo"][0]["documentContent"]
    
    if not text_to_summarize or len(text_to_summarize.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no text content"
        )
    
    # Choose prompt based on detailed flag
    prompt_type = "analyze" if detailed else "summarize"
    
    # Get summary from AI
    summary = await process_with_groq_ai(text_to_summarize, prompt_type)
    
    # Extract key points
    key_points = await extract_key_points_from_summary(summary)
    
    processing_time = time.time() - start_time
    
    # Save to responses collection
    await save_ai_response(
        user_id=current_user["_id"],
        document_id=document_id,
        ai_response=summary,
        key_points=key_points,
        processing_time=processing_time
    )
    
    return SummaryResponse(
        summary=summary,
        key_points=key_points,
        document_id=document_id,
        processing_time=round(processing_time, 2)
    )

@app.get("/responses")
async def get_user_responses(
    document_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get all AI responses for the current user"""
    response_doc = await responses_collection.find_one(
        {"userId": current_user["_id"]}
    )
    
    if not response_doc:
        return {"responses": [], "count": 0}
    
    summarizations = response_doc.get("summarizations", [])
    
    # Filter by document_id if provided
    if document_id:
        summarizations = [s for s in summarizations if s.get("document_id") == document_id]
    
    return {
        "responses": summarizations,
        "count": len(summarizations)
    }

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check MongoDB connection
    try:
        await client.admin.command('ping')
        mongo_status = "connected"
    except Exception as e:
        mongo_status = f"disconnected: {str(e)}"
    
    # Check Groq API
    try:
        groq_client.models.list()
        groq_status = "connected"
    except Exception as e:
        groq_status = f"disconnected: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "pdf_parser": PDF_PARSER,
        "services": {
            "mongodb": mongo_status,
            "groq_ai": groq_status
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)