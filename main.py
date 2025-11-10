"""
FastAPI OCR Service
A professional REST API for text extraction from images using PaddleOCR
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import tempfile
import os

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from paddleocr import PaddleOCR
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ocr_service.log')
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OCR Text Extraction API",
    description="Extract text from images using PaddleOCR",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Response model
class OCRResponse(BaseModel):
    """Response model for OCR results"""
    success: bool = Field(..., description="Whether the OCR operation was successful")
    scanned_text: str = Field(..., description="Extracted text from the image")
    confidence: Optional[float] = Field(None, description="Average confidence score")
    message: Optional[str] = Field(None, description="Additional information or error message")

# Initialize PaddleOCR (singleton pattern)
ocr_instance = None

def get_ocr_instance():
    """Get or create OCR instance (lazy loading)"""
    global ocr_instance
    if ocr_instance is None:
        logger.info("Initializing PaddleOCR instance...")
        try:
            ocr_instance = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang='en'
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise
    return ocr_instance

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting OCR API service...")
    try:
        get_ocr_instance()
        logger.info("Service started successfully")
    except Exception as e:
        logger.critical(f"Failed to start service: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down OCR API service...")

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "service": "OCR Text Extraction API",
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint"""
    logger.info("Detailed health check requested")
    try:
        ocr = get_ocr_instance()
        return {
            "status": "healthy",
            "ocr_ready": ocr is not None,
            "message": "Service is operational"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "ocr_ready": False,
                "message": str(e)
            }
        )

@app.post("/ocr/extract", response_model=OCRResponse, tags=["OCR"])
async def extract_text(
    image: UploadFile = File(..., description="Image file to process (JPEG, PNG, etc.)")
):
    """
    Extract text from an uploaded image using OCR
    
    Args:
        image: Image file to process
        
    Returns:
        OCRResponse with extracted text and metadata
    """
    logger.info(f"OCR request received - File: {image.filename}, Type: {image.content_type}")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff", "application/pdf"]
    if image.content_type not in allowed_types:
        logger.warning(f"Invalid file type: {image.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    temp_file_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as temp_file:
            temp_file_path = temp_file.name
            content = await image.read()
            temp_file.write(content)
            logger.info(f"Temporary file created: {temp_file_path}")
        
        # Get OCR instance
        ocr = get_ocr_instance()
        
        # Run OCR
        logger.info(f"Running OCR on: {image.filename}")
        result = ocr.ocr(temp_file_path)
        
        if not result or not result[0]:
            logger.warning(f"No text detected in image: {image.filename}")
            return OCRResponse(
                success=True,
                scanned_text="",
                confidence=0.0,
                message="No text detected in the image"
            )
        
        # Extract text and confidence scores
        extracted_lines = []
        confidence_scores = []
        
        for line in result[0]:
            # PaddleOCR result format: [bbox, (text, confidence)]
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = text_info[0]
                    confidence = text_info[1]
                    extracted_lines.append(str(text))
                    confidence_scores.append(float(confidence))
                elif isinstance(text_info, str):
                    # Sometimes it's just text without confidence
                    extracted_lines.append(text_info)
            elif isinstance(line, str):
                # Direct text without bbox
                extracted_lines.append(line)
        
        # Combine all text
        scanned_text = "\n".join(extracted_lines)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        logger.info(f"OCR completed - Lines: {len(extracted_lines)}, Avg Confidence: {avg_confidence:.2f}")
        
        return OCRResponse(
            success=True,
            scanned_text=scanned_text,
            confidence=round(avg_confidence, 4),
            message="Text extracted successfully"
        )
        
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )
    
    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Temporary file deleted: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting OCR API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )