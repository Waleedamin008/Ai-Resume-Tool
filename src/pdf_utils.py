"""
PDF Utilities for Resume Text Extraction
Extracts text content from uploaded PDF resumes
"""

import pdfplumber
from io import BytesIO


def extract_text_from_pdf(pdf_file):
    """
    Extract text content from a PDF file
    
    Args:
        pdf_file: Uploaded PDF file object (from Streamlit file_uploader)
    
    Returns:
        str: Extracted text from the PDF
    
    Raises:
        Exception: If PDF cannot be read or processed
    """
    try:
        # Read PDF file
        pdf_bytes = pdf_file.read()
        pdf_stream = BytesIO(pdf_bytes)
        
        # Extract text from all pages
        text = ""
        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
