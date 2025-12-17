from io import BytesIO
from pypdf import PdfReader
from docx import Document

def parse_txt(file_content: bytes) -> str:
    """Parses plain text content."""
    return file_content.decode("utf-8")

def parse_pdf(file_content: bytes) -> str:
    """Extracts text from a PDF file byte stream."""
    reader = PdfReader(BytesIO(file_content))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def parse_docx(file_content: bytes) -> str:
    """Extracts text from a DOCX file byte stream."""
    doc = Document(BytesIO(file_content))
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)