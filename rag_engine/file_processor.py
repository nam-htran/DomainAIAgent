# rag_engine/file_processor.py
import os
import fitz  # PyMuPDF
import docx
import tiktoken

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

def _parse_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def _parse_docx(file):
    doc = docx.Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

def _parse_txt(file):
    return file.read().decode("utf-8")

def parse_file(file):
    file_ext = os.path.splitext(file.name)[1].lower()
    if file_ext == ".pdf": return _parse_pdf(file)
    elif file_ext == ".docx": return _parse_docx(file)
    elif file_ext == ".txt": return _parse_txt(file)
    else: raise ValueError(f"Định dạng file không được hỗ trợ: {file_ext}")

def smart_chunk(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks