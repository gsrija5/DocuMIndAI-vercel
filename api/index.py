import os
import math
import time

import fitz
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types


# --------------------------------------------------
# 1. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(title="DocuMind AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 2. Configure Gemini
# --------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

client = None

if api_key:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120000),
    )

GENERATION_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"


def get_client():
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on Vercel.",
        )
    return client


# --------------------------------------------------
# 3. Extract text from PDF
# --------------------------------------------------

async def extract_pdf_text(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )

    if len(pdf_bytes) > 4 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="For this prototype, upload a PDF smaller than 4 MB.",
        )

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.page_count == 0:
                raise HTTPException(
                    status_code=400,
                    detail="The PDF contains no pages.",
                )

            pages = [
                page.get_text().strip()
                for page in document
            ]

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF. Please check that it is valid.",
        ) from error

    text = "\n\n".join(
        f"Page {index + 1}:\n{page}"
        for index, page in enumerate(pages)
        if page
    )

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No selectable text was found. "
                "This prototype does not yet support scanned PDFs requiring OCR."
            ),
        )

    return text, len(pages)


# --------------------------------------------------
# 4. Split document into overlapping chunks
# --------------------------------------------------

def split_into_chunks(text, chunk_size=700, overlap=100):
    words = text.split()

    if not words:
        return []

    step = chunk_size - overlap

    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
    ]


# --------------------------------------------------
# 5. Retry temporary Gemini errors
# --------------------------------------------------

def call_with_retries(function, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return function()

        except Exception as error:
            message = str(error).upper()

            if any(marker in message for marker in [
                "QUOTA EXCEEDED",
                "GENERATE_CONTENT_FREE_TIER_REQUESTS",
                "EXCEEDED YOUR CURRENT QUOTA",
            ]):
                raise HTTPException(
                    status_code=429,
                    detail="Gemini quota exceeded. Please try again later.",
                ) from error

            retryable = any(marker in message for marker in [
                "503",
                "UNAVAILABLE",
                "429",
                "RESOURCE_EXHAUSTED",
                "500",
                "502",
                "504",
                "INTERNAL",
                "TIMEOUT",
            ])

            if not retryable or attempt == max_attempts - 1:
                raise

            time.sleep(2 ** (attempt + 1))


# --------------------------------------------------
# 6. Generate content with a fallback model
# --------------------------------------------------

def generate_with_fallback(prompt):
    gemini = get_client()
    models_to_try = [GENERATION_MODEL, FALLBACK_MODEL]
    last_error = None

    for model in models_to_try:
        try:
            return call_with_retries(
                lambda model=model: gemini.models.generate_content(
                    model=model,
                    contents=prompt,
                ),
                max_attempts=2,
            )

        except HTTPException as error:
            if error.status_code == 429:
                raise
            last_error = error

        except Exception as error:
            last_error = error

    raise HTTPException(
        status_code=502,
        detail=(
            "Gemini models are temporarily unavailable or unsupported. "
            "Please try again shortly. "
            f"{str(last_error)[:180]}"
        ),
    )


# --------------------------------------------------
# 7. Create embeddings
# --------------------------------------------------

def create_embeddings(texts):
    if not texts:
        return []

    vectors = []
    gemini = get_client()

    for start in range(0, len(texts), 20):
        batch = texts[start:start + 20]

        response = call_with_retries(
            lambda batch=batch: gemini.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
            )
        )

        if not response.embeddings:
            raise HTTPException(
                status_code=502,
                detail="Gemini did not return document embeddings.",
            )

        vectors.extend(
            embedding.values
            for embedding in response.embeddings
        )

    return vectors


# --------------------------------------------------
# 8. Calculate cosine similarity
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    dot = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if not magnitude_a or not magnitude_b:
        return 0.0

    return dot / (magnitude_a * magnitude_b)


# --------------------------------------------------
# 9. Retrieve relevant chunks
# --------------------------------------------------

def retrieve_relevant_chunks(question, chunks, vectors, top_k=3):
    gemini = get_client()

    response = call_with_retries(
        lambda: gemini.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question,
        )
    )

    if not response.embeddings:
        raise HTTPException(
            status_code=502,
            detail="Gemini could not create a question embedding.",
        )

    question_vector = response.embeddings[0].values

    scored_chunks = []

    for index, vector in enumerate(vectors):
        score = cosine_similarity(question_vector, vector)
        scored_chunks.append((score, chunks[index]))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    return [
        chunk
        for _, chunk in scored_chunks[:top_k]
    ]


# --------------------------------------------------
# 10. Health check
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": "DocuMind AI",
        "gemini_configured": bool(api_key),
    }


# --------------------------------------------------
# 11. Summarize a PDF
# --------------------------------------------------

@app.post("/api/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    text, page_count = await extract_pdf_text(file)

    document_text = text[:60000]

    prompt = f"""
You are DocuMind AI, a helpful document assistant.

Summarize the following document in clear, simple language.

Include:
1. Main idea
2. Important points
3. Key findings or conclusions
4. Important terms, if any

Do not invent facts. If the provided text is incomplete, say so.

DOCUMENT:
{document_text}
"""

    try:
        response = generate_with_fallback(prompt)

        return {
            "filename": file.filename,
            "pages": page_count,
            "summary": response.text or "No summary was returned.",
            "truncated": len(text) > 60000,
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Summary generation failed: {str(error)[:250]}",
        ) from error


# --------------------------------------------------
# 12. Ask a question about a PDF
# --------------------------------------------------

@app.post("/api/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    text, page_count = await extract_pdf_text(file)
    chunks = split_into_chunks(text)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No text is available to answer questions.",
        )

    try:
        # Keep this prototype within reasonable processing limits.
        chunks = chunks[:100]
        vectors = create_embeddings(chunks)

        relevant_chunks = retrieve_relevant_chunks(
            question,
            chunks,
            vectors,
            top_k=3,
        )

        context = "\n\n---\n\n".join(relevant_chunks)

        prompt = f"""
You are DocuMind AI, a document question-answering assistant.

Answer the user's question using the supplied document excerpts.

Rules:
- Use the document excerpts as your primary source.
- Do not invent information.
- If the excerpts do not contain the answer, say that the answer
  could not be found in the retrieved parts of the document.
- Explain the answer clearly and simply.

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}
"""

        response = generate_with_fallback(prompt)

        return {
            "filename": file.filename,
            "pages": page_count,
            "question": question,
            "answer": response.text or "No answer was returned.",
            "sources": relevant_chunks,
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Question answering failed: {str(error)[:250]}",
        ) from error
