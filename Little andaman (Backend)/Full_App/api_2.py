from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import asyncio
import httpx
import textwrap

# ---- FastAPI setup ----
app = FastAPI(title="Offline RAG API (Chroma + Qwen via Ollama)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class PromptRequest(BaseModel):
    prompt: str

# Generation and retrieval parameters (backend-controlled)
TEMPERATURE = 0.0
TOP_P = 0.9
REPEAT_PENALTY = 1.1
MAX_TOKENS = 500
TOP_K = 10# <--- Fixed top_k value for similarity search

# Globals
vector_store = None
MAX_CONTEXT_LENGTH = 8000  # Maximum context length for the model 

@app.on_event("startup")
async def load_components():
    global vector_store

    print("[Startup] Loading embeddings & vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma(
        persist_directory="chroma_store",
        embedding_function=embeddings
    )
    print("[Startup] Vector store ready.")

@app.post("/generate", response_model=Dict[str, object])
async def generate_response(request: PromptRequest):
    try:

        

        # Retrieve relevant documents with scores
        docs = await asyncio.to_thread(
            vector_store.similarity_search_with_score, request.prompt, TOP_K
        )

        if not docs:
            return {
                "response": "Sorry, I cannot answer that based on the available documents.",
                "context_sources": []
            }

        # Concatenate context
        context = "\n\n".join([doc[0].page_content for doc in docs])
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH]

        prompt = textwrap.dedent(f"""
            You are an expert on Little Andaman Island.

            Always assume the user is asking about Little Andaman, even if it's not explicitly mentioned. Answer every question based only on the provided context.

            If the user says something casual or a greeting (like hello, hi, namaste, good morning, thanks, okay), respond briefly and politely.

            Examples:
            - "hello" → "Hello! How can I help you?"
            - "namaste" or "Namaste" → "Namaste"
            - "good morning" → "Good morning!"
            - "thank you" → "You're welcome!"
            - "okay" or "ok" → "Alright!"


            If the question asks about the existence of something (e.g., starts with “Is there”, “Are there”, “Do we have”), begin your answer with “Yes,” or “No,”:
            - If **Yes**, briefly provide relevant details (name, location, service, etc.).
            - If **No**, respond simply (e.g., “No, it does not exist.”).

            Be clear, direct, and grounded in the context.

            If a similar or repeated question has already been answered, ensure the same correct information is given again, even if the phrasing is different. Do not contradict earlier responses.
            If the question asks about the existence of something (e.g., starts with “Is there”, “Are there”, “Do we have”), begin your answer with “Yes, there is...” or “No, there is not...” and repeat part of the question before giving a short explanation.

            For all other questions:
            - Answer clearly in one or two crisp lines.
            - Do not repeat unnecessary details or list multiple places unless explicitly asked.
            - Be detailed only when the question is specific.
            - If multiple questions are asked, answer all of them.                  

            If the answer is not found in the context, say:
            **“Sorry, I cannot answer that based on the available documents.”**

            Do not make up information.  
            Do not show your reasoning.  
            Be clear, direct, and grounded in the context.


            [Context Starts]
            {context}
            [Context Ends]

            Question: {request.prompt}

            Answer:
        """)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3:0.6b",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "repeat_penalty": REPEAT_PENALTY,
                    "max_tokens": MAX_TOKENS
                },
                timeout=120.0
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Ollama model error.")

        result = response.json()
        response_text = result.get("response", "").strip()

        return {
            "response": response_text,
            "context_sources": [
                {
                    "source": doc[0].metadata.get("source", "unknown"),
                    "text": doc[0].page_content,
                    "score": doc[1]
                }
                for doc in docs
            ]
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Ollama: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy" if vector_store else "unhealthy"}