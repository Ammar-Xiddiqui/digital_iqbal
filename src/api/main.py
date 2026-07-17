from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
from pathlib import Path

# Import your working pipeline modules
from src.graph.graph import query_router
from src.retrieval.hybrid_retriever import HybridRetriever
from src.graph.generate import generate_answer

# Define strict input validation
class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="The natural language search query")
    model: Optional[str] = Field("llama3", description="The local Ollama model identifier to target")

# Define strict output contracts
class CitationVerse(BaseModel):
    verse_id: str
    book: str
    text_original: str
    text_roman: Optional[str] = None
    rrf_score: float

class AskResponse(BaseModel):
    query: str
    intent: str
    answer: str
    latency_ms: float
    citations: List[CitationVerse]

# Initialize FastAPI App
app = FastAPI(
    title="Digital Iqbal API",
    description="Production REST API for the Allama Iqbal RAG Poetry Copilot Pipeline",
    version="1.0.0"
)

# ADDED: CORS Middleware to allow the frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global holder for components to optimize cold-starts
retriever = None

@app.on_event("startup")
def startup_event():
    global retriever
    print("Pre-loading Hybrid Retriever Indices into RAM...")
    retriever = HybridRetriever()
    print("API Layer Initialization Complete.")

# ADDED: Serve the Single-Page Frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the frontend UI directly from the backend server."""
    html_file = Path("src/frontend/index.html")
    if not html_file.exists():
        return "<h1>Frontend not found. Please create src/frontend/index.html</h1>"
    return html_file.read_text(encoding="utf-8")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """System health check endpoint for monitoring status."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "retriever_loaded": retriever is not None
    }

@app.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_pipeline(payload: AskRequest):
    """
    Core RAG Endpoint. Executes Routing -> Hybrid Retrieval -> Citation Grounded Generation.
    """
    start_time = time.time()
    
    try:
        # 1. Classify Intent via LangGraph Router
        routing_state = query_router.invoke({"query": payload.query})
        intent = routing_state.get("intent", "literal_lookup")

        # 2. Retrieve Relevant Verses using the pre-loaded Hybrid Retriever
        # Fetch 10 candidates to ensure context availability
        retrieved_results = retriever.search(payload.query, top_k=10)

        # 3. Generate Answer via Grounded Generation Layer
        answer, used_citations, valid_citations = generate_answer(
            query=payload.query,
            retrieved_verses=retrieved_results,
            ollama_model=payload.model
        )

        # 4. Map and Filter Citations for the Response Schema
        # Only return items that the LLM explicitly checked or used
        matched_citations = []
        for v in retrieved_results[:5]:  # Limit response payload size
            if v['verse_id'] in valid_citations:
                matched_citations.append(
                    CitationVerse(
                        verse_id=v['verse_id'],
                        book=v['book'],
                        text_original=v['text'],
                        text_roman=v.get('roman'),
                        rrf_score=v.get('rrf_score', 0.0)
                    )
                )

        latency_ms = (time.time() - start_time) * 1000

        return AskResponse(
            query=payload.query,
            intent=intent,
            answer=answer,
            latency_ms=round(latency_ms, 2),
            citations=matched_citations
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline Processing Failure: {str(e)}"
        )