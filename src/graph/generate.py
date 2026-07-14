import httpx
import re

# The refusal string that the system must use if it doesn't know the answer
REFUSAL_STRING = "I cannot find a relevant verse to answer this query."

def generate_answer(query: str, retrieved_verses: list, ollama_model: str = "llama3"):
    """
    Takes a query and retrieved verses, applies a confidence guardrail, 
    and generates a strictly cited answer using a local Ollama model.
    """
    
    # 1. HARD GUARDRAIL: RRF scores are typically around 0.016 for top matches.
    # If the highest score is below 0.005, it means the retriever found garbage.
    if not retrieved_verses or retrieved_verses[0].get('rrf_score', 0) < 0.005:
        return REFUSAL_STRING, []

    # 2. Format Context & Extract Valid Citation IDs
    context_text = ""
    valid_citations = []
    
    # Only feed the top 5 verses to the LLM to keep the context window tight and focused
    for verse in retrieved_verses[:5]: 
        context_text += f"Verse ID: {verse['verse_id']} | Book: {verse['book']}\n"
        context_text += f"Original: {verse['text']}\n"
        context_text += f"Roman: {verse.get('roman', 'N/A')}\n\n"
        valid_citations.append(verse['verse_id'])

    # 3. The Anti-Hallucination Prompt
    prompt = f"""You are a strict, highly accurate assistant specializing in Allama Iqbal's poetry.
    
User Query: {query}

Retrieved Verses:
{context_text}

INSTRUCTIONS:
1. Answer the user's query using ONLY the provided retrieved verses.
2. If the retrieved verses do not contain the answer, or if the user is asking about something Iqbal never wrote about, you MUST output exactly: "{REFUSAL_STRING}". Do not attempt to guess.
3. Every single time you reference a verse, quote a verse, or explain a verse, you MUST append its exact Verse ID in brackets. Example: "In this verse, Iqbal says... [001_001_001]."
4. NEVER invent or hallucinate a Verse ID. You may only use the exact Verse IDs provided in the context above.
"""
    
    # 4. Execute LLM Call (Assuming Ollama is running locally on port 11434)
    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.0 # Strict, deterministic output
            },
            timeout=300.0
        )
        response.raise_for_status()
        answer = response.json().get("response", "")
        
        # 5. Extract the citations the LLM actually used in its response
        used_citations = re.findall(r'\[(.*?)\]', answer)
        
        return answer, used_citations, valid_citations
        
    except Exception as e:
        return f"Error contacting LLM: {str(e)}", [], []