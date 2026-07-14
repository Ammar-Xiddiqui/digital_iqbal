import json
from fastapi.testclient import TestClient
from src.api.main import app

def test_production_endpoints():
    print("========================================")
    print(" Running Phase 7 Integration Tests")
    print("========================================")

    # The 'with' statement forces FastAPI to run @app.on_event("startup")
    with TestClient(app) as client:
        
        # Test 1: Health Diagnostic
        print("Testing GET /health...")
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        print(f"-> Health check OK: {health_resp.json()}\n")

        # Test 2: Valid Live RAG Execution
        print("Testing POST /ask with live query...")
        payload = {"query": "khudi ko kar buland", "model": "llama3"}
        ask_resp = client.post("/ask", json=payload)
        
        # If it fails, print the actual error from the API so we can debug
        if ask_resp.status_code != 200:
            print(f"API Error Response: {ask_resp.text}")
            
        assert ask_resp.status_code == 200
        data = ask_resp.json()
        
        assert "query" in data
        assert "intent" in data
        assert "answer" in data
        assert "latency_ms" in data
        assert isinstance(data["citations"], list)
        
        print("-> RAG Response Structure Validated Successfully!")
        print(f"   Intent Routed: {data['intent']}")
        print(f"   Latency Tracked: {data['latency_ms']} ms\n")

        print("GATE STATUS: PASS")
        print("API Layer successfully wrapped and verified.")

if __name__ == "__main__":
    test_production_endpoints()