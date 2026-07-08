import json
import faiss
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_FILE = Path("data/processed/faiss_index.bin")
ID_MAP_FILE = Path("data/processed/id_map.json")
CORPUS_FILE = Path("data/processed/corpus.jsonl")

class DenseRetriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_FILE))
        
        with open(ID_MAP_FILE, 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)
            
        self.corpus_data = {}
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                verse = json.loads(line)
                self.corpus_data[verse['verse_id']] = verse
                
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        
    def search(self, query: str, top_k: int = 5):
        start_time = time.time()
        
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for j, i in enumerate(indices[0]):
            if i != -1:
                verse_id = self.id_map[i]
                verse_info = self.corpus_data[verse_id]
                results.append({
                    "verse_id": verse_id,
                    "score": float(distances[0][j]),
                    "text": verse_info['text_original'],
                    "translation": verse_info.get('translation_en', ''),
                    "book": verse_info['book']
                })
                
        latency = (time.time() - start_time) * 1000
        return results, latency

if __name__ == "__main__":
    print("Initializing Dense Retriever...")
    retriever = DenseRetriever()
    print("Ready.")
    
    while True:
        q = input("\nEnter query (or 'quit' to exit): ")
        if q.lower() == 'quit':
            break
            
        results, latency = retriever.search(q, top_k=5)
        print(f"\nResults (Latency: {latency:.2f}ms):")
        for r in results:
            print(f"[{r['score']:.4f}] {r['book']} | {r['verse_id']}")
            print(f"{r['text']}")
            print("-" * 50)