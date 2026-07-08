import json
import time
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

def evaluate():
    print("Loading FAISS index and LaBSE model...")
    index = faiss.read_index("data/processed/faiss_index.bin")
    
    with open("data/processed/id_map.json", 'r', encoding='utf-8') as f:
        id_map = json.load(f)
        
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    eval_data = []
    with open("eval/retrieval_eval_set.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
            
    print(f"Loaded {len(eval_data)} evaluation queries. Running benchmark...\n")
    
    top_5_hits = 0
    top_10_hits = 0
    latencies = []
    
    for item in eval_data:
        query = item['query']
        expected_id = item['expected_verse_id']
        
        start = time.time()
        query_vector = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, 10)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        # Map FAISS indices back to verse IDs
        retrieved_ids = [id_map[i] for i in indices[0] if i != -1]
        
        if expected_id in retrieved_ids[:5]:
            top_5_hits += 1
        if expected_id in retrieved_ids[:10]:
            top_10_hits += 1
            
    recall_5 = (top_5_hits / len(eval_data)) * 100
    recall_10 = (top_10_hits / len(eval_data)) * 100
    avg_latency = sum(latencies) / len(latencies)
    
    print("="*40)
    print(" Phase 2: Dense Retrieval Benchmark")
    print("="*40)
    print(f" Recall@5:      {recall_5:.1f}%")
    print(f" Recall@10:     {recall_10:.1f}%")
    print(f" Avg Latency:   {avg_latency:.2f} ms")
    print("="*40)
    
    if recall_10 < 50.0:
        print("\nGATE STATUS: FAIL (Recall@10 < 50%)")
        print("Error: Dense retrieval is catastrophically low. Debug embeddings.")
    else:
        print("\nGATE STATUS: PASS")
        print("Record these numbers. Ready for Phase 3 (Hybrid Retrieval).")

if __name__ == "__main__":
    evaluate()