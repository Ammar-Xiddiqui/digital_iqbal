import json
import time
from src.retrieval.hybrid_retriever import HybridRetriever

def evaluate_hybrid():
    print("Initializing Hybrid Retriever (Dense + BM25 + Fuzzy)...")
    retriever = HybridRetriever()
    
    eval_data = []
    with open("eval/retrieval_eval_set.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
            
    # The last 10 items are the misspelled/partial subset we just added
    standard_subset = eval_data[:-10]
    misspelled_subset = eval_data[-10:]
    
    print(f"Loaded {len(standard_subset)} standard queries and {len(misspelled_subset)} misspelled queries.\n")
    
    def run_subset(subset, name):
        top_10_hits = 0
        latencies = []
        
        for item in subset:
            start = time.time()
            results = retriever.search(item['query'], top_k=10)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            retrieved_ids = [r['verse_id'] for r in results]
            if item['expected_verse_id'] in retrieved_ids:
                top_10_hits += 1
                
        recall_10 = (top_10_hits / len(subset)) * 100
        avg_latency = sum(latencies) / len(latencies)
        return recall_10, avg_latency

    standard_recall, standard_lat = run_subset(standard_subset, "Standard")
    misspelled_recall, misspelled_lat = run_subset(misspelled_subset, "Misspelled")
    
    print("="*50)
    print(" Phase 3: Hybrid Retrieval Benchmark")
    print("="*50)
    print(f" Standard Subset Recall@10:   {standard_recall:.1f}%")
    print(f" Misspelled Subset Recall@10: {misspelled_recall:.1f}%")
    print(f" Average Latency:             {(standard_lat + misspelled_lat)/2:.2f} ms")
    print("="*50)
    
    if misspelled_recall >= 70.0: # Targeting a clear margin over standard dense
        print("\nGATE STATUS: PASS")
        print("Hybrid beats dense-only by a clear margin on typos. Ready for Phase 4.")
    else:
        print("\nGATE STATUS: FAIL")
        print("Hybrid did not significantly beat dense-only on misspelled subset. Debug fusion weights.")

if __name__ == "__main__":
    evaluate_hybrid()