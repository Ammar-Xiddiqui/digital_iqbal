import json
import time
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rerank.rerank import CrossEncoderReranker

def evaluate_reranker():
    print("Initializing Full Search Stack...")
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()
    
    eval_data = []
    with open("eval/retrieval_eval_set.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
            
    print(f"\nLoaded {len(eval_data)} queries. Running Phase 4 Benchmark...")
    
    hybrid_top3_hits = 0
    reranked_top3_hits = 0
    total_latencies = []
    
    for item in eval_data:
        query = item['query']
        expected_id = item['expected_verse_id']
        
        start = time.time()
        
        # 1. Broad fetch (Get top 20 to give the reranker enough options)
        hybrid_results = retriever.search(query, top_k=20)
        
        # Calculate Hybrid-only Precision@3 (did it naturally land in the top 3?)
        hybrid_ids = [r['verse_id'] for r in hybrid_results[:3]]
        if expected_id in hybrid_ids:
            hybrid_top3_hits += 1
            
        # 2. Sniper focus (Rerank the top 20 down to top 3)
        reranked_results = reranker.rerank(query, hybrid_results, top_k=3)
        
        latency = (time.time() - start) * 1000
        total_latencies.append(latency)
        
        # Calculate Reranked Precision@3
        reranked_ids = [r['verse_id'] for r in reranked_results]
        if expected_id in reranked_ids:
            reranked_top3_hits += 1
            
    hybrid_precision = (hybrid_top3_hits / len(eval_data)) * 100
    reranked_precision = (reranked_top3_hits / len(eval_data)) * 100
    avg_latency = sum(total_latencies) / len(total_latencies)
    
    precision_lift = reranked_precision - hybrid_precision
    
    print("\n" + "="*50)
    print(" Phase 4: Cross-Encoder Reranker Benchmark")
    print("="*50)
    print(f" Hybrid-Only Precision@3: {hybrid_precision:.1f}%")
    print(f" Reranked Precision@3:    {reranked_precision:.1f}%")
    print(f" Precision Lift:          +{precision_lift:.1f} points")
    print(f" Full Pipeline Latency:   {avg_latency:.2f} ms")
    print("="*50)
    
    if precision_lift > 0:
        print("\nGATE STATUS: PASS")
        print("The reranker successfully improved precision. Ready for Phase 5.")
    else:
        print("\nGATE STATUS: FAIL")
        print("The reranker did not improve precision over the hybrid baseline.")

if __name__ == "__main__":
    evaluate_reranker()