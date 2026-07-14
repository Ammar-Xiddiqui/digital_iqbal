import json
import time
from src.retrieval.hybrid_retriever import HybridRetriever
from src.graph.generate import generate_answer, REFUSAL_STRING

def evaluate_generation():
    print("Loading Hybrid Retriever and checking LLM connection...")
    retriever = HybridRetriever()
    
    eval_data = []
    with open("eval/hallucination_eval_set.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
            
    print(f"Loaded {len(eval_data)} queries (20 Adversarial, 5 Real). Running Phase 6 Benchmark...\n")
    
    adversarial_count = 0
    adversarial_refusals = 0
    
    real_count = 0
    citation_errors = 0
    total_citations_checked = 0
    
    for item in eval_data:
        query = item['query']
        is_adv = item['is_adversarial']
        
        # 1. Retrieve Context
        retrieved_verses = retriever.search(query, top_k=10)
        
        # 2. Generate Answer
        # Ensure your Ollama model name here matches what is running on your machine!
        answer, used_citations, valid_citations = generate_answer(query, retrieved_verses, ollama_model="llama3")
        
        if is_adv:
            adversarial_count += 1
            if REFUSAL_STRING in answer or not used_citations:
                adversarial_refusals += 1
            else:
                print(f"[HALLUCINATION DETECTED] Query: {query}")
                print(f"LLM Answer: {answer}\n")
        else:
            real_count += 1
            for citation in used_citations:
                total_citations_checked += 1
                if citation not in valid_citations:
                    citation_errors += 1
                    print(f"[FAKE CITATION DETECTED] LLM invented verse ID: {citation} for query: {query}")

    refusal_rate = (adversarial_refusals / adversarial_count) * 100 if adversarial_count > 0 else 0
    
    # If there are no citations at all, we can't divide by zero, but it's 100% accurate if there are no errors
    citation_accuracy = 100.0
    if total_citations_checked > 0:
        citation_accuracy = ((total_citations_checked - citation_errors) / total_citations_checked) * 100

    print("="*50)
    print(" Phase 6: Citation-Grounded Generation Benchmark")
    print("="*50)
    print(f" Adversarial Refusal Rate: {refusal_rate:.1f}% (Target: >=90%)")
    print(f" Citation Accuracy:        {citation_accuracy:.1f}% (Target: 100%)")
    print(f" Total Citations Checked:  {total_citations_checked}")
    print("="*50)
    
    if refusal_rate >= 90.0 and citation_accuracy == 100.0:
        print("\nGATE STATUS: PASS")
        print("Zero-hallucination layer verified. Ready for Phase 7 (FastAPI).")
    else:
        print("\nGATE STATUS: FAIL")
        print("The LLM hallucinated. Adjust the guardrail threshold or prompt instructions.")

if __name__ == "__main__":
    evaluate_generation()