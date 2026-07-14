import json
from src.graph.graph import query_router

def evaluate_router():
    eval_data = []
    with open("eval/routing_eval_set.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
            
    correct = 0
    
    for item in eval_data:
        # LangGraph invoke returns a dict containing the final state
        result = query_router.invoke({"query": item["query"]})
        
        if result["intent"] == item["expected_route"]:
            correct += 1
            
    accuracy = (correct / len(eval_data)) * 100
    
    print("="*40)
    print(" Phase 5: LangGraph Routing Benchmark")
    print("="*40)
    print(f" Total Queries: {len(eval_data)}")
    print(f" Routing Accuracy: {accuracy:.1f}%")
    print("="*40)
    
    if accuracy >= 85.0:
        print("\nGATE STATUS: PASS")
        print("Ready for Phase 6 (Generation Layer).")
    else:
        print("\nGATE STATUS: FAIL")
        print("Improve intent classifier logic.")

if __name__ == "__main__":
    evaluate_router()