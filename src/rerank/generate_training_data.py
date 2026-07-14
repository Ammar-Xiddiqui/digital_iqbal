import json
import random
from pathlib import Path
from rank_bm25 import BM25Okapi

CORPUS_FILE = Path("data/processed/corpus.jsonl")
TRAIN_DATA_FILE = Path("data/processed/reranker_train.jsonl")

def generate_data():
    print("Loading corpus...")
    corpus = []
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
            
    print("Building BM25 for Hard Negatives...")
    tokenized_corpus = [verse['text_original'].split() for verse in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    training_pairs = []
    
    print("Generating Positives and Hard Negatives (Including Native Script)...")
    for i, verse in enumerate(corpus):
        if i >= 3000: 
            break
            
        original_text = verse['text_original']
        roman_text = verse.get('text_roman', '')
        english_text = verse.get('translation_en', '')
        
        # --- 1. Positives (Label 1.0) ---
        # Added native text matching to train the model on Urdu-to-Urdu similarity
        training_pairs.append({"query": original_text, "text": original_text, "label": 1.0})
        
        if roman_text:
            training_pairs.append({"query": roman_text, "text": original_text, "label": 1.0})
        if english_text:
            training_pairs.append({"query": english_text, "text": original_text, "label": 1.0})
            
        # --- 2. Hard Negatives (Label 0.0) ---
        tokenized_query = original_text.split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[1:4]
        
        for idx in top_indices:
            negative_text = corpus[idx]['text_original']
            if negative_text != original_text:
                # Add native negative
                training_pairs.append({"query": original_text, "text": negative_text, "label": 0.0})
                if roman_text:
                    training_pairs.append({"query": roman_text, "text": negative_text, "label": 0.0})
                if english_text:
                    training_pairs.append({"query": english_text, "text": negative_text, "label": 0.0})

    random.shuffle(training_pairs)
    
    print(f"\nSaving {len(training_pairs)} pairs to {TRAIN_DATA_FILE}...")
    with open(TRAIN_DATA_FILE, 'w', encoding='utf-8') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print("Training data generation complete!")

if __name__ == "__main__":
    generate_data()