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
    
    print("Generating Positives and Hard Negatives...")
    for i, verse in enumerate(corpus):
        # We don't need to train on the entire 11k corpus, a subset of 3000 is plenty
        if i >= 3000: 
            break
            
        original_text = verse['text_original']
        roman_text = verse.get('text_roman', '')
        english_text = verse.get('translation_en', '')
        
        # --- 1. Generate Positives (Label 1.0) ---
        if roman_text:
            training_pairs.append({"query": roman_text, "text": original_text, "label": 1.0})
        if english_text:
            training_pairs.append({"query": english_text, "text": original_text, "label": 1.0})
            
        # --- 2. Generate Hard Negatives (Label 0.0) ---
        # Find structurally similar but incorrect verses using BM25
        if roman_text or english_text:
            tokenized_query = original_text.split()
            scores = bm25.get_scores(tokenized_query)
            
            # Get top BM25 matches, skip the first one (which is the actual correct verse)
            top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[1:4]
            
            for idx in top_indices:
                negative_text = corpus[idx]['text_original']
                if negative_text != original_text:
                    if roman_text:
                        training_pairs.append({"query": roman_text, "text": negative_text, "label": 0.0})
                    if english_text:
                        training_pairs.append({"query": english_text, "text": negative_text, "label": 0.0})

    # Shuffle the dataset to prevent the model from memorizing patterns
    random.shuffle(training_pairs)
    
    # Validation check to ensure balanced classes
    positives = sum(1 for p in training_pairs if p['label'] == 1.0)
    negatives = sum(1 for p in training_pairs if p['label'] == 0.0)
    
    print(f"\nGenerated {len(training_pairs)} total training pairs.")
    print(f" -> Positives (1.0): {positives}")
    print(f" -> Negatives (0.0): {negatives}")
    
    print(f"\nSaving to {TRAIN_DATA_FILE}...")
    with open(TRAIN_DATA_FILE, 'w', encoding='utf-8') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print("Training data generation complete!")

if __name__ == "__main__":
    generate_data()