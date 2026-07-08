import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths
CORPUS_FILE = Path("data/processed/corpus.jsonl")
INDEX_DIR = Path("data/processed")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
ID_MAP_FILE = INDEX_DIR / "id_map.json"

def build_dense_index():
    print("Loading corpus...")
    corpus = []
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
            
    print(f"Loaded {len(corpus)} verses.")
    
    print("Loading SentenceTransformer model (LaBSE)...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    # Prepare texts for embedding. Combining original text and English translation 
    # creates a richer semantic vector for cross-lingual queries.
    texts_to_embed = []
    verse_ids = []
    
    for verse in corpus:
        text = verse['text_original']
        if verse['translation_en']:
            text += " " + verse['translation_en']
            
        texts_to_embed.append(text)
        verse_ids.append(verse['verse_id'])
        
    print("Encoding verses (This will take some time on the CPU)...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize vectors for Cosine Similarity (IndexFlatIP)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    
    print("Building FAISS index...")
    index.add(embeddings)
    
    print(f"Writing index to {INDEX_FILE}...")
    faiss.write_index(index, str(INDEX_FILE))
    
    print(f"Writing ID map to {ID_MAP_FILE}...")
    with open(ID_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(verse_ids, f)
        
    print("Phase 2 Indexing complete!")

if __name__ == "__main__":
    build_dense_index()