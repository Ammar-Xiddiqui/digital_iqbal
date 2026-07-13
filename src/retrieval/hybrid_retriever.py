import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz

INDEX_FILE = Path("data/processed/faiss_index.bin")
ID_MAP_FILE = Path("data/processed/id_map.json")
CORPUS_FILE = Path("data/processed/corpus.jsonl")

class HybridRetriever:
    def __init__(self):
        # 1. Load Dense Index
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(ID_MAP_FILE, 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        
        # 2. Load Corpus and Build Lexical/Fuzzy Indexes
        self.corpus_data = []
        self.roman_texts = []
        tokenized_corpus = []
        
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                verse = json.loads(line)
                self.corpus_data.append(verse)
                roman = verse.get('text_roman', '')
                self.roman_texts.append(roman)
                
                # BM25 Tokenization (combines original Urdu/Persian and Roman)
                combined_text = verse['text_original'] + " " + roman
                tokenized_corpus.append(combined_text.lower().split())
                
        self.bm25 = BM25Okapi(tokenized_corpus)
        # Create a fast lookup dict
        self.corpus_lookup = {v['verse_id']: v for v in self.corpus_data}

    def search(self, query: str, top_k: int = 10, rrf_k: int = 60):
        # --- A. Dense Search ---
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        _, dense_indices = self.index.search(query_vector, top_k * 2)
        
        dense_ranks = {}
        for rank, idx in enumerate(dense_indices[0]):
            if idx != -1:
                verse_id = self.id_map[idx]
                dense_ranks[verse_id] = rank + 1

        # --- B. Lexical Search (BM25) ---
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        bm25_ranks = {}
        for rank, idx in enumerate(bm25_top_indices):
            if bm25_scores[idx] > 0: 
                verse_id = self.corpus_data[idx]['verse_id']
                bm25_ranks[verse_id] = rank + 1

        # --- C. Fuzzy Search (Rapidfuzz) ---
        fuzzy_results = process.extract(
            query, self.roman_texts, scorer=fuzz.WRatio, limit=top_k * 2
        )
        
        fuzzy_ranks = {}
        for rank, (match_str, score, idx) in enumerate(fuzzy_results):
            if score > 55.0: # Minimum confidence threshold for typos
                verse_id = self.corpus_data[idx]['verse_id']
                fuzzy_ranks[verse_id] = rank + 1

        # --- D. Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        all_ids = set(list(dense_ranks.keys()) + list(bm25_ranks.keys()) + list(fuzzy_ranks.keys()))
        
        for vid in all_ids:
            score = 0.0
            if vid in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[vid])
            if vid in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[vid])
            if vid in fuzzy_ranks:
                score += 1.0 / (rrf_k + fuzzy_ranks[vid])
            rrf_scores[vid] = score
            
        # Sort by final RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        
        # Format output
        return [{
            "verse_id": vid,
            "rrf_score": score,
            "text": self.corpus_lookup[vid]['text_original'],
            "roman": self.corpus_lookup[vid].get('text_roman', ''),
            "book": self.corpus_lookup[vid]['book']
        } for vid, score in sorted_results]