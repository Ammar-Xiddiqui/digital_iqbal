import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

MODEL_DIR = Path("reranker_model/")

class CrossEncoderReranker:
    def __init__(self):
        print(f"Loading Fine-Tuned Reranker from {MODEL_DIR}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, hybrid_results: list, top_k: int = 5):
        if not hybrid_results:
            return []

        # FIX: Separate lists for queries and texts so BERT generates token_type_ids properly
        queries = [query] * len(hybrid_results)
        texts = [res['text'] for res in hybrid_results]

        encoded = self.tokenizer(
            queries,
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            scores = outputs.logits.squeeze(-1).cpu().numpy()

        for i, res in enumerate(hybrid_results):
            # Fallback for single-item batches where numpy returns a scalar
            res['cross_encoder_score'] = float(scores[i] if scores.ndim > 0 else scores)

        reranked_results = sorted(hybrid_results, key=lambda x: x['cross_encoder_score'], reverse=True)
        return reranked_results[:top_k]