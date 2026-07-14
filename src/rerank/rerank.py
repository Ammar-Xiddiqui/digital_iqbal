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
        self.model.eval() # Set to evaluation mode

    def rerank(self, query: str, hybrid_results: list, top_k: int = 5):
        """
        Takes a list of dictionary results from the Hybrid Retriever and reranks them.
        """
        if not hybrid_results:
            return []

        # Prepare pairs for the Cross-Encoder: (Query, Text)
        pairs = []
        for res in hybrid_results:
            # We compare against the original text as that is what the model was trained on
            pairs.append((query, res['text']))

        # Tokenize all pairs in a single batch for speed
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Squeeze to get a 1D array of scores
            scores = outputs.logits.squeeze(-1).cpu().numpy()

        # Inject the new cross-encoder scores into the results
        for i, res in enumerate(hybrid_results):
            res['cross_encoder_score'] = float(scores[i])

        # Sort by the new score descending
        reranked_results = sorted(hybrid_results, key=lambda x: x['cross_encoder_score'], reverse=True)
        
        return reranked_results[:top_k]

# Quick sanity check block
if __name__ == "__main__":
    reranker = CrossEncoderReranker()
    print("Reranker loaded successfully and ready for inference!")