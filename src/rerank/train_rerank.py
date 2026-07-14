import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm

# Paths
TRAIN_DATA_FILE = Path("data/processed/reranker_train.jsonl")
MODEL_SAVE_DIR = Path("reranker_model/")

class RerankDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Cross-Encoder format: [CLS] Query [SEP] Text [SEP]
        encoded = self.tokenizer(
            item['query'],
            item['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.float)
        }

def train():
    print("Initializing PyTorch Training...")
    # Auto-detect CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_name = "bert-base-multilingual-cased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # num_labels=1 configures BERT for regression (outputting a continuous score)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    dataset = RerankDataset(TRAIN_DATA_FILE, tokenizer)
    # Batch size 8 is a safe sweet spot for CPU training to avoid out-of-memory errors
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.MSELoss()

    # We will do 1 epoch to establish the baseline and pass the gate.
    epochs = 1 

    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.squeeze(-1)
            
            # Calculate loss and backpropagate
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Average Epoch Loss: {avg_loss:.4f}")

    print(f"\nSaving fine-tuned model to {MODEL_SAVE_DIR}...")
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print("Training complete!")

if __name__ == "__main__":
    train()