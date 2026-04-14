import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.bit_transformer import BitRouterModel

# ==========================================
# 1. SYNTHETIC DATASET & CUSTOM TOKENIZER
# ==========================================
# 0 = Simple Intent (Route Local)
# 1 = Complex Intent (Route to Cloud LLM)
TRAINING_DATA = [
    ("hello, how are you?", 0),
    ("what is the weather today?", 0),
    ("tell me a joke", 0),
    ("who is the president?", 0),
    ("can you greet the user?", 0),
    ("write a complex python script using asyncio to scrape web data", 1),
    ("explain the mathematical derivation of the navier-stokes equation", 1),
    ("build a multi-agent system using langgraph and crewai", 1),
    ("refactor this 1000 line enterprise java codebase", 1),
    ("analyze this financial report and predict future stock trends", 1),
]

class BasicTokenizer:
    """A minimal character-level tokenizer built from scratch."""
    def __init__(self, data):
        # Create a unique vocabulary of all characters used in the dataset
        chars = set("".join([text for text, _ in data]))
        chars.add("<PAD>") # Add a padding token
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab["<PAD>"]

    def encode(self, text, max_len=64):
        # Convert characters to IDs
        encoded = [self.vocab.get(ch, self.pad_id) for ch in text]
        # Truncate if too long
        encoded = encoded[:max_len]
        # Pad with <PAD> IDs if too short
        encoded += [self.pad_id] * (max_len - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)


# 2. PYTORCH DATASET PREPARATION
# ==========================================
class AgenticDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_ids = self.tokenizer.encode(text, self.max_len)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return input_ids, label_tensor


# 3. THE TRAINING LOOP
# ==========================================
def train_model():
    print("Initializing Tokenizer and Dataset...")
    tokenizer = BasicTokenizer(TRAINING_DATA)
    dataset = AgenticDataset(TRAINING_DATA, tokenizer, max_len=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize the BitRouter Model
    print("Building 1.58-bit Transformer Model...")
    model = BitRouterModel(
        vocab_size=tokenizer.vocab_size,
        dim=64,           # Small dimension for fast CPU training
        num_heads=2,      
        hidden_dim=128,   # Compressed FFN dimension
        num_layers=2,     # 2 stacked blocks
        num_classes=2,    # Simple vs. Complex
        max_seq_len=64
    )

    # Standard Classification setup
    criterion = nn.CrossEntropyLoss()
    # Notice we use AdamW. It updates the FP32 latent weights in the background!
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    epochs = 20
    print(f"\nStarting Training on CPU for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_texts, batch_labels in dataloader:
            # 1. Clear previous gradients
            optimizer.zero_grad()
            
            # 2. Forward Pass (Weights are quantized to -1, 0, 1 on the fly here)
            logits = model(batch_texts)
            
            # 3. Calculate how wrong the model is
            loss = criterion(logits, batch_labels)
            
            # 4. Backward Pass (Gradients pass straight through the quantization step)
            loss.backward()
            
            # 5. Update latent weights
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")

    # Save the trained weights
    torch.save(model.state_dict(), "bitrouter_weights.pt")
    print("\n✅ Training Complete. Model weights saved to 'bitrouter_weights.pt'.")

if __name__ == "__main__":
    train_model()