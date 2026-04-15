import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer

# ==========================================
# 1. GPU SETUP & HUGGING FACE DATA
# ==========================================
# Check if T4 GPU is active (Go to Runtime -> Change runtime type -> T4 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

print("Loading Banking77 Dataset and Tokenizer...")
# We use a standard lightweight tokenizer
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
raw_dataset = load_dataset("banking77")

# ==========================================
# 2. REAL-WORLD INTENT MAPPING 
# ==========================================
# Banking77 has 77 classes. We map a subset to our Binary Router.
# 0 = Simple (Local LLM), 1 = Complex (Cloud LLM)
simple_intents = [0, 11, 24, 25, 40] # e.g., activate_card, get_balance
complex_intents = [3, 15, 29, 65, 70] # e.g., compromise_card, order_physical_card

# Step 2A: Filter - Only keep the rows we care about
def filter_intents(example):
    return (example['label'] in simple_intents) or (example['label'] in complex_intents)

filtered_dataset = raw_dataset.filter(filter_intents)

# Step 2B: Map - Create the new binary_label column
def create_binary_labels(example):
    if example['label'] in simple_intents:
        example['binary_label'] = 0
    else:
        example['binary_label'] = 1
    return example

mapped_dataset = filtered_dataset.map(create_binary_labels)

# Step 2C: Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = mapped_dataset.map(tokenize_function, batched=True)

# Format for PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids", "binary_label"])

# Create DataLoader
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=32)

# ==========================================
# 3. SCALED UP BIT-ROUTER ARCHITECTURE
# ==========================================
print("Initializing Scaled 1.58-bit Model...")
# Make sure BitRouterModel is defined in a cell above this!
model = BitRouterModel(
    vocab_size=tokenizer.vocab_size,
    dim=256,          # 4x larger than local prototype
    num_heads=4,      # Double the attention heads
    hidden_dim=1024,  # Massive ternary FFN
    num_layers=4,     # 4 stacked hybrid blocks
    num_classes=2,
    max_seq_len=64
).to(device) # Move entire model to GPU

criterion = nn.CrossEntropyLoss()
# Slightly lower learning rate for the scaled-up model
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# ==========================================
# 4. THE GPU TRAINING LOOP
# ==========================================
epochs = 5 
print(f"\nStarting GPU Training for {epochs} epochs...")

model.train()
for epoch in range(epochs):
    total_loss = 0.0
    
    for batch in train_dataloader:
        # Move data to GPU
        input_ids = batch["input_ids"].to(device)
        labels = batch["binary_label"].to(device)
        
        # 1. Clear gradients
        optimizer.zero_grad()
        
        # 2. Forward pass (Quantization happens dynamically here)
        logits = model(input_ids)
        
        # 3. Calculate loss
        loss = criterion(logits, labels)
        
        # 4. Backward pass (STE bypasses the non-differentiable rounding)
        loss.backward()
        
        # 5. Update latent FP32 weights
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_dataloader):.4f}")

# Save the production weights
torch.save(model.state_dict(), "bitrouter_production_weights.pt")
print("\n✅ GPU Training Complete. Weights saved as 'bitrouter_production_weights.pt'!")