import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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