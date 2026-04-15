import time
import torch
from typing import TypedDict
from langgraph.graph import StateGraph, END
from transformers import BertTokenizer
import sys
import os

# Repo root (parent of src/) — stable regardless of cwd when running as a module
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the root directory to the python path so we can import our models
sys.path.append(_REPO_ROOT)
from models.bit_transformer import BitRouterModel


def _resolve_weights_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_REPO_ROOT, path))

# ==========================================
# 1. GRAPH STATE & ROUTER INITIALIZATION
# ==========================================
class AgentState(TypedDict):
    query: str
    classification: int  # 0 for Simple, 1 for Complex
    response: str

class TernaryRouterNode:
    """Loads the trained 1.58-bit model to act as the gatekeeper."""
    def __init__(self, weights_path):
        self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
        # Initialize architecture with production scaling
        self.model = BitRouterModel(
            vocab_size=self.tokenizer.vocab_size, dim=256, num_heads=4, 
            hidden_dim=1024, num_layers=4, num_classes=2, max_seq_len=64
        )
        # Load your trained weights
        resolved = _resolve_weights_path(weights_path)
        self.model.load_state_dict(torch.load(resolved, map_location="cpu", weights_only=True))
        self.model.eval()

    def process(self, state: AgentState):
        """Runs the query through the BitRouter."""
        print(f"\n[BitRouter] Analyzing query: '{state['query']}'")
        
        # Tokenize and run inference
        inputs = self.tokenizer(state['query'], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs["input_ids"])
            prediction = torch.argmax(logits, dim=1).item()
        
        state["classification"] = prediction
        print(f"[BitRouter] Classification -> {'[0] SIMPLE' if prediction == 0 else '[1] COMPLEX'}")
        return state

# ==========================================
# 2. THE ACTION NODES
# ==========================================
def local_simple_handler(state: AgentState):
    """Handles basic queries locally, instantly."""
    print("[Action Node] Executing Local CPU Handler...")
    state["response"] = f"Local Agent: I can handle this instantly! You asked about: '{state['query']}'."
    return state

def cloud_complex_handler(state: AgentState):
    """Simulates routing to an expensive API like GPT-4."""
    print("[Action Node] Routing to Cloud API (Simulated GPT-4)...")
    time.sleep(1.5) # Simulate network delay
    state["response"] = f"Cloud Agent: I have processed your complex request using heavy compute. Result for: '{state['query']}'."
    return state

# ==========================================
# 3. GRAPH ASSEMBLY & CONDITION LOGIC
# ==========================================
def decide_route(state: AgentState):
    """The conditional logic edge based on the BitRouter's output."""
    if state["classification"] == 0:
        return "local_handler"
    else:
        return "cloud_handler"

def build_graph(weights_path="bitrouter_production_weights.pt"):
    # 1. Initialize Nodes
    router = TernaryRouterNode(weights_path)
    workflow = StateGraph(AgentState)

    # 2. Add Nodes to Graph
    workflow.add_node("router", router.process)
    workflow.add_node("local_handler", local_simple_handler)
    workflow.add_node("cloud_handler", cloud_complex_handler)

    # 3. Define the Flow
    workflow.set_entry_point("router")
    
    # Conditional Edge: Router -> [Local OR Cloud]
    workflow.add_conditional_edges(
        "router", 
        decide_route,
        {"local_handler": "local_handler", "cloud_handler": "cloud_handler"}
    )
    
    # Both handlers end the process
    workflow.add_edge("local_handler", END)
    workflow.add_edge("cloud_handler", END)

    return workflow.compile()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("🚀 Initializing LangGraph BitRouter Agent...")
    app = build_graph("bitrouter_production_weights.pt")
    
    test_queries = [
        "Can you check my account balance?",
        "I need to dispute a fraudulent transaction from yesterday.",
    ]
    
    for query in test_queries:
        result = app.invoke({"query": query})
        print(f"Final Output: {result['response']}\n")
        print("-" * 50)