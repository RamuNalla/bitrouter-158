# BitRouter-158: Edge-First LLM Routing via Ternary Quantization

🚀 **High-performance 1.58-bit (BitNet) routing engine minimizing latency and memory overhead in multi-agent LangGraph systems.**

---

## **Project Overview**

* **The Problem:** Defaulting every query to a cloud LLM (GPT-4/Claude) in agentic workflows creates severe API cost and network latency bottlenecks.
* **The Solution:** An ultra-lightweight, local CPU "Gatekeeper" model that intercepts and routes traffic before it hits the cloud.
* **The Mechanism:** Implements **BitNet b1.58** architecture. By constraining Feed-Forward weights strictly to `{-1, 0, 1}`, the model replaces expensive Float32 matrix multiplications with native addition.

---

## **Key Technical Implementation**

* **Custom Autograd & STE:** 
    * Bypassed the zero-gradient problem inherent in discrete step functions (rounding).
    * Engineered a custom PyTorch `BitLinear` module using a **Straight-Through Estimator (STE)** to flow gradients seamlessly to latent FP32 weights during training.
* **Dynamic Bit-Mixing Architecture:** 
    * **Attention Heads:** Maintained in standard FP32 precision to preserve delicate semantic reasoning.
    * **Feed-Forward Networks (FFNs):** Aggressively compressed to 1.58-bits (where the bulk of model parameters reside).
* **LangGraph Integration:** 
    * Deployed the trained 1.58-bit model as the primary conditional routing edge inside a LangGraph state machine to autonomously dictate Local vs. Cloud execution.

---

## **Mathematical Intuition: The STE Trick**

* **The Challenge:** Backpropagation fails when encountering `round()` operations due to zero derivatives.
* **Forward Pass (Quantization):** Weights are scaled dynamically and forced into ternary states:
  * `W_quant = Round(Clamp(W / gamma, -1, 1))`
* **Backward Pass (Gradient Bypass):** The PyTorch autograd engine is overridden to act as an identity function across the quantization step:
  * `grad_input = grad_output`

---

## **Hardware Benchmarking Results**

Trained on a subset of Hugging Face `Banking77`. Benchmarked locally on consumer CPU against a 1:1 dimensionally scaled standard PyTorch `nn.Linear` model.

| **Metric** | **Standard Model (FP32)** | **BitRouter (1.58-bit)** |
| :--- | :--- | :--- |
| **RAM Footprint (MB)** | 41.16 MB | **2.77 MB** |
| **Latency (ms / token)** | 0.036 ms | 0.081 ms* |

* 🏆 **System Impact:** Achieved a **93.3% memory footprint reduction**.
* ⚠️ ***Latency Tradeoff (Simulated Quantization):** The observed CPU latency increase perfectly illustrates the overhead of high-level Python frameworks. PyTorch lacks native fused kernels for 1.58-bit operations, meaning on-the-fly Python calculation overhead masks the algorithmic speedup. 
* 🚀 **Future Iteration:** Requires custom C++ or Triton kernels to bypass multiplication circuits and unlock true hardware-level acceleration.

---

## **Repository Map**

* **`models/`** * `bit_linear.py` - Core ternary quantization logic and STE autograd module.
  * `bit_transformer.py` - Hybrid 1.58-bit Transformer architecture.
* **`scripts/`**
  * `train.py` - GPU-accelerated training pipeline (Hugging Face datasets).
  * `benchmark.py` - Hardware-aware CPU latency and RAM profiler.
* **`src/agent/`** * `router.py` - LangGraph state machine and conditional logic.
* **`tests/`** * `test_bit_linear.py` - Verifies STE gradient flow and ternary assertions.
  * `test_transformer.py` - Dimension validation and forward-pass checks.

---

## **Quickstart Execution**

**1. Install Dependencies**
```bash
pip install torch transformers datasets langgraph psutil
```

**2. Run the Benchmarking Suite** *(Validates the 93% memory reduction locally)*
```bash
python -m scripts.benchmark
```

**3. Execute the LangGraph Agent Workflow** *(Tests the end-to-end routing logic)*
```bash
python -m src.agent.router
```