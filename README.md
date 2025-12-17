# ASCENTo.1 🚀  
*A personal, terminal-based LLM built from scratch for AI exploration and creative experimentation.*

---

## 🌟 Highlights

- **Custom Causal LLM** – Trained from scratch (no pre-trained weights) on your own hardware.
- **Hardware Accelerated** – Fully optimized for **Apple Silicon (M1/M2/M3)** using MPS (Metal Performance Shaders).
- **Alpaca-Instruct Data** – Learns from 50k+ high-quality instruction-following examples.
- **Unified Protocol** – Standardized GPT-style architecture (Causal Masking, Concatenated Training, Tied Embeddings).
- **Interactive Console** – Train, chat, and save directly from the terminal.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **PyTorch** (with MPS support)
- **Hugging Face Datasets** (for data acquisition)

---

## ⚙️ Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/codycherrington/ASCENTo.1.git
cd ASCENTo.1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data & Vocab

This will download the Alpaca dataset and generate a custom tokenizer vocabulary (~32k tokens).

```bash
# Download and format data
python3 data_loader.py

# Build the tokenizer
python3 build_tokenizer.py
```

### 4. Launch Ascent

```bash
python3 model.py
```

- Select **[1] Train** to start the training loop.
- Select **[2] Chat** to test the model's responses.

### Training Notes (Apple Silicon)
The model is pre-configured for **8GB RAM** MacBooks:
- **Batch Size:** 1 (with Gradient Accumulation = 32)
- **Sequence Length:** 256 tokens
- **Swap Strategy:** Uses `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to safely utilize SSD swap memory without crashing.

---

## 📂 Project Structure

```
ASCENTo.1/
├── ascent_data/                   # Training data and vocab files
│   ├── alpaca_data.json          # Main instruction dataset (Ignored in git)
│   ├── identity.json             # Persona definition
│   ├── vocab.json                # Tokenizer vocabulary
│   └── ...
├── model.py                      # Core LLM implementation (Training & Inference)
├── data_loader.py                # Downloads Alpaca from Hugging Face
├── build_tokenizer.py            # Generates vocabulary from JSON data
├── training_logs/                # Live loss/perplexity logs
├── requirements.txt              # Dependency list
└── README.md                     # This file
```

---

## 📝 License

**MIT License** - Free to use, modify, and learn from.
Copyright (c) 2025 Cody Cherrington
