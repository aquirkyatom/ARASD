```markdown
# Coursework Timeline: Phase 0 & Next Steps

## Current Status: Phase 0 (Foundation & Literature Deep-Dive)
You are currently laying the groundwork for your project, which includes:
- Setting up your development and experimentation environment.
- Loading baseline models (e.g., LLaMA-3.1 8B and 70B).
- Building your retrieval index (e.g., FAISS over the Natural Questions corpus).
- Running initial baseline benchmarks.

---

## Moving on to Week 2: Literature Consolidation & Method Formalization
If you are ready to transition into the next phase, here are the key tasks to tackle:
1. **Mathematical Formulation:** Draft the mathematical formulation for the $q_{\text{RASD}}$ draft distribution.
2. **Algorithm Design:** Write out the formal algorithm pseudocode.
3. **Related Work:** Structure the Related Work section to clearly differentiate RASD from existing methods like *Speculative RAG* and *Online Speculative Decoding*.

What specific part of your project would you like to tackle next?



Week 1: Set up standard speculative decoding pipeline using existing implementations (e.g., HuggingFace assisted_generation or Medusa codebase); build FAISS index over a target corpus (e.g., Wikipedia, domain docs)

Week 2: Modify the drafter to prepend retrieved context chunks (top-3 passages from ColBERT/FAISS) to its input before generating speculative tokens; implement a trigger mechanism that retrieves every N draft tokens

Week 3: Tune acceptance rate by experimenting with retrieval frequency, chunk size, and temperature alignment between drafter and verifier; implement adaptive retrieval (only retrieve when drafter confidence is low)

Week 4: Benchmark on knowledge-intensive tasks (Natural Questions, TriviaQA, FEVER) measuring tokens/second, acceptance rate, and factual accuracy vs. vanilla speculative decoding and vanilla RAG
```

  

````markdown
# load models

# Hardware Configuration Strategy for RASD with 12GB + 50GB Shared GPU Memory

Given your hardware constraints (12 GB dedicated + 50 GB shared GPU memory), you have **approximately 62 GB total GPU memory available**, which is actually quite feasible for RASD. Here's a tailored strategy:

---

## Recommended Setup

### Primary Configuration: Use the 50GB Shared Memory Pool
```
Total available: ~62 GB GPU memory
├── LLaMA-3.1 70B (4-bit, NF4 quantization): ~35–40 GB
├── LLaMA-3.1 8B (4-bit, NF4 quantization): ~5–6 GB
├── FAISS index (NQ corpus, FP16): ~5 GB
├── Activation buffers + batch processing: ~8–10 GB
└── Safety margin: ~2–5 GB
```

**This is tight but workable.** You'll need to:
1. **Use the shared memory as primary** (it's the larger pool)
2. **Offload to CPU RAM** when not actively computing
3. **Reduce batch size** to 1–2 samples during inference
4. **Use gradient checkpointing** if fine-tuning (not needed for inference-only RASD)

---

## Fallback Configuration: If 70B Doesn't Fit

If you hit OOM errors, immediately switch to:

```
LLaMA-3.1 8B (verifier, 4-bit): ~5–6 GB
LLaMA-3.1 1B (drafter, FP16): ~2 GB
FAISS index (NQ corpus): ~5 GB
Free headroom: ~50 GB
```

This is **highly conservative and will definitely work** but uses smaller models. Document this clearly in your report's Future Work section [1]: "We evaluated on the smaller 8B+1B configuration due to GPU memory constraints of the available compute cluster. Testing on the intended 70B+8B configuration would require access to A100 (80GB) instances, which was unavailable during the coursework timeline."

---

## Setup Steps for Your Hardware

### Step 1 — Verify Memory Availability
```python
# Run this in a Jupyter cell to check actual GPU memory
import torch
print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Monitor during execution
from torch.cuda import memory_summary
print(memory_summary())
```

### Step 2 — Configure for 50GB Shared Memory
```python
# Set PyTorch to use the full available GPU memory pool
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# If using a multi-GPU setup, pin to the larger GPU
import torch
torch.cuda.set_device(0)  # Adjust if needed
```

### Step 3 — Load Models with CPU Offloading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# LLaMA-3.1 70B with 4-bit quantization + device_map for auto-offloading
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

verifier = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ),
    device_map="auto",  # Automatically offloads to CPU RAM if needed
    max_memory={0: "48GB", "cpu": "20GB"}  # Allocate 48GB GPU + 20GB CPU RAM
)

drafter = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ),
    device_map="auto",
    max_memory={0: "8GB", "cpu": "10GB"}
)
```

### Step 4 — Optimize Inference (Batch Size = 1)
```python
# Keep batch size minimal to fit in 62GB
batch_size = 1  # Process one query at a time

# Enable flash attention for faster, memory-efficient inference
from transformers import GenerationConfig

gen_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    use_cache=True,  # Enables KV caching
)
```

### Step 5 — Reduce FAISS Index Size (Optional but Recommended)
```python
import faiss

# Instead of full Wikipedia, use NQ evidence corpus only
# This saves ~25GB vs. full Wikipedia

# Alternative: Use only top-tier vectors
# Store FP16 instead of FP32
index = faiss.IndexFlatIP(384)
index.add(embeddings_fp16.astype('float16'))
```

---

## Memory Monitoring During Runs

```python
# Add this logging to your RASD loop
import torch

def log_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"GPU: {allocated:.1f}GB allocated | {reserved:.1f}GB reserved | Peak: {max_allocated:.1f}GB")
    
# Call in your draft loop
for query in queries:
    log_memory()
    # ... speculative decoding loop ...
```

---

## Decision Tree

```
Start with 70B+8B config
├─ Runs without OOM? → Continue ✅
│  └─ Monitor memory; if >95% utilization, 
│     add warnings in Future Work section
│
└─ Hits OOM? → Immediately switch to 8B+1B fallback
   └─ Document in Future Work:
      "We were constrained to use an 8B verifier + 1B drafter
       due to the 62 GB GPU memory ceiling on available hardware.
       The intended 70B + 8B configuration requires ≥80GB VRAM,
       achievable on A100-80GB or H100 systems."
```

---

## Important Notes for Your Report [1]

When documenting your hardware in the **Future Work** section, be specific:

> "Our experiments were conducted on a GPU with 12 GB dedicated + 50 GB shared memory (62 GB total). The 70B verifier + 8B drafter configuration fits within this constraint using 4-bit quantization, but with <5GB headroom. To enable more aggressive batch processing or longer context windows, the method would benefit from A100-80GB or H100 hardware. This would allow: (a) larger batch sizes (4–8) for faster overall wall-clock throughput, (b) longer sequence lengths for richer retrieved context, and (c) evaluation on larger corpora like full Wikipedia."

Remember: **Negative or constrained results do not affect your grade [1].** A scientifically rigorous report about limitations is valued as highly as positive results.

---

## TL;DR: Start Here

1. **Use the 50GB shared pool as your primary GPU memory.**
2. **Load 70B verifier + 8B drafter both in 4-bit quantization.**
3. **Test with `torch.cuda.memory_allocated()` in your first benchmark run.**
4. **If OOM occurs, immediately fall back to 8B+1B and document why.**
5. **You have enough memory — the tight fit is exactly why you need careful profiling.**

Good luck with RASD!

# virtual environment

# Virtual Environment Setup for RASD: Anaconda vs. Alternatives

The coursework specification does not mandate a specific environment management tool. However, I can provide guidance based on best practices for your RASD project:

## Recommendation: **Yes, use Anaconda (or Miniconda)**

**Why Anaconda is ideal for your RASD project:**

1. **Dependency Management**: Anaconda handles complex C/CUDA dependencies (PyTorch, FAISS, etc.) more reliably than pip alone. This is critical because:
   - FAISS GPU requires CUDA toolkit alignment
   - PyTorch CUDA versions are strict
   - BitsAndBytes (for 4-bit quantization) needs proper compiler support

2. **Environment Isolation**: Create separate environments for:
   - `rasd-main` (production experiments)
   - `rasd-dev` (debugging, testing)
   - `rasd-colab` (if running on Colab)

3. **Reproducibility**: Your submitted `.zip` source code should include an `environment.yml` file so graders/reviewers can exactly replicate your environment.

4. **GPU/CPU Flexibility**: Anaconda makes it easier to switch between CPU-only testing (local) and GPU-heavy benchmarking (Colab) without dependency conflicts.

## Setup Steps

### 1. Install Anaconda/Miniconda
```bash
# Download Miniconda (lighter than Anaconda) from:
# https://docs.conda.io/en/latest/miniconda.html

# Verify installation
conda --version
```

### 2. Create a Virtual Environment
```bash
# Create environment for RASD
conda create -n rasd python=3.10 -y

# Activate it
conda activate rasd
```

**Why Python 3.10?** LLaMA-3.1 tooling, PyTorch 2.3.0, and FAISS all have robust support for 3.10. Avoid 3.11+ if uncertain about CUDA compatibility.

### 3. Create an `environment.yml` File
```yaml
name: rasd
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pytorch::pytorch::2.3.0
  - pytorch::pytorch-cuda=12.1
  - pytorch::torchvision
  - pytorch::torchaudio
  - conda-forge::numpy
  - conda-forge::pandas
  - conda-forge::matplotlib
  - conda-forge::jupyter
  - conda-forge::ipywidgets
  - pip
  - pip:
    - transformers==4.43.0
    - accelerate==0.31.0
    - datasets==2.20.0
    - bitsandbytes==0.43.1
    - faiss-gpu==1.7.4
    - sentence-transformers==3.0.1
    - evaluate==0.4.2
    - vllm==0.5.0
    - tqdm
    - scikit-learn
    - rouge-score
    - nltk
```

### 4. Reproduce Environment from File
```bash
# Export your environment for submission
conda env export > environment.yml

# Others (or graders) can reproduce with:
conda env create -f environment.yml
conda activate rasd
```

### 5. Include in Your `.zip` Submission
Your submitted `.zip` should contain:
```
rasd-submission.zip
├── rasd_notebook.ipynb          (main Colab notebook)
├── environment.yml              (conda environment file)
├── README.md                    (setup instructions)
├── src/
│   ├── rasd_pipeline.py        (core RASD implementation)
│   ├── retrieval.py            (FAISS retrieval module)
│   ├── evaluation.py           (benchmarking code)
│   └── ...
├── data/
│   └── (sample data, if needed)
└── logs/
    └── (results, if needed)
```

---

## Alternative: Use `requirements.txt` Only

If you prefer **not** to use Anaconda (e.g., if running only on Colab):

```bash
# Install with pip
pip install -r requirements.txt
```

**Drawback**: Harder to manage CUDA/GPU dependencies across different machines. Not recommended for reproducibility.

---

## For Colab Submission

The coursework spec explicitly mentions `.ipynb` files from Colab as acceptable deliverables. You can:

1. **Develop locally with Anaconda** (faster iteration)
2. **Export as `.ipynb`** and run on Colab for final benchmarks (to use A100 GPUs)
3. **Include both** in your `.zip`:
   - `rasd_local_dev.ipynb` (local development)
   - `rasd_colab_final.ipynb` (production runs on Colab)
   - `environment.yml` (for local reproducibility)

---

## TL;DR

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| **Anaconda + `.yml`** | CUDA deps managed, reproducible, flexible | ~5 min setup | ✅ **Use this** |
| **pip + `requirements.txt`** | Simple, lightweight | GPU deps fragile | Fallback only |
| **Colab only** | No local setup, GPU included | Hard to reproduce locally | For final runs only |

**For your 62 GB GPU setup**, Anaconda is ideal because you can test locally on your 12 GB GPU, then scale up on Colab's A100 without environment conflicts.

# GitHub guide
# GitHub Repository Management for RASD Project

Since you're running the project on your local machine, here's a comprehensive guide to managing your GitHub repository effectively while maintaining reproducibility for the final submission.

---

## Repository Structure

Create this directory structure in your GitHub repo:

```
rasd-project/
├── README.md                          # Setup instructions & overview
├── environment.yml                    # Conda environment file
├── requirements.txt                   # Pip requirements (backup)
├── .gitignore                         # Exclude large files
│
├── src/
│   ├── __init__.py
│   ├── rasd_pipeline.py              # Core RASD implementation
│   ├── retrieval.py                  # FAISS retrieval module
│   ├── baseline_speculative.py       # Vanilla speculative decoding
│   ├── evaluation.py                 # Benchmarking & metrics
│   ├── utils.py                      # Helper functions
│   └── config.py                     # Hyperparameters & settings
│
├── notebooks/
│   ├── 01_setup_and_baseline.ipynb   # Week 1 exploration
│   ├── 02_rasd_implementation.ipynb  # Week 3-4 development
│   ├── 03_final_benchmarks.ipynb     # Week 5 full runs
│   └── rasd_final.ipynb              # Submission notebook
│
├── data/
│   ├── nq_sample.json                # Small test dataset
│   ├── triviaqa_sample.json          # Small test dataset
│   └── README.md                     # Data source documentation
│
├── logs/
│   ├── baseline_metrics.csv
│   ├── rasd_metrics.csv
│   └── ablation_results.csv
│
├── results/
│   ├── figures/                      # Plots for report
│   ├── tables/                       # Results tables (CSV/LaTeX)
│   └── results_summary.md            # Key findings
│
├── report/
│   ├── rasd_report.tex               # LaTeX source (from template)
│   ├── rasd_report.pdf               # Final PDF
│   └── references.bib                # BibTeX references
│
└── checkpoints/
    ├── .gitkeep                      # Placeholder (DO NOT commit model files)
    └── README.md                     # Instructions for downloading models
```

---

## Step 1: Initialize GitHub Repository

### Create on GitHub
1. Go to https://github.com/new
2. Create repository `rasd-project` (private recommended if coursework is confidential)
3. Initialize with README (uncheck for now)

### Clone and Set Up Locally
```bash
# Clone repo
git clone https://github.com/YOUR-USERNAME/rasd-project.git
cd rasd-project

# Create directory structure
mkdir -p src notebooks data logs results/figures results/tables report checkpoints

# Create placeholder files
touch src/__init__.py
touch data/README.md
touch checkpoints/README.md
```

---

## Step 2: Create Installation Files

### `environment.yml` (Conda)
```yaml
name: rasd
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pytorch::pytorch::2.3.0
  - pytorch::pytorch-cuda=12.1
  - pytorch::torchvision
  - pytorch::torchaudio
  - conda-forge::numpy
  - conda-forge::pandas
  - conda-forge::matplotlib
  - conda-forge::jupyter
  - conda-forge::scikit-learn
  - pip
  - pip:
    - transformers==4.43.0
    - accelerate==0.31.0
    - datasets==2.20.0
    - bitsandbytes==0.43.1
    - faiss-gpu==1.7.4
    - sentence-transformers==3.0.1
    - evaluate==0.4.2
    - vllm==0.5.0
    - tqdm
    - rouge-score
    - nltk
```

### `requirements.txt` (Pip Backup)
```
torch==2.3.0
transformers==4.43.0
accelerate==0.31.0
datasets==2.20.0
bitsandbytes==0.43.1
faiss-gpu==1.7.4
sentence-transformers==3.0.1
evaluate==0.4.2
vllm==0.5.0
tqdm
numpy
pandas
matplotlib
scikit-learn
rouge-score
nltk
```

---

## Step 3: Create `.gitignore`

```bash
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Large model files (CRITICAL: Don't commit LLaMA weights)
*.bin
*.safetensors
checkpoints/*.pt
checkpoints/*.pth
*.onnx

# FAISS indexes (too large)
*.faiss
*.pkl

# Data files (optional; include only samples)
data/*
!data/README.md
!data/*_sample.json

# Logs and outputs (optional)
logs/*.csv
logs/*.json
results/tables/*.csv
results/figures/*.png

# OS
.DS_Store
Thumbs.db

# Colab temp files
.colab-internals/
```

---

## Step 4: Create README.md

```markdown
# RASD: Retrieval-Augmented Speculative Decoding

A coursework project (APAI4099) implementing token-level speculative decoding 
augmented with retrieval-based context for improved factual grounding and speed.

## Quick Start

### Prerequisites
- CUDA 12.1 compatible GPU (12GB+ VRAM)
- ~50GB disk space for models and indexes

### Installation

#### Option 1: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate rasd
```

#### Option 2: Pip
```bash
pip install -r requirements.txt
```

### Download Models & Indexes

```bash
# Models are downloaded automatically from HuggingFace on first run
# You need a HuggingFace token with LLaMA access:
huggingface-cli login

# FAISS index will be built or downloaded automatically
# See notebooks/01_setup_and_baseline.ipynb for details
```

## Project Structure

```
src/              - Core implementation
notebooks/        - Jupyter notebooks (development → submission)
data/             - Sample datasets
results/          - Benchmarking results & plots
report/           - LaTeX report source
```

## Usage

### Development (Local)
```bash
# Run baseline experiments
jupyter notebook notebooks/01_setup_and_baseline.ipynb

# Develop RASD pipeline
jupyter notebook notebooks/02_rasd_implementation.ipynb
```

### Benchmarking
```bash
# Full benchmark run
python src/evaluation.py --config config.yaml --output results/
```

### Generate Report
```bash
# Compile LaTeX (requires pdflatex)
cd report/
pdflatex -interaction=nonstopmode rasd_report.tex
```

## Evaluation Results

See `results/results_summary.md` for key findings.

| Method | Speed (tokens/sec) | Acceptance Rate | F1 Score (NQ) |
|--------|-------------------|-----------------|---------------|
| Vanilla Speculative | X.XX | X.XX% | X.XX |
| RASD (adaptive) | X.XX | X.XX% | X.XX |

## Repository Notes

- **Models**: Not committed (see `checkpoints/README.md`)
- **Large indexes**: Not committed; see setup instructions
- **Results**: Committed for reproducibility tracking

## Author

[Your Name] - APAI4099 Coursework

## License

Academic use only (coursework submission).
```

---

## Step 5: Commit and Push Initial Setup

```bash
cd rasd-project

# Stage all files
git add .

# Initial commit
git commit -m "Initial project structure with setup files"

# Push to GitHub
git push -u origin main
```

---

## Step 6: Weekly Workflow During Development

### Each Week's Commits

```bash
# Week 1: Setup complete
git add environment.yml requirements.txt notebooks/01_setup_and_baseline.ipynb
git commit -m "Week 1: Environment setup & baseline speculative decoding"
git push

# Week 2: Literature & methodology
git add src/config.py src/utils.py notebooks/02_rasd_implementation.ipynb
git commit -m "Week 2: Algorithm formalization & draft implementation"
git push

# Week 3-4: Core RASD development
git add src/rasd_pipeline.py src/retrieval.py
git commit -m "Week 3-4: RASD pipeline with adaptive retrieval"
git push

# Week 5: Benchmarks
git add logs/ results/ notebooks/03_final_benchmarks.ipynb
git commit -m "Week 5: Full benchmark runs on NQ & TriviaQA"
git push

# Week 6-8: Report writing
git add report/ results/results_summary.md
git commit -m "Week 6-8: Report generation & final submission notebook"
git push
```

---

## Step 7: Final Submission Package

One week before deadline, create the submission `.zip`:

```bash
# Create clean checkout (avoid uncommitted files)
cd /tmp
git clone https://github.com/YOUR-USERNAME/rasd-project.git rasd-submission
cd rasd-submission

# Remove unnecessary files to reduce size
rm -rf .git .github .gitignore

# Create zip (max ~100MB)
zip -r rasd-submission.zip . \
    -x "logs/*" "results/figures/*" "checkpoints/*.bin" "checkpoints/*.pt"

# Verify size
ls -lh rasd-submission.zip

# Move to Desktop for upload to Moodle
mv rasd-submission.zip ~/Desktop/
```

### Submission Checklist
- ✅ `environment.yml` present
- ✅ `README.md` with clear setup instructions
- ✅ `src/` directory with all Python modules
- ✅ Final notebook (`rasd_final.ipynb` or equivalent)
- ✅ `report/rasd_report.pdf` (compiled from LaTeX)
- ✅ `results/results_summary.md` with key metrics
- ✅ Large model files **excluded**
- ✅ `.zip` file size <100 MB

---

## Important Notes for Graders

### Installation from Your `.zip`

Graders should be able to reproduce your work with:

```bash
# Extract submission
unzip rasd-submission.zip
cd rasd-project

# Set up environment
conda env create -f environment.yml
conda activate rasd

# Download models on first run
huggingface-cli login
python notebooks/rasd_final.ipynb  # or jupyter notebook

# Re-run benchmarks
python src/evaluation.py --output results_reproduction/
```

The first run will download ~40GB of models; the environment.yml ensures dependency consistency.

---

## GitHub Best Practices for Your Project

| Practice | Example |
|----------|---------|
| **Commit early, commit often** | Commit after each notebook milestone (weekly) |
| **Meaningful commit messages** | "Week 3: Implement adaptive retrieval trigger" ✅ vs. "update code" ❌ |
| **Branch for experiments** | `git checkout -b exp/eager-retrieval` for trying ideas |
| **Exclude large files** | `.gitignore` prevents accidental 40GB model uploads |
| **Document setup** | `environment.yml` + `README.md` = reproducible for graders |

---

## Quick Reference: Common Commands

```bash
# Check status
git status

# Add specific files
git add src/rasd_pipeline.py notebooks/

# Commit with message
git commit -m "Week 3: Implement RASD core pipeline"

# Push to GitHub
git push

# View commit history
git log --oneline

# Revert changes to a file
git checkout -- src/retrieval.py

# Create a new branch (for experiments)
git checkout -b exp/new-idea
git push -u origin exp/new-idea
```

---

## Final Note

Your GitHub repository serves two purposes:
1. **Development tracking**: Shows your progress week-by-week
2. **Reproducibility**: Graders can verify report ↔ code consistency [1]

Keep commits clean, include setup files, and exclude large model weights. This ensures the final `.zip` submission is grader-friendly while maintaining full reproducibility.




# Concerns
# Your Concern About Overlap is Valid — But Manageable

You're right to flag this. The three papers you found (RAPID, RASD from ACL, and RACER) do occupy the same intersection of retrieval + speculative decoding. However, this does **not** disqualify your project. Let me explain why, and then show you how to **pivot your design** to create a clear, defensible novelty gap.

---

## Why It's Still Fine for Coursework

The specification states that your modification method must **"differ from existing methods, even if the difference is minor"** [1]. It does not require that the entire research area be unexplored. Furthermore, you are **exempt from considering papers published within 4 months before the submission deadline** (i.e., after 2026-07-30) [1]. So first, check publication dates:

- If any of RAPID, RASD (ACL), or RACER were published after **July 30, 2026**, you are formally exempt from comparing against them
- If they were published before that date, you must cite them in Related Work and **clearly explain how your method differs** [1]

Either way, overlapping related work is expected and manageable — what matters is that your specific mechanism is not identical.

---

## Detailed Comparison: Where Each Paper Differs from Your Idea

| Aspect | **RAPID (ICML)** | **RASD (ACL)** | **RACER** | **Your Proposed Method** |
|---|---|---|---|---|
| **What is retrieved** | Shortened context for the drafter | Prefix/suffix token matches from a datastore | Exact-match text patterns | Semantic passage chunks via dense retrieval (FAISS/ColBERT) |
| **Draft structure** | Standard token-by-token draft on shorter context | Retrieval tree fused with model draft tree | Pattern-based draft sequences merged with logits | Token-by-token draft conditioned on prepended retrieved passages |
| **Retrieval trigger** | Once at the start (context compression) | At each step via n-gram matching | At each step via pattern matching | **Adaptive**: triggered by drafter entropy threshold |
| **Retrieval method** | Not really retrieval; context shortening | Exact token matching (datastore lookup) | Exact string matching | **Dense semantic retrieval** (embedding similarity) |
| **Requires training** | No | No | No | No (but optionally KL-regularized fine-tuning) |
| **Primary goal** | Speed up long-context inference | Improve draft acceptance in domain shift | Training-free speedup | **Joint optimization of speed AND factual accuracy** |

---

## The Key Novelty Gaps You Can Claim

Looking at the comparison above, here are **concrete differentiators** that none of the three papers address:

### 1. **Dense Semantic Retrieval (not exact-match)**
RASD (ACL) and RACER both use **exact token/string matching** against a datastore. Your method uses **dense vector similarity** (FAISS with sentence embeddings), which retrieves semantically related passages even when there is no surface-level overlap. This is a fundamentally different retrieval paradigm.

### 2. **Adaptive Entropy-Based Retrieval Triggering**
None of the three papers use a **confidence-aware trigger**. RAPID retrieves once upfront. RASD and RACER retrieve at every step. Your method retrieves **only when the drafter is uncertain** (high entropy), which is a novel design point that balances retrieval overhead against benefit.

### 3. **Factual Accuracy as a Co-Objective**
RAPID, RASD (ACL), and RACER all focus **purely on speedup** (tokens/sec, acceptance rate). Your method explicitly evaluates and optimizes for **factual accuracy** (exact-match on NQ/TriviaQA) as a co-objective alongside speed. This dual-objective framing is distinct.

---

## Recommended Pivot: Rename and Sharpen Your Method

To avoid confusion with the existing "RASD" paper from ACL, I strongly recommend **renaming your method**. Options:

| Name | Rationale |
|---|---|
| **SENSE** (Semantic ENtropy-triggered Speculative dEcoding) | Emphasizes both the semantic retrieval and entropy trigger |
| **ADRIFT** (ADaptive Retrieval-Informed Draft Token generation) | Emphasizes the adaptive retrieval mechanism |
| **SEEKER** (Speculative dEcoding with Entropy-Keyed External Retrieval) | Emphasizes the entropy-gated retrieval |

Pick one and use it consistently. This immediately differentiates you from the ACL paper titled "RASD."

---

## Revised Novelty Statement for Your Report

Here is how your Introduction and Related Work sections should frame the contribution:

> *Several recent works explore retrieval-augmented speculative decoding. RAPID compresses long contexts for the drafter but does not perform external retrieval. RASD (He et al., ACL) and RACER use exact token matching against a datastore to construct draft trees. Our method, **SENSE**, differs in three ways: (1) we use **dense semantic retrieval** via FAISS to retrieve passage-level evidence, enabling generalization beyond surface-level token overlap; (2) we introduce an **adaptive entropy-based trigger** that invokes retrieval only when the drafter's predictive uncertainty exceeds a threshold, reducing retrieval overhead; and (3) we evaluate not only generation speed but also **factual accuracy** on knowledge-intensive QA benchmarks, treating factual grounding as a co-objective rather than a byproduct.*

---

## Updated Timeline Adjustment

Given these existing papers, add the following to your **Week 2 literature review**:

| Day | Additional Task |
|---|---|
| Day 1 | Read RAPID, RASD (ACL), and RACER in full. Create a detailed comparison table (like the one above) |
| Day 2 | Verify publication dates — confirm whether each falls within or outside the 4-month exemption window [1] |
| Day 3 | Finalize your method name and the 3 differentiators. Write the novelty statement paragraph |

And in your **Related Work section**, you must clearly explain the similarities and differences between your method and these existing approaches [1]. The comparison table above gives you a ready-made structure.

---

## Bottom Line

Your concern is well-founded but the overlap is **not a problem** — it's an opportunity. These papers validate that the research direction is important. Your job is to carve out a clear, specific niche. The three differentiators (dense semantic retrieval, entropy-based adaptive triggering, dual speed+accuracy objective) give you a defensible and genuine contribution. Just rename the method to avoid direct confusion with the ACL "RASD" paper, cite all three papers properly, and explain the differences clearly in your report [1].
````