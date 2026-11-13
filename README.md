# ARASD
Adaptive Retrieval-Augmented Speculative Decoding
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
conda activate arasd
