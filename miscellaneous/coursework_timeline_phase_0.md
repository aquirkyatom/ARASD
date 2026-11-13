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


### Summary of **"Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting"** (arXiv:2407.08223)

---

### **1. Core Concept & Motivation**

Traditional **Retrieval-Augmented Generation (RAG)** systems feed all retrieved documents into a large language model (LLM) to answer knowledge-intensive queries. However, this approach faces two major challenges:

* **High Latency & Computation:** Processing lengthy inputs containing multiple retrieved documents slows down generation.
* **Contextual Limitations ("Lost-in-the-Middle"):** LLMs often struggle to reason effectively over long, noisy contexts or suffer from position bias.

To solve this, the authors introduce **SPECULATIVE RAG**, a framework inspired by *Speculative Decoding*, shifting the paradigm from **token-level drafting** to **answer-level drafting**.

---

### **2. How Speculative RAG Works**

The framework employs a **divide-and-conquer strategy** using two distinct models:

1. **Document Partitioning & Clustering:**
Retrieved documents are grouped using content similarity clustering, and diverse subsets are sampled to ensure varied perspectives while eliminating redundancy.
2. **Parallel Drafting (Specialist LM):**
A smaller, highly efficient "specialist" LM (the RAG drafter) processes these distinct document subsets in parallel. It quickly generates multiple draft answers along with supporting rationales.
3. **Verification & Integration (Generalist LM):**
A larger "generalist" LM performs a single, parallel verification pass over the drafts and rationales. It assesses confidence scores and selects or refines the optimal final answer.

---

### **3. Key Advantages**

* **No Special Fine-Tuning Required:** The larger generalist LM does not require custom instruction tuning or self-reflection tags.
* **Significant Speedups:** By offloading heavy document reading and initial drafting to a smaller model, latency is drastically reduced.
* **State-of-the-Art Performance:** Outperforms traditional RAG frameworks across major benchmarks (including *TriviaQA, MuSiQue, PopQA, PubHealth,* and *ARC-Challenge*). For instance, on **PubHealth**, it improves accuracy by **12.97%** while reducing latency by **50.83%**.

### Summary of **"Online Speculative Decoding"** (arXiv:2310.07177)

---

### **1. Core Concept & Motivation**

Speculative decoding accelerates Large Language Model (LLM) inference by using a smaller, lightweight **draft model** to predict tokens, which are then verified in parallel by a larger target model.

However, a major bottleneck in standard speculative decoding is **distribution mismatch**: if the queries handled by the serving system shift or differ from the data the draft model was originally trained on, the draft model's predictions become inaccurate. This leads to a low token acceptance rate, which severely degrades inference speedup.

To solve this, the authors introduce **Online Speculative Decoding (OSD)**—a framework that continuously updates and adapts the draft model on the fly during deployment.

---

### **2. How Online Speculative Decoding (OSD) Works**

* **Leveraging Free Supervision Data:** The speculative decoding process inherently identifies where the draft model makes mistakes and provides corrective token targets from the larger model. OSD captures these interactions without extra cost.
* **Online Knowledge Distillation:** Instead of static fine-tuning, OSD uses generalized knowledge distillation (GKD) objectives to continuously align the draft model's output probability distributions with those of the target model using a live replay buffer.
* **Dynamic Adaptation:** As new user queries stream into the production system, the draft model adapts to changing domain distributions in real time, increasing its accuracy and relevance.

---

### **3. Key Findings & Performance**

* **Higher Acceptance Rates:** OSD substantially increases the token acceptance rate of the draft model (boosting acceptance by **0.1 to 0.65**).
* **Significant Latency Reduction:** By improving token acceptance, OSD achieves an end-to-end latency reduction of **1.42× to 2.17×** (and up to **3.06×** under optimal configurations) compared to traditional static speculative decoding methods, without degrading user experience or output quality.