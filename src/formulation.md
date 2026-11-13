## SENSE: Semantic Entropy-triggered Speculative Decoding

### Notation
- **q**: input query
- **D_verif**: large verifier model
- **D_draft**: small drafter model
- **R()**: retrieval function (FAISS dense search)
- **H(p)**: entropy of logit distribution
- **τ**: entropy threshold for retrieval trigger

### Algorithm
1. Generate draft tokens: t_1, ..., t_n from D_draft
2. At each step i:
   - Compute entropy H_i of D_draft logits
   - If H_i > τ: retrieve top-3 passages via FAISS
   - Prepend passages to next draft input
   - Otherwise: continue without retrieval
3. Verify draft with D_verif, accept/reject tokens

### Key Difference from RASD (ACL)
- RASD uses exact n-gram matching
- SENSE uses dense semantic retrieval
- Adaptive triggering (entropy-based) vs. fixed-frequency