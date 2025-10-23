# GLVQ — Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression  


This repository accompanies the paper:  
**Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression (GLVQ)**  
**NeurIPS 2025**



---

## 🔍 What is GLVQ?

GLVQ is a **post-training quantization (PTQ)** framework for LLMs that learns **group-specific lattice vector quantizers** to encode weights at 2–4 bits while preserving accuracy and decoding efficiency.



---

## 🚫 Code Release Status

This repository currently hosts documentation only.  
As this work is partially funded by **Alibaba Group**, the code release is pending internal legal / compliance approval.

- Status: **Under Alibaba legal/compliance review**
- ETA: **Will be released once approval is granted**

Please **watch / star** this repo to get notified when the code becomes available.


---

## 🧠 Conceptual Overview

![GLVQ Overview](assets/glvq-overview.png)




---

## 📊 Main Results (excerpt from Table 1 of paper)

| Method | Bits | WikiText2 (2-70) ↓ | C4 (2-70) ↓ |
|-------:|:----:|:------------------:|:-----------:|
| FP16 | 16 | 3.12 | 4.97 |
| QuIP# | 2 | 3.91 | 5.71 |
| QTIP | 2 | 3.78 | 5.56 |
| **GLVQ-8D (ours)** | **2** | **3.62** | **5.25** |
| **GLVQ-32D (ours)** | **2** | **3.36** | **5.04** |

GLVQ achieves **lower perplexity** while maintaining **similar throughput**.



---

## 📄 Paper

(coming after NeurIPS camera-ready)

---

## 📌 Citation

```bibtex
@inproceedings{GLVQ_NeurIPS25,
  title = {Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression},
  author = {Zhang, Xi and Wu, Xiaolin and Wang, Jiamang and Lin, Weisi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2025}
}
