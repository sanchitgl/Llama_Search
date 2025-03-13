# 🦙 LLama-IR: Combining Lexical & Semantic Retrieval for Query Search

## Project Overview

This project explores the integration of **lexical and semantic retrieval methods** to enhance query search accuracy. Traditional search engines typically rely on either lexical (e.g., **BM25**) or semantic retrieval (e.g., **Transformer-based models**). We propose a hybrid approach that **combines BM25 with advanced semantic models like DistilBERT and LLama** to improve retrieval effectiveness.

Our results demonstrate that ensemble methods outperform individual approaches, leading to significant improvements in **nDCG, MAP, Recall, and Precision** scores. The **RepLlama** (bi-encoder LLama) model was a key focus of this project, showcasing state-of-the-art performance when combined with BM25.

## Key Features

- **Lexical Retrieval**: Uses BM25 for keyword-based matching.
- **Semantic Retrieval**: Employs DistilBERT and **RepLlama (LLama2-7B fine-tuned for retrieval)** for context-aware matching.
- **Hybrid Ensemble Approach**: Combines BM25 and transformer-based models using a **weighted scoring** technique.
- **Datasets Used**:
  - **Scifact** (Scientific literature retrieval)
  - **MS Marco Tiny** (Real-world search queries)


## Methodology

### 1. Lexical Retrieval (BM25)
BM25 ranks documents based on term frequency and document length. While effective for simple keyword-based queries, it **lacks contextual understanding**.

### 2. Semantic Retrieval (Transformer-based Models)
- **DistilBERT**: A lightweight alternative to BERT, offering high performance with reduced computational cost.
- **RepLlama (LLama2-7B Bi-Encoder)**:
  - Fine-tuned using **contrastive loss (InfoNCE loss)**.
  - Generates embeddings for queries and documents.
  - Uses cosine similarity to rank results.

### 3. Hybrid Approach (BM25 + Transformer Model)
- We compute **BM25 scores** and **semantic similarity scores** separately.
- A **weighted ensemble** approach is applied to combine both scores for final ranking.
- **Fine-tuned weight selection** optimizes retrieval performance.

## Experimental Results

### Performance on **Scifact Dataset**

| Model | nDCG@10 | MAP@10 | Recall@10 | P@10 |
|-------|--------|-------|---------|------|
| BM25 | 0.6776 | 0.6324 | 0.8004 | 0.089 |
| **RepLlama** | **0.7599** | **0.7110** | **0.8987** | **0.1017** |
| **RepLlama + BM25** | **0.7647** | **0.7185** | **0.8937** | **0.1013** |

### Performance on **MS Marco Tiny Dataset**

| Model | nDCG@10 | MAP@10 | Recall@10 | P@10 |
|-------|--------|-------|---------|------|
| BM25 | 0.4873 | 0.1379 | 0.1666 | 0.5407 |
| **RepLlama** | **0.7088** | **0.2131** | **0.2446** | **0.7629** |
| **RepLlama + BM25** | **0.6922** | **0.2026** | **0.2351** | **0.7389** |

### Key Takeaways
**RepLlama outperforms BM25 and DistilBERT individually.**  
**Combining BM25 with LLama enhances retrieval performance further.**  
**Our ensemble approach achieves higher contextual relevance and precision.**


## Full Project Report
For a detailed explanation of our methodology, experiments, and results, check out the **full project report** here:  [LLama_IR_Project.pdf](https://github.com/sanchitgl/Llama_Search/blob/main/LLama_IR_Project.pdf)


