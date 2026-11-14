# Lightweight Kazakh Near-Duplicate Detector

Final project for **Programming for Data Analysis and Database** (Applied Artificial Intelligence, MSc), topic 25.

**Team:** Temirlan Koshkimbay, Zhanserik Batyrkhanov (group AAI-2604M)

## Goal

Analyze duplicate types in Kazakh texts and build a compact transformer model for similarity scoring and duplicate-type classification.

## Dataset

- KazakhTextDuplicates v2.0 (207K document pairs, 1.77 GB CSV)
- https://huggingface.co/datasets/Arailym-tleubayeva/KazakhTextDuplicatesv2.0

The raw data is not stored in this repository because of its size.

## Planned approach

- **Baseline:** TF-IDF cosine similarity and MinHash (datasketch)
- **Compact model:** fine-tuned `intfloat/multilingual-e5-small` (sentence-transformers)
- **Retrieval:** FAISS for Recall@k
- **Metrics:** Spearman, F1-macro, Recall@k, model size, inference time
- **Data checks:** EDA, duplicate types, train–test content overlap check
- **Data handling:** Pandas with `chunksize`, results stored in SQLite

## Tools

Python 3.x, Pandas, sentence-transformers, FAISS, datasketch, SQLite

## Project status

- [x] Repository created
- [ ] EDA and duplicate-type analysis
- [ ] Baseline (TF-IDF, MinHash)
- [ ] Model fine-tuning and comparison
- [ ] SQLite storage and SQL queries
- [ ] Interactive console demo (user input → similarity / duplicate type)

## How to run

Instructions will be added once the code is ready.
