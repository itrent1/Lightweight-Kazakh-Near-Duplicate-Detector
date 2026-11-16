"""Compact transformer: zero-shot and fine-tuned multilingual-e5-small (sentence-transformers)."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

from . import config


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(path_or_name: str | Path) -> SentenceTransformer:
    model = SentenceTransformer(str(path_or_name), device=get_device())
    model.max_seq_length = config.MAX_SEQ_LENGTH
    return model


def build_training_datasets(train_df: pd.DataFrame) -> DatasetDict:
    """Two training views of the same pairs.

    regression  - (text A, text B, similarity_score) for CoSENT. Types whose score contradicts the real
                  text overlap (config.REGRESSION_EXCLUDED_TYPES, i.e. partial) are left out: v1 trained
                  on them and pushed partial pairs to cosine ~0.7 (down to -0.4), below the other types
                  and out of the top-10 of retrieval (Recall@10 fell from 0.99 to 0.76).
    contrastive - (text A, text B) positives for in-batch negatives (MultipleNegativesRankingLoss).
                  Every non-exact variant, including partial, is a positive; the other originals in the
                  batch act as negatives, which is what retrieval needs and CoSENT alone does not give.
    """
    required = {"content", "modified_content", "similarity_score", "type_duplicate"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"train_df is missing columns: {sorted(missing)}")

    rows = config.TRAIN_ROWS_PER_OBJECTIVE
    regression_df = train_df[~train_df["type_duplicate"].isin(config.REGRESSION_EXCLUDED_TYPES)]
    contrastive_df = train_df[train_df["type_duplicate"] != "exact"]
    regression_df = regression_df.sample(n=min(rows, len(regression_df)), random_state=config.RANDOM_STATE)
    contrastive_df = contrastive_df.sample(n=min(rows, len(contrastive_df)), random_state=config.RANDOM_STATE)

    prefix = config.E5_PREFIX
    return DatasetDict(
        {
            "regression": Dataset.from_dict(
                {
                    "sentence1": [prefix + t for t in regression_df["content"]],
                    "sentence2": [prefix + t for t in regression_df["modified_content"]],
                    "label": regression_df["similarity_score"].astype("float32").tolist(),
                }
            ),
            "contrastive": Dataset.from_dict(
                {
                    "anchor": [prefix + t for t in contrastive_df["content"]],
                    "positive": [prefix + t for t in contrastive_df["modified_content"]],
                }
            ),
        }
    )


def finetune_encoder(train_df: pd.DataFrame, out_dir: Path = config.FINETUNED_MODEL_DIR) -> Path:
    """Fine-tune e5-small with CoSENT regression + in-batch-negative contrastive loss (round-robin)."""
    model = load_encoder(config.BASE_MODEL_NAME)
    datasets = build_training_datasets(train_df)
    losses = {
        "regression": CoSENTLoss(model),
        "contrastive": MultipleNegativesRankingLoss(model),
    }
    args = SentenceTransformerTrainingArguments(
        output_dir=str(config.MODELS_DIR / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        # Variants of one original share the same text; without this they would sit in one batch
        # and be used as false negatives for each other.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.ROUND_ROBIN,
        logging_steps=200,
        save_strategy="no",
        report_to="none",
        seed=config.RANDOM_STATE,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=datasets, loss=losses
    )
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    return out_dir


EMBED_SUMMARY_DIM = 11  # 8 blockwise cosines + L1 + L2 + cosine


def embed(model: SentenceTransformer, texts: list[str], batch_size: int = 128) -> np.ndarray:
    """L2-normalised float32 embeddings, deduplicating repeated strings for speed."""
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype="float32")
    unique_texts = list(dict.fromkeys(texts))
    idx_map = {t: i for i, t in enumerate(unique_texts)}
    prefixed = [config.E5_PREFIX + t for t in unique_texts]
    use_amp = torch.cuda.is_available()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        uniq_emb = model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
    indices = np.fromiter((idx_map[t] for t in texts), dtype=np.int64, count=len(texts))
    return uniq_emb[indices]


def pair_features(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Classifier features for a pair: |u-v|, u-v, u*v, and 11 compact summary features.

    The final 11 columns are [block_cos_0..7, l1_dist, l2_dist, cosine].
    """
    diff = ea - eb
    abs_diff = np.abs(diff)
    prod = ea * eb
    n, d = ea.shape
    block_dim = d // 8
    eab = ea[:, : block_dim * 8].reshape(n, 8, block_dim)
    ebb = eb[:, : block_dim * 8].reshape(n, 8, block_dim)
    denom = np.linalg.norm(eab, axis=2) * np.linalg.norm(ebb, axis=2) + 1e-8
    block_cos = (eab * ebb).sum(axis=2) / denom
    l1 = abs_diff.sum(axis=1, keepdims=True)
    l2 = np.linalg.norm(diff, axis=1, keepdims=True)
    cos = prod.sum(axis=1, keepdims=True)
    return np.hstack([abs_diff, diff, prod, block_cos, l1, l2, cos]).astype("float32")


def faiss_topk(query_emb: np.ndarray, corpus_emb: np.ndarray, k_max: int) -> np.ndarray:
    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(np.ascontiguousarray(corpus_emb, dtype="float32"))
    _, idx = index.search(np.ascontiguousarray(query_emb, dtype="float32"), k_max)
    return idx


def model_size_mb(model: SentenceTransformer) -> float:
    """Size of fp32 weights in megabytes."""
    return sum(p.numel() for p in model.parameters()) * 4 / 1e6
