"""Compact transformer: zero-shot and fine-tuned multilingual-e5-small (sentence-transformers)."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CoSENTLoss

from . import config


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(path_or_name: str | Path) -> SentenceTransformer:
    model = SentenceTransformer(str(path_or_name), device=get_device())
    model.max_seq_length = config.MAX_SEQ_LENGTH
    return model


def finetune_encoder(train_df: pd.DataFrame, out_dir: Path = config.FINETUNED_MODEL_DIR) -> Path:
    """Fine-tune e5-small on (content, modified_content) pairs with CoSENT regression on similarity_score."""
    model = load_encoder(config.BASE_MODEL_NAME)
    dataset = Dataset.from_dict(
        {
            "sentence1": [config.E5_PREFIX + t for t in train_df["content"]],
            "sentence2": [config.E5_PREFIX + t for t in train_df["modified_content"]],
            "label": train_df["similarity_score"].astype("float32").tolist(),
        }
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(config.MODELS_DIR / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_strategy="no",
        report_to="none",
        seed=config.RANDOM_STATE,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=dataset, loss=CoSENTLoss(model)
    )
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    return out_dir


def embed(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """L2-normalised float32 embeddings."""
    return model.encode(
        [config.E5_PREFIX + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")


def pair_features(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Classifier features for a pair: |u-v|, u*v and cosine."""
    cos = (ea * eb).sum(axis=1, keepdims=True)
    return np.hstack([np.abs(ea - eb), ea * eb, cos])


def faiss_topk(query_emb: np.ndarray, corpus_emb: np.ndarray, k_max: int) -> np.ndarray:
    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(np.ascontiguousarray(corpus_emb, dtype="float32"))
    _, idx = index.search(np.ascontiguousarray(query_emb, dtype="float32"), k_max)
    return idx


def model_size_mb(model: SentenceTransformer) -> float:
    """Size of fp32 weights in megabytes."""
    return sum(p.numel() for p in model.parameters()) * 4 / 1e6
