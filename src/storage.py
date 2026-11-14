"""SQLite storage for test pairs, experiment metrics and per-pair predictions."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import config
from .sql_queries import QUERIES

SCHEMA = """
CREATE TABLE IF NOT EXISTS pairs (
    pair_id                   TEXT PRIMARY KEY,
    base_id                   TEXT NOT NULL,
    category                  TEXT,
    type_duplicate            TEXT NOT NULL,
    similarity_score          REAL NOT NULL,
    content_len_tokens        INTEGER,
    modified_len_tokens       INTEGER
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    method                    TEXT NOT NULL,
    n_test_pairs              INTEGER,
    spearman                  REAL,
    f1_macro                  REAL,
    accuracy                  REAL,
    recall_at_1               REAL,
    recall_at_5               REAL,
    recall_at_10              REAL,
    hit_at_1                  REAL,
    hit_at_5                  REAL,
    hit_at_10                 REAL,
    fit_seconds               REAL,
    infer_seconds             REAL,
    infer_seconds_per_1000    REAL,
    model_size_mb             REAL,
    rss_mb                    REAL,
    created_at                TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    experiment_id             INTEGER NOT NULL REFERENCES experiments(experiment_id),
    pair_id                   TEXT NOT NULL REFERENCES pairs(pair_id),
    pred_score                REAL,
    pred_type                 TEXT,
    PRIMARY KEY (experiment_id, pair_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_pair ON predictions(pair_id);
"""

EXPERIMENT_COLUMNS = [
    "method", "n_test_pairs", "spearman", "f1_macro", "accuracy",
    "recall_at_1", "recall_at_5", "recall_at_10", "hit_at_1", "hit_at_5", "hit_at_10",
    "fit_seconds", "infer_seconds",
    "infer_seconds_per_1000", "model_size_mb", "rss_mb",
]


def connect(db_path=config.DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _nan_to_none(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _to_int(value):
    value = _nan_to_none(value)
    return None if value is None else int(value)


def save_pairs(conn: sqlite3.Connection, test: pd.DataFrame) -> int:
    rows = [
        (
            str(r.id), str(r.base_id), _nan_to_none(getattr(r, "category", None)),
            str(r.type_duplicate), float(r.similarity_score),
            _to_int(r.content_len_tokens), _to_int(r.modified_content_len_tokens),
        )
        for r in test.drop_duplicates("id").itertuples(index=False)
    ]
    conn.executemany("INSERT OR REPLACE INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    return len(rows)


def save_experiment(conn: sqlite3.Connection, metrics: dict, predictions: pd.DataFrame) -> int:
    """Store one method's metrics and predictions; an earlier run of the same method is replaced."""
    method = metrics["method"]
    conn.execute(
        "DELETE FROM predictions WHERE experiment_id IN "
        "(SELECT experiment_id FROM experiments WHERE method = ?)",
        (method,),
    )
    conn.execute("DELETE FROM experiments WHERE method = ?", (method,))
    values = [_nan_to_none(metrics.get(c)) for c in EXPERIMENT_COLUMNS]
    cur = conn.execute(
        f"INSERT INTO experiments ({', '.join(EXPERIMENT_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(EXPERIMENT_COLUMNS))})",
        values,
    )
    exp_id = cur.lastrowid
    rows = [
        (exp_id, str(pid), _nan_to_none(score), str(t))
        for pid, score, t in zip(predictions["pair_id"], predictions["pred_score"], predictions["pred_type"])
    ]
    conn.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    return exp_id


def run_query(conn: sqlite3.Connection, name: str, method: str = config.HYBRID_METHOD) -> pd.DataFrame:
    """Execute a named report query and return the result as a DataFrame."""
    if name not in QUERIES:
        raise KeyError(f"Unknown query '{name}'. Available: {', '.join(QUERIES)}")
    sql = QUERIES[name]["sql"]
    return pd.read_sql_query(sql, conn, params={"method": method})
