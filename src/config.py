"""Project-wide paths and constants."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
DB_PATH = DATA_DIR / "duplicates.db"
RAW_CSV = RAW_DIR / "KazakhTextDuplicates_v2.csv"

BASE_MODEL_NAME = "intfloat/multilingual-e5-small"
FINETUNED_MODEL_DIR = MODELS_DIR / "e5-small-kk-duplicates"
FINETUNED_METHOD = "e5_finetuned"  # name used for the fine-tuned model in results and SQLite
E5_PREFIX = "query: "

# Group-aware splits by original document (see src/splits.py); the official split leaks.
TRAIN_SPLIT = "group_train"
VAL_SPLIT = "group_validation"
TEST_SPLIT = "group_test"

RANDOM_STATE = 42
CSV_CHUNKSIZE = 20_000
MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
TFIDF_FIT_SIZE = 20_000        # train pairs used to fit the TF-IDF vocabulary (memory/time limit)
RETRIEVAL_QUERIES = 5_000      # test queries used for Recall@k (full corpus is the test modified_content)
TOP_K = (1, 5, 10)

MINHASH_PERMS = 128
MINHASH_SHINGLE_SIZE = 5
MINHASH_LSH_THRESHOLD = 0.8    # leakage check threshold
MINHASH_RETRIEVAL_THRESHOLD = 0.2

TYPE_LABELS = [
    "exact",
    "noisy_soft",
    "paraphrase",
    "noisy_medium",
    "contextual",
    "noisy_hard",
    "partial",
]
