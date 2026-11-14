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
HYBRID_SUFFIX = "+lex"              # method variant that adds lexical features to the classifier
HYBRID_METHOD = FINETUNED_METHOD + HYBRID_SUFFIX
LEXICAL_METHOD = "lexical_features"
E5_PREFIX = "query: "

# Group-aware splits by original document (see src/splits.py); the official split leaks.
TRAIN_SPLIT = "group_train"
VAL_SPLIT = "group_validation"
TEST_SPLIT = "group_test"

RANDOM_STATE = 42
CSV_CHUNKSIZE = 20_000
# v1 used 256 tokens, which cut 58.5% of the documents (median 333 tokens, 2.08 tokens per word),
# so partial_cut / partial_ext differences at the end of the text were invisible.
# 512 is the model maximum (still truncates 35.1%); the batch is halved to fit an 8 GB GPU.
MAX_SEQ_LENGTH = 512
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
# Two objectives are mixed (see src/model.py): CoSENT regression and in-batch-negative contrastive.
TRAIN_ROWS_PER_OBJECTIVE = 60_000
# Scores of these types are assigned per type and contradict the real text overlap
# (e.g. partial_ext shares 95% of its words with the original but is labelled 0.40),
# so they are left out of the regression objective and used only as contrastive positives.
REGRESSION_EXCLUDED_TYPES = ("partial",)
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
