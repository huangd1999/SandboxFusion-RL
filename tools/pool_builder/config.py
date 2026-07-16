# coding: utf-8
# Pool-builder configuration (vendored from the UnLeakedTestBench pipeline).
#
# Every location is env-overridable so the builder runs from a fresh clone:
#   TESTGEN_PYTHON_BIN   bin/ dir of the env with cosmic-ray + pytest-cov
#   TESTGEN_DATASETS_DIR directory holding <subset>.jsonl task files
#   TESTGEN_POOL_ROOT    where pools are written (same var the reward layer reads)

import os

# --- Interpreter -------------------------------------------------------------
# Must be an env with cosmic-ray + pytest-cov installed (the sandbox runtime
# env works). Defaults preserve the original machine layout.
_BIN = os.environ.get("TESTGEN_PYTHON_BIN", "/home/nvidia/miniconda3/envs/testrl/bin")
PYTHON = f"{_BIN}/python"
COSMIC_RAY = f"{_BIN}/cosmic-ray"
CR_REPORT = f"{_BIN}/cr-report"

# --- Dataset locations --------------------------------------------------------
DATASETS_DIR = os.environ.get("TESTGEN_DATASETS_DIR",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets"))
DATASET_FILES = {
    "PLT": os.path.join(DATASETS_DIR, "PLT.jsonl"),
    "ULT": os.path.join(DATASETS_DIR, "ULT.jsonl"),
    "ULT_Lite": os.path.join(DATASETS_DIR, "ULT_Lite.jsonl"),
}

# --- Storage layout -----------------------------------------------------------
DATA_ROOT = os.environ.get("TESTGEN_DATA_ROOT", "/data/UnLeakedTestBench")
POOL_ROOT = os.environ.get("TESTGEN_POOL_ROOT", os.path.join(DATA_ROOT, "pool"))
EVAL_CACHE = os.path.join(DATA_ROOT, "eval_cache")
SUBSETS_DIR = os.path.join(DATA_ROOT, "subsets")
LOGS_DIR = os.path.join(DATA_ROOT, "logs")

for _d in (POOL_ROOT, EVAL_CACHE, SUBSETS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)


def pool_dir(subset: str, task_id) -> str:
    return os.path.join(POOL_ROOT, subset, f"task_{task_id}")
