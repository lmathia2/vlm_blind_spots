"""Global configuration for the VLM blind spots evaluation framework."""

import os
from pathlib import Path

# Model
MODEL = os.environ.get("VLM_MODEL", "claude-haiku-4-5-20251001")
TEMPERATURE = 0.0
MAX_TOKENS = 512

# Parallelism
MAX_WORKERS = int(os.environ.get("VLM_MAX_WORKERS", "10"))

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_ASSETS_DIR = PROJECT_ROOT / "report_assets"
REFERENCE_DIR = PROJECT_ROOT / "reference" / "vision-llms-are-blind"

# Ensure directories exist
for d in [DATA_DIR, RESULTS_DIR, REPORT_ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
