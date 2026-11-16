"""Print the environment and model/config versions used by the demo (cross-platform)."""
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import faiss
import sentence_transformers
import torch
import transformers

cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
d = cfg["detection"]
print("Python               :", sys.version.split()[0])
print("Platform             :", platform.platform())
print("torch                :", torch.__version__)
print("transformers         :", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("faiss                :", faiss.__version__)
gen = cfg["generation"]
print("Generator backend    :", gen.get("backend", "ollama"))
print("Generator model      :", gen.get("model_name"))
print("Generator max tokens :", gen.get("max_tokens"), "| temperature:", gen.get("temperature"))
print("Embedding model      :", cfg["retrieval"]["embedding_model"])
print("Top-K retrieval      :", cfg["retrieval"]["top_k"])
print("NLI model            :", cfg["nli"]["model"])
print("BERTScore model      :", cfg["bertscore"]["model_type"])
print("Evidence windows     :", d.get("use_evidence_windows"), "| size:", d.get("window_size"), "| top_n:", d.get("window_top_n"))
print("Fusion weights       :", d["weight_nli"], d["weight_bertscore"], d["weight_cosine"])
print("Thresholds           :", d["high_support_threshold"], d["medium_support_threshold"])
print("Max block ratio      :", cfg["controller"]["max_block_ratio"])
