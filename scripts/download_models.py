"""
Download every model TrustRAG uses into the Hugging Face cache (one-time, needs internet).

    python scripts/download_models.py

After this the app and the demo run fully offline.

Behind a corporate HTTPS proxy, downloads can fail with CERTIFICATE_VERIFY_FAILED.
On macOS this script builds a CA bundle from the system keychain (which holds the
proxy's root certificate) and relaxes Python 3.13+'s strict X.509 check, which
rejects proxy CA certificates that lack a key-usage extension. Verification of the
server certificate itself stays on.
"""
import os
import platform
import ssl
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Only the PyTorch weights and tokenizer files; the repos also hold ONNX/OpenVINO/TF copies.
MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", ["*.json", "*.safetensors", "tokenizer.model"]),
    ("sentence-transformers/all-MiniLM-L6-v2", ["*.json", "*.safetensors", "*.txt", "1_Pooling/*"]),
    ("cross-encoder/nli-deberta-v3-small", ["*.json", "*.safetensors", "*.model"]),
    ("distilbert-base-uncased", ["*.json", "*.safetensors", "*.txt"]),
]


def build_macos_bundle() -> str:
    import certifi

    bundle = ROOT / ".certs" / "macos_bundle.pem"
    bundle.parent.mkdir(exist_ok=True)
    pem = subprocess.run(
        ["security", "find-certificate", "-a", "-p",
         "/Library/Keychains/System.keychain",
         "/System/Library/Keychains/SystemRootCertificates.keychain"],
        check=True, capture_output=True, text=True,
    ).stdout
    bundle.write_text(pem + Path(certifi.where()).read_text())
    return str(bundle)


def patch_ssl(cafile: str) -> None:
    original = ssl.create_default_context

    def context(*args, **kwargs):
        ctx = original(ssl.Purpose.SERVER_AUTH, cafile=cafile)
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        return ctx

    ssl.create_default_context = context


def main() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    if platform.system() == "Darwin":
        patch_ssl(build_macos_bundle())

    from huggingface_hub import snapshot_download

    for repo, patterns in MODELS:
        print(f"Downloading {repo} ...")
        path = snapshot_download(repo, allow_patterns=patterns)
        print(f"  ready: {path}")
    print("All models are cached. The app now runs offline.")


if __name__ == "__main__":
    sys.exit(main())
