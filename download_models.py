"""Download the AI models used by the engine into ./models/huggingface.

The engine runs fully offline (HF_HUB_OFFLINE=1), so run this ONCE with an
internet connection before your first render.

    python download_models.py            # required models (~15 GB)
    python download_models.py --all      # + metric depth models and SAM (~5 GB more)
"""
import argparse
import os
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "models" / "huggingface"
os.environ["HF_HOME"] = str(CACHE)

REQUIRED = [
    "prs-eth/marigold-depth-v1-1",            # depth (config.DEPTH_MODE = "marigold")
    "prs-eth/marigold-normals-v1-1",          # surface normals
    "prs-eth/marigold-iid-lighting-v1-1",     # learned illumination
    "nvidia/segformer-b5-finetuned-ade-640-640",   # scene segmentation
    "depth-anything/Depth-Anything-V2-Large-hf",   # depth (DEPTH_MODE = "relative")
]
OPTIONAL = [
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",  # DEPTH_MODE = "metric"
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "facebook/sam-vit-huge",                  # ENABLE_SAM_SEGMENTATION = True
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also fetch optional models")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    repos = REQUIRED + (OPTIONAL if args.all else [])
    for i, repo in enumerate(repos, 1):
        print(f"[{i}/{len(repos)}] {repo}")
        snapshot_download(repo_id=repo, cache_dir=str(CACHE / "hub"))
    print(f"\nDone. Models are in {CACHE}")


if __name__ == "__main__":
    main()
