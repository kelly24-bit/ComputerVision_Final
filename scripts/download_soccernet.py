"""Download + verify SoccerNet play-recognition data (features + labels).

Needs a Hugging Face account with access approved on SoccerNet/SN-Features and
SoccerNet/SN-Labels. We pull pre-extracted features + Labels-v2.json only --
NOT the NDA-gated raw video.

Run once:   huggingface-cli login
Then:       python scripts/download_soccernet.py

Success means: one game's ResNet feature .npy and its Labels-v2.json download
and open with sane shapes.
"""
import glob
import json
import os
import sys

try:
    import numpy as np
    from SoccerNet.Downloader import SoccerNetDownloader as SNdl
except ImportError:
    sys.exit("deps missing. Run: pip install -r requirements.txt")

LOCAL = "data/raw/soccernet"
FILES = ["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy", "Labels-v2.json"]


def main():
    os.makedirs(LOCAL, exist_ok=True)
    d = SNdl(LocalDirectory=LOCAL)

    print("Downloading the 'valid' split subset to confirm access...")
    try:
        d.downloadGames(files=FILES, split=["valid"])
    except Exception as e:  # noqa: BLE001 - surface the real cause to the user
        sys.exit(
            "[fail] download error -- most likely you are not logged in "
            "(`huggingface-cli login`) or access is not yet approved on "
            f"SN-Features / SN-Labels.\n\n{e}"
        )

    npys = glob.glob(os.path.join(LOCAL, "**", "*ResNET_TF2_PCA512.npy"), recursive=True)
    labels = glob.glob(os.path.join(LOCAL, "**", "Labels-v2.json"), recursive=True)
    if not npys or not labels:
        sys.exit("[fail] expected files did not land under " + LOCAL)

    arr = np.load(npys[0])
    with open(labels[0]) as f:
        lab = json.load(f)

    print(f"[ok] feature array: shape {arr.shape}  (frames x feature_dim)")
    print(f"[ok] labels file:   {len(lab.get('annotations', []))} annotations")
    print("\nSoccerNet features + labels confirmed working.")


if __name__ == "__main__":
    main()
