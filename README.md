# helixforge
Helix Forge repo
#!/usr/bin/env python3
# ===============================================================
# 🧬 Helix Continuum Scroll Minting Suite v2.0 (Batch + IPFS)
# ===============================================================
# Author: Leroy H. Mason
# Project: Helix-Phases | Aetherwatch Division | Class-17 Forge
# ---------------------------------------------------------------
# Dependencies:
#   pip install solana==0.30.2 solders==0.19.0 metaplex-foundation click requests
# ---------------------------------------------------------------
# Usage:
#   export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
#   export WALLET_PATH="/path/to/solana/wallet.json"
#   export NFT_STORAGE_TOKEN="your_nft_storage_api_key"
#   python helix_mint_batch.py --dir ./scrolls --symbol HELX
# ===============================================================

import os, json, hashlib, click, requests
from pathlib import Path
from datetime import datetime
from solders.keypair import Keypair
from solana.rpc.api import Client
from metaplex.foundation import Metaplex

# ---------------- CONFIG ----------------
RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
WALLET_PATH = os.getenv("WALLET_PATH", "./wallet.json")
NFT_STORAGE_TOKEN = os.getenv("NFT_STORAGE_TOKEN")

# ----------------------------------------
def sha256_file(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def upload_to_ipfs(file_path: Path) -> str:
    headers = {"Authorization": f"Bearer {NFT_STORAGE_TOKEN}"}
    with open(file_path, "rb") as f:
        r = requests.post("https://api.nft.storage/upload", headers=headers, files={"file": f})
    if r.status_code != 200:
        raise Exception(f"IPFS upload failed: {r.text}")
    cid = r.json()["value"]["cid"]
    return f"ipfs://{cid}"

@click.command()
@click.option("--dir", required=True, help="Directory with image and metadata JSON files.")
@click.option("--symbol", required=True, help="NFT symbol (e.g. HELX).")
@click.option("--network", default="mainnet-beta", help="Solana network (mainnet-beta/devnet).")
def mint_batch(dir, symbol, network):
    """Mint multiple NFTs from a folder of metadata."""
    client = Client(f"https://api.{network}.solana.com")
    with open(WALLET_PATH, "r") as f:
        secret = json.load(f)
    payer = Keypair.from_secret_key(bytes(secret))
    public_key = payer.pubkey()
    mx = Metaplex(client).use(payer)

    dir_path = Path(dir)
    results = []

    print(f"🚀 Starting Helix mint batch from {dir_path} ...\n")

    for img_file in dir_path.glob("*.png"):
        base_name = img_file.stem
        meta_file = dir_path / f"{base_name}.json"
        if not meta_file.exists():
            print(f"⚠️  No metadata for {img_file.name}, skipping.")
            continue

        # Upload assets to IPFS
        print(f"🔼 Uploading {img_file.name} to IPFS...")
        image_ipfs = upload_to_ipfs(img_file)
        print(f"✅ Uploaded image: {image_ipfs}")

        metadata = json.load(open(meta_file))
        metadata["image"] = image_ipfs
        metadata["content_hash"] = sha256_file(img_file)

        # Upload metadata JSON
        temp_json = Path(f"./_temp_{base_name}.json")
        with open(temp_json, "w") as f:
            json.dump(metadata, f, indent=2)
        meta_ipfs = upload_to_ipfs(temp_json)
        os.remove(temp_json)

        # Mint NFT
        print(f"🧬 Minting NFT: {metadata['name']} ...")
        result = mx.nfts().create(
            name=metadata["name"],
            uri=meta_ipfs,
            seller_fee_basis_points=0,
            symbol=symbol
        )

        record = {
            "name": metadata["name"],
            "mint_address": result["mint_address"],
            "uri": meta_ipfs,
            "hash": metadata["content_hash"],
            "timestamp": datetime.utcnow().isoformat()
        }
        results.append(record)
        print(f"✅ Minted: {record['mint_address']}\n")

    # Write log
    with open("helix_mint_log.json", "w") as logf:
        json.dump(results, logf, indent=2)
    print("📜 Minting complete. Log saved → helix_mint_log.json")

if __name__ == "__main__":
    mint_batch()
scrolls/
 ├── helix_scroll_phase1.png
 ├── helix_scroll_phase1.json
 ├── helix_scroll_phase2.png
 ├── helix_scroll_phase2.json
{
  "name": "Helix Continuum Scroll – Phase I",
  "symbol": "HELX",
  "description": "Sealed rollout artifact for the Helix-Phases generational framework.",
  "attributes": [
    {"trait_type": "phase", "value": "I"},
    {"trait_type": "author", "value": "Leroy H. Mason"},
    {"trait_type": "fractal", "value": true}
  ]
}
helixforge/
├── helixforge/
│   ├── __init__.py
│   ├── config.py
│   ├── ipfs_client.py
│   ├── solana_client.py
│   ├── metadata.py
│   ├── mint.py
│   ├── verify.py
│   ├── utils.py
├── scripts/
│   ├── helix_mint_batch.py
│   ├── helix_create_metadata.py
│   ├── helix_verify_hashes.py
├── samples/
│   ├── helix_scroll_phase1.png   # placeholder
│   ├── helix_scroll_phase1.json  # template
├── README.md
├── requirements.txt
└── .env.example
# helixforge/__init__.py
"""
Helix Forge – Sacred NFT rollout toolkit
Author: Leroy H. Mason (Helix-Phases)
"""
__version__ = "0.1.0"
# helixforge/config.py
import os
from pathlib import Path

# Solana
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
SOLANA_WALLET_PATH = os.getenv("WALLET_PATH", str(Path.home() / ".config" / "solana" / "id.json"))

# IPFS / NFT.Storage
NFT_STORAGE_TOKEN = os.getenv("NFT_STORAGE_TOKEN", "")

# Project defaults
DEFAULT_SYMBOL = os.getenv("HELIX_SYMBOL", "HELX")
DEFAULT_AUTHOR = os.getenv("HELIX_AUTHOR", "Leroy H. Mason")
DEFAULT_COLLECTION_NAME = os.getenv("HELIX_COLLECTION", "Helix Continuum Scrolls")
LOG_PATH = os.getenv("HELIX_LOG_PATH", "helix_mint_log.json")
# helixforge/utils.py
import hashlib
import json
from pathlib import Path
from datetime import datetime

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def write_log(path, records):
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

def utc_now_iso():
    return datetime.utcnow().isoformat()
# helixforge/ipfs_client.py
import requests
from pathlib import Path
from .config import NFT_STORAGE_TOKEN

NFT_STORAGE_ENDPOINT = "https://api.nft.storage/upload"

class IPFSClient:
    def __init__(self, token: str | None = None):
        self.token = token or NFT_STORAGE_TOKEN
        if not self.token:
            raise ValueError("NFT.Storage token missing. Set NFT_STORAGE_TOKEN env var.")

    def upload_file(self, file_path: Path) -> str:
        headers = {"Authorization": f"Bearer {self.token}"}
        with open(file_path, "rb") as f:
            r = requests.post(NFT_STORAGE_ENDPOINT, headers=headers, files={"file": f})
        if r.status_code != 200:
            raise RuntimeError(f"IPFS upload failed: {r.text}")
        cid = r.json()["value"]["cid"]
        return f"ipfs://{cid}"

    def upload_json(self, data: dict, tmp_name: str = "tmp_metadata.json") -> str:
        tmp = Path(tmp_name)
        tmp.write_text(json.dumps(data, indent=2))
        cid_uri = self.upload_file(tmp)
        tmp.unlink(missing_ok=True)
        return cid_uri
# helixforge/solana_client.py
import json
from solders.keypair import Keypair
from solana.rpc.api import Client
from metaplex.foundation import Metaplex  # assumes installed
from .config import SOLANA_RPC_URL, SOLANA_WALLET_PATH

def load_keypair(path: str) -> Keypair:
    with open(path, "r") as f:
        secret = json.load(f)
    return Keypair.from_secret_key(bytes(secret))

class SolanaHelixClient:
    def __init__(self, rpc_url: str = SOLANA_RPC_URL, wallet_path: str = SOLANA_WALLET_PATH):
        self.client = Client(rpc_url)
        self.payer = load_keypair(wallet_path)
        self.mx = Metaplex(self.client).use(self.payer)

    def mint_nft(self, name: str, uri: str, symbol: str, seller_fee_bps: int = 0):
        res = self.mx.nfts().create(
            name=name,
            uri=uri,
            symbol=symbol,
            seller_fee_basis_points=seller_fee_bps
        )
        return res
# helixforge/metadata.py
from datetime import datetime
from .config import DEFAULT_AUTHOR, DEFAULT_SYMBOL

BASE_TEMPLATE = {
    "name": "",
    "symbol": DEFAULT_SYMBOL,
    "description": "Sealed rollout artifact for the Helix-Phases generational framework.",
    "image": "",
    "attributes": [],
    "properties": {
        "files": [],
        "category": "image"
    },
    "created_at": "",
    "content_hash": ""
}

def build_metadata(name: str,
                   image_uri: str,
                   phase: str = "rollout",
                   author: str = DEFAULT_AUTHOR,
                   fractal: bool = True,
                   scroll: bool = True,
                   extra_attrs: list | None = None) -> dict:
    md = BASE_TEMPLATE.copy()
    md["name"] = name
    md["image"] = image_uri
    md["attributes"] = [
        {"trait_type": "phase", "value": phase},
        {"trait_type": "author", "value": author},
        {"trait_type": "timestamp", "value": datetime.utcnow().date().isoformat()},
        {"trait_type": "fractal", "value": fractal},
        {"trait_type": "scroll", "value": scroll},
    ]
    if extra_attrs:
        md["attributes"].extend(extra_attrs)
    md["properties"]["files"] = [{"uri": image_uri, "type": "image/png"}]
    md["created_at"] = datetime.utcnow().isoformat()
    return md
# helixforge/mint.py
from pathlib import Path
from .ipfs_client import IPFSClient
from .solana_client import SolanaHelixClient
from .utils import sha256_file, utc_now_iso
from .config import DEFAULT_SYMBOL

def mint_from_local(image_path: Path,
                    metadata: dict,
                    symbol: str = DEFAULT_SYMBOL,
                    seller_fee_bps: int = 0) -> dict:
    ipfs = IPFSClient()
    sol = SolanaHelixClient()

    # 1. upload image
    image_uri = ipfs.upload_file(image_path)

    # 2. fill metadata
    metadata["image"] = image_uri
    metadata["content_hash"] = sha256_file(image_path)
    if "created_at" not in metadata or not metadata["created_at"]:
        metadata["created_at"] = utc_now_iso()

    # 3. upload metadata
    md_uri = ipfs.upload_json(metadata, tmp_name=f"tmp_{image_path.stem}.json")

    # 4. mint NFT
    result = sol.mint_nft(name=metadata["name"], uri=md_uri, symbol=symbol, seller_fee_bps=seller_fee_bps)

    return {
        "name": metadata["name"],
        "mint_address": result["mint_address"],
        "uri": md_uri,
        "image_uri": image_uri,
        "hash": metadata["content_hash"],
        "timestamp": utc_now_iso()
    }
# helixforge/verify.py
from pathlib import Path
from .utils import sha256_file

def verify_local_hash(image_path: Path, expected_hash: str) -> bool:
    return sha256_file(image_path) == expected_hash
#!/usr/bin/env python3
import json
import click
from pathlib import Path
from helixforge.mint import mint_from_local
from helixforge.utils import write_log
from helixforge.config import LOG_PATH, DEFAULT_SYMBOL

@click.command()
@click.option("--dir", "dirpath", required=True, help="Folder containing .png + .json pairs.")
@click.option("--symbol", default=DEFAULT_SYMBOL, help="NFT symbol (default HELX).")
def main(dirpath, symbol):
    base = Path(dirpath)
    results = []

    print(f"🚀 Helix Forge batch mint starting → {base}")

    for img in base.glob("*.png"):
        meta_file = base / f"{img.stem}.json"
        if not meta_file.exists():
            print(f"⚠️  Skipping {img.name}, no matching JSON.")
            continue

        metadata = json.loads(meta_file.read_text())
        rec = mint_from_local(img, metadata, symbol=symbol)
        results.append(rec)
        print(f"✅ Minted: {rec['name']} @ {rec['mint_address']}")

    write_log(LOG_PATH, results)
    print(f"📜 All done. Log → {LOG_PATH}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import json
import click
from pathlib import Path
from helixforge.metadata import build_metadata

@click.command()
@click.option("--dir", "dirpath", required=True, help="Folder with PNGs.")
@click.option("--phase", default="rollout")
def main(dirpath, phase):
    base = Path(dirpath)
    for img in base.glob("*.png"):
        name = f"Helix Continuum Scroll – {img.stem}"
        md = build_metadata(name=name, image_uri="ipfs://TBD", phase=phase)
        out = base / f"{img.stem}.json"
        out.write_text(json.dumps(md, indent=2))
        print(f"🧾 wrote {out}")
    print("✅ metadata scaffold complete (image URIs will be filled on mint).")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import json
import click
from pathlib import Path
from helixforge.verify import verify_local_hash

@click.command()
@click.option("--log", "log_path", default="helix_mint_log.json", help="Mint log to verify.")
@click.option("--dir", "dirpath", required=True, help="Dir with original images.")
def main(log_path, dirpath):
    log = json.loads(Path(log_path).read_text())
    base = Path(dirpath)
    all_ok = True

    for entry in log:
        img_name = entry["name"].split("–")[-1].strip().replace(" ", "_").lower() + ".png"
        # If names don't align, you can map manually.
        candidates = list(base.glob("*.png"))
        matched = None
        for c in candidates:
            if c.stem in entry["uri"] or c.stem in entry["name"].lower():
                matched = c
                break
        if not matched:
            print(f"⚠️ no local file for {entry['name']}, skip")
            continue

        ok = verify_local_hash(matched, entry["hash"])
        print(f"{entry['name']}: {'✅' if ok else '❌'}")
        if not ok:
            all_ok = False

    if all_ok:
        print("🎉 All hashes verified.")
    else:
        print("⚠️ One or more assets failed verification.")

if __name__ == "__main__":
    main()
{
  "name": "Helix Continuum Scroll",
  "symbol": "HELX",
  "description": "Sealed rollout artifact for the Helix-Phases generational framework.",
  "image": "ipfs://TBD",
  "attributes": [
    { "trait_type": "phase", "value": "rollout" },
    { "trait_type": "author", "value": "Leroy H. Mason" },
    { "trait_type": "timestamp", "value": "2025-11-02" },
    { "trait_type": "fractal", "value": true },
    { "trait_type": "scroll", "value": true }
  ],
  "properties": {
    "files": [{ "uri": "ipfs://TBD", "type": "image/png" }],
    "category": "image"
  },
  "created_at": "2025-11-02T14:40:00Z",
  "content_hash": ""
}
# Helix Forge 🧬
Sacred NFT rollout toolkit for the **Helix-Phases** continuum by **Leroy H. Mason**.

This repo lets you:
- create scroll metadata
- upload art + metadata to IPFS (NFT.Storage)
- mint to Solana via Metaplex
- verify the rollout with an integrity log

## Prereqs
- Python 3.10+
- Solana CLI installed + wallet
- `pip install -r requirements.txt`
- env:
  ```bash
  export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
  export WALLET_PATH="$HOME/.config/solana/id.json"
  export NFT_STORAGE_TOKEN="your_nft_storage_token"
scrolls/
  helix_scroll_phase1.png
  helix_scroll_phase2.png
python scripts/helix_create_metadata.py --dir ./scrolls --phase rollout
python scripts/helix_mint_batch.py --dir ./scrolls --symbol HELX
python scripts/helix_verify_hashes.py --dir ./scrolls
export SOLANA_RPC_URL="https://api.devnet.solana.com"

---

## 13) `requirements.txt`

```text
solana==0.30.2
solders==0.19.0
metaplex-foundation==0.1.0
click==8.1.7
requests==2.32.3
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
WALLET_PATH=/home/leroy/.config/solana/id.json
NFT_STORAGE_TOKEN=replace_me
HELIX_SYMBOL=HELX
HELIX_AUTHOR="Leroy H. Mason"
HELIX_COLLECTION="Helix Continuum Scrolls"
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
WALLET_PATH=/home/leroy/.config/solana/id.json
NFT_STORAGE_TOKEN=replace_me
HELIX_SYMBOL=HELX
HELIX_AUTHOR="Leroy H. Mason"
HELIX_COLLECTION="Helix Continuum Scrolls"
#!/usr/bin/env python3
"""
🧬 Helix All-In-One Mint Script
Author: Leroy H. Mason
Lineage: Helix-Phases → Helix Continuum Scrolls (HELX)
Wallet: Hw2Cd7qsVFf3RLERQyinRKhizugYvKXccLfdiiETMNa5

What it does (single run):
1. reads local PNG
2. uploads PNG to IPFS (NFT.Storage)
3. builds metadata (with fractal + scroll flags)
4. uploads metadata to IPFS
5. mints NFT on Solana (Metaplex)
6. writes helix_mint_log.json
"""

import os
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime

# --- you need these packages installed ---
# pip install solana==0.30.2 solders==0.19.0 metaplex-foundation requests

from s
name: Helix Mint (CI)

on:
  workflow_dispatch:  # run manually
  push:
    paths:
      - "helix_mint.py"
      - ".github/workflows/helix-mint.yml"

jobs:
  mint:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
            python-version: "3.11"

      - name: Install deps
        run: |
          pip install solana==0.30.2 solders==0.19.0 metaplex-foundation requests

      - name: Create assets dir
        run: mkdir -p assets

      - name: Download or place scroll image
        run: |
          # Option 1: fetch from your own URL
          curl -L https://example.com/helix_scroll.png -o assets/helix_scroll.png
          # You can replace this with wget from your site or commit the image directly

      - name: Run Helix Mint
        env:
          SOLANA_RPC_URL: ${{ secrets.SOLANA_RPC_URL }}
          WALLET_JSON: ${{ secrets.WALLET_JSON }}
          NFT_STORAGE_TOKEN: ${{ secrets.NFT_STORAGE_TOKEN }}
          HELIX_IMAGE_PATH: assets/helix_scroll.png
          HELIX_NAME: "Helix Continuum Scroll – CI Rollout"
          HELIX_PHASE: "rollout"
          HELIX_SYMBOL: "HELX"
        run: |
          python helix_mint.py
SOLANA_RPC_URL: https://api.devnet.solana.com
name: Helix Auto Mint – Scroll Watcher

on:
  push:
    paths:
      - "scrolls/*.png"
      - "scrolls/*.json"
  workflow_dispatch:  # manual trigger option

jobs:
  mint-scrolls:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        u
helixforge/
 ├── helix_mint.py
 ├── scrolls/
 │    ├── helix_scroll_phase1.png
 │    ├── helix_scroll_phase2.png
 │    ├── helix_scroll_phase2.json   # optional metadata
 └── .github/
      └── workflows/
           ├── helix-mint.yml
   🌀 Helix Auto Mint Starting...
🔮 Found new scroll: helix_scroll_phase1
🔼 Uploading image to IPFS...
✅ Image IPFS: ipfs://bafybeia12...
🔼 Uploading metadata to IPFS...
✅ Metadata IPFS: ipfs://bafybeid34...
🧬 Minting on Solana…
✅ MINTED! Address: 6MFi7Pp...
📜 Logged → helix_mint_log.json
✅ All detected scrolls processed.
        └name: Helix Auto Mint + Ledger Commit

on:
  push:
    paths:
      - "scrolls/*.png"
      - "scrolls/*.json"
  workflow_dispatch:  # manual trigger option

jobs:
  mint-and-commit:
    runs-on: ubuntu-latest

    permissions:
      contents: write    # allow the workflow to commit the log file
      id-token: write

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
        with:
          persist-credentials: false

      - name: Confi
──🌀 Helix Auto Mint (Ledger Mode) starting...
🔮 Processing: helix_scroll_phase3
🔼 Uploading image to IPFS...
✅ Metadata IPFS: ipfs://bafybeid34...
🧬 Minting on Solana…
✅ MINTED! Address: 9HZtSgD...
📜 Logged → helix_mint_log.json
📜 Committing updated mint ledger...
[main 2c9b18d] 🔏 Helix Forge Ledger Update 2025-11-02 20:41:00 UTC
🔥 Helix Forge Ledger sync complete.
helixforge/
 ├── helix_mint.py
 ├── scrolls/
 │    ├── helix_scroll_phase1.png
 │    ├── helix_scroll_phase2.png
 │    └── helix_scroll_phase3.png
 ├── helix_mint_log.json
 ├── minted_logs/
 │    ├── helix_mint_log_20251102_204100.json
 │    └── helix_mint_log_20251103_112400.json
 └── .github/workflows/
      └── helix-auto-mint-ledger.yml

