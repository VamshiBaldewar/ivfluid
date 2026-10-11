"""Download a public IV fill-level baseline (Mendeley SESOVERA saline bottles, CC BY 4.0).

This is NOT the Roboflow/IEEE sets (those need an account). It is a real labeled
saline-bottle fill-level set: empty / 50% / 80% / 100%.

Default: 30 images per class, resized to 640px, ~few hundred MB max.
Full original dump is ~13 GB of 3456px JPEGs — use --per-class 0 for every file
and --no-resize only if you have disk and time.

  python -m tools.download_m1_public
  python -m tools.download_m1_public --per-class 80
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from app.paths import DATA_RAW, PROJECT_ROOT

UA = {"User-Agent": "IVFluid-student-project/0.1 (research; CC-BY dataset fetch)"}

# folder_id -> class label (counts match Simonetta & Pau 2020 Data in Brief)
DATASETS = {
    "9mcj3rvvxb": {
        "9ea47d96-c15a-497d-a527-ba50a687cfa4": "0",
        "98ba848c-eb39-43bf-9941-a8ae881df319": "50",
    },
    "n8k2zfr6xm": {
        "fee1459e-e301-43d2-a5f2-ec289df38171": "80",
        "52f5caf9-41f2-4044-a81c-9abcc8a340ed": "100",
    },
}

SOURCE_NOTE = """\
M1 public baseline — SESOVERA-ST saline bottle fill-level (CC BY 4.0)
Simonetta L., Pau D. Dataset of sodium chloride sterile liquid in bottles
for intravenous administration and fill-level monitoring. Data in Brief, 2020.
https://doi.org/10.1016/j.dib.2020.106394
https://data.mendeley.com/datasets/9mcj3rvvxb
https://data.mendeley.com/datasets/n8k2zfr6xm
Classes: 0, 50, 80, 100 (percent fill). Bottles, not IV bags — fine-tune on your rig later.
"""


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _download_one(url: str, dest: Path, max_side: int | None) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    if max_side:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            dest.write_bytes(raw)
            return "raw"
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        dest.write_bytes(buf.tobytes() if ok else raw)
        return "ok"
    dest.write_bytes(raw)
    return "ok"


def fetch_file_list() -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = []
    for ds_id, folder_map in DATASETS.items():
        meta = _get_json(f"https://data.mendeley.com/public-api/datasets/{ds_id}")
        for f in meta.get("files", []):
            label = folder_map.get(f.get("folder_id"))
            if not label:
                continue
            url = f["content_details"]["download_url"]
            jobs.append((label, f["filename"], url))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=30, help="0 = download every image")
    parser.add_argument("--max-side", type=int, default=640, help="0 = keep original resolution")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    max_side = args.max_side if args.max_side > 0 else None

    dest_root = DATA_RAW / "fluid_level" / "public_mendeley"
    print("Listing Mendeley files…")
    jobs = fetch_file_list()
    by_label: dict[str, list[tuple[str, str, str]]] = {}
    for item in jobs:
        by_label.setdefault(item[0], []).append(item)
    selected: list[tuple[str, str, str]] = []
    for label, items in sorted(by_label.items()):
        items = sorted(items, key=lambda x: x[1])
        take = items if args.per_class <= 0 else items[: args.per_class]
        print(f"  class {label}%: {len(take)} / {len(items)}")
        selected.extend(take)

    (dest_root / "SOURCE.txt").parent.mkdir(parents=True, exist_ok=True)
    (dest_root / "SOURCE.txt").write_text(SOURCE_NOTE, encoding="utf-8")

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for label, name, url in selected:
            path = dest_root / label / name
            futs[pool.submit(_download_one, url, path, max_side)] = path
        n = len(futs)
        for i, fut in enumerate(as_completed(futs), 1):
            path = futs[fut]
            try:
                status = fut.result()
                if status == "skip":
                    skip += 1
                else:
                    ok += 1
            except Exception as exc:
                fail += 1
                print(f"  FAIL {path.name}: {exc}")
            if i % 10 == 0 or i == n:
                print(f"  {i}/{n} done (saved {ok}, skipped {skip}, failed {fail})")

    print(f"Finished -> {dest_root.relative_to(PROJECT_ROOT)}")
    print(f"saved={ok} skipped={skip} failed={fail}")


if __name__ == "__main__":
    main()
