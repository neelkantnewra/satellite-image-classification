# ============================================================
# Part 1: Dataset Preparation
# ============================================================

import bz2, json, os, random, shutil, subprocess
from pathlib import Path

random.seed(42)

BUCKET  = "s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb"
OUT_DIR = Path("fmow_data")
N_EACH  = 700

URBAN = [
    "multi-unit_residential",
    "single-unit_residential",
    "office_building",
    "shopping_mall",
    "parking_lot_or_garage",
    "interchange",
    "road_bridge",
    "ground_transportation_station",
]

NATURAL = [
    "crop_field",
    "lake_or_pond",
    "park",
    "aquaculture",
    "golf_course",
    "wind_farm",
    "surface_mine",
    "forest",
]

# ── Clean slate ───────────────────────────────────────────────
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

for folder in [
    "urban", "natural", "external",
    "splits/train/urban", "splits/train/natural",
    "splits/test/urban",  "splits/test/natural",
]:
    (OUT_DIR / folder).mkdir(parents=True, exist_ok=True)

# ── Step 1: Download manifest ─────────────────────────────────
print("Downloading manifest...")
subprocess.run([
    "aws", "s3", "cp",
    f"{BUCKET}/manifest.json.bz2",
    str(OUT_DIR / "manifest.json.bz2"),
    "--no-sign-request"
], check=True)

# ── Step 2: Parse manifest ────────────────────────────────────
print("Parsing manifest...")
with bz2.open(OUT_DIR / "manifest.json.bz2", "rt") as f:
    manifest = json.load(f)

cat_files = {}
for e in manifest:
    if not e.endswith("_rgb.jpg"):
        continue
    parts = e.split("/")
    if len(parts) >= 2 and parts[0] == "train":
        cat_files.setdefault(parts[1], []).append(e)

def get_sample(categories, n):
    all_files = []
    for cat in categories:
        all_files.extend(cat_files.get(cat, []))
    return random.sample(all_files, min(n, len(all_files)))

# 701 urban (1 for external) + 700 natural
urban_sample   = get_sample(URBAN,   N_EACH + 1)
natural_sample = get_sample(NATURAL, N_EACH)

external_img  = urban_sample.pop()   # remove 1 → urban_sample = exactly 700

print(f"Urban:    {len(urban_sample)} images")
print(f"Natural:  {len(natural_sample)} images")
print(f"External: 1 image")

# ── Step 3: Download (20 parallel) ───────────────────────────
def batch_download(file_list, dest_dir, label):
    print(f"\nDownloading {label}...")
    procs = []
    for i, path in enumerate(file_list):
        fname = path.split("/")[-1]          # just the filename, no path prefix
        out   = dest_dir / fname
        p = subprocess.Popen([
            "aws", "s3", "cp",
            f"{BUCKET}/{path}", str(out),
            "--no-sign-request", "--quiet"
        ])
        procs.append(p)
        if len(procs) >= 20:
            for p in procs: p.wait()
            print(f"  {len(list(dest_dir.glob('*.jpg')))}/{len(file_list)}")
            procs = []
    for p in procs: p.wait()
    total = len(list(dest_dir.glob("*.jpg")))
    print(f"  ✅ {label}: {total} images")
    assert total == len(file_list), f"Expected {len(file_list)}, got {total}"

batch_download(urban_sample,   OUT_DIR / "urban",   "urban")
batch_download(natural_sample, OUT_DIR / "natural", "natural")

# External
ext_fname = external_img.split("/")[-1]
subprocess.run([
    "aws", "s3", "cp",
    f"{BUCKET}/{external_img}",
    str(OUT_DIR / "external" / ext_fname),
    "--no-sign-request", "--quiet"
], check=True)
print("✅ External image saved.")

# ── Step 4: 80/20 Split (400 train, 100 test) ─────────────────
print("\nCreating 80/20 split...")
for label in ["urban", "natural"]:
    files = sorted((OUT_DIR / label).glob("*.jpg"))
    assert len(files) == N_EACH, f"Expected {N_EACH}, got {len(files)}"
    random.shuffle(files)

    train_files = files[:N_EACH//5*4]  # first 80% → 560
    test_files  = files[N_EACH//5*4:N_EACH] # last 20% → 140

    for f in train_files:
        shutil.copy(f, OUT_DIR / "splits" / "train" / label / f.name)
    for f in test_files:
        shutil.copy(f, OUT_DIR / "splits" / "test"  / label / f.name)

# ── Summary ───────────────────────────────────────────────────
size = sum(f.stat().st_size for f in OUT_DIR.rglob("*.jpg")) / 1024 / 1024

print(f"""
✅ Part 1 Complete!

fmow_data/
├── urban/               ({len(list((OUT_DIR/'urban').glob('*.jpg')))} images)
├── natural/             ({len(list((OUT_DIR/'natural').glob('*.jpg')))} images)
├── external/            ({len(list((OUT_DIR/'external').glob('*.jpg')))} image)
└── splits/
    ├── train/urban/     ({len(list((OUT_DIR/'splits/train/urban').glob('*.jpg')))} images)
    ├── train/natural/   ({len(list((OUT_DIR/'splits/train/natural').glob('*.jpg')))} images)
    ├── test/urban/      ({len(list((OUT_DIR/'splits/test/urban').glob('*.jpg')))} images)
    └── test/natural/    ({len(list((OUT_DIR/'splits/test/natural').glob('*.jpg')))} images)

Total size: {size:.0f} MB
""")