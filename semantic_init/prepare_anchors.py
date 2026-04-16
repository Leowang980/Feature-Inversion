"""
prepare_anchors.py — Set up the anchor image library for semantic_inversion.py.

What this script does
---------------------
1. Creates the anchors/ directory (default: semantic_init/anchors/).
2. Tries to download one representative image per category from Wikimedia Commons
   (stable, freely-licensed HTTPS URLs).
3. If a download fails, generates a synthetic fallback image using PIL
   (characteristic colour + simple gradient pattern).

10 default categories
---------------------
  animal, building, vehicle, nature, food, sport, person, technology, art, indoor

Replace any image in anchors/ with a real photograph of your choice.
The filename stem (e.g. "animal") becomes the category label in the output.

Usage
-----
    python prepare_anchors.py                          # creates anchors/ in this folder
    python prepare_anchors.py --anchors-dir /my/path   # custom path
    python prepare_anchors.py --synthetic-only          # skip downloads, pure PIL
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# ─── Download URLs ────────────────────────────────────────────────────────────
# Wikimedia Commons 400 px thumbnails (stable permanent URLs, CC-licensed).
_URLS: dict[str, str] = {
    "animal": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/"
        "YellowLabradorLooking_new.jpg/400px-YellowLabradorLooking_new.jpg"
    ),
    "building": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/"
        "Tour_Eiffel_Wikimedia_Commons.jpg/400px-Tour_Eiffel_Wikimedia_Commons.jpg"
    ),
    "vehicle": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/"
        "2018_Volkswagen_Golf_Match_1.0_TSI_Front.jpg/"
        "400px-2018_Volkswagen_Golf_Match_1.0_TSI_Front.jpg"
    ),
    "nature": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/"
        "Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/"
        "400px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg"
    ),
    "food": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/"
        "Good_Food_Display_-_NCI_Visuals_Online.jpg/"
        "400px-Good_Food_Display_-_NCI_Visuals_Online.jpg"
    ),
    "sport": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/"
        "Sunrise_over_the_sea.jpg/"
        "400px-Sunrise_over_the_sea.jpg"
    ),
    "person": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/"
        "Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/"
        "400px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
    ),
    "technology": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
        "Bigl%C3%B6tkolben.jpg/400px-Bigl%C3%B6tkolben.jpg"
    ),
    "art": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/"
        "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/"
        "400px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
    ),
    "indoor": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/"
        "Interior_design_example.jpg/400px-Interior_design_example.jpg"
    ),
}

# ─── Synthetic fallback images ────────────────────────────────────────────────
# Each entry: (background_RGB, stripe_RGB, description)
_SYNTHETIC: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], str]] = {
    "animal":     ((180, 120,  60), (220, 170, 100), "warm brown gradient (fur-like)"),
    "building":   ((160, 160, 160), (200, 200, 200), "grey vertical stripes (facade-like)"),
    "vehicle":    ((180, 180, 190), (100, 120, 160), "silver-blue (car-like)"),
    "nature":     (( 60, 130,  60), ( 80, 160, 200), "green-blue gradient (landscape)"),
    "food":       ((230, 160,  50), (240, 210, 100), "warm orange-yellow (food-like)"),
    "sport":      (( 50, 140,  50), (230, 230, 230), "green field with white lines"),
    "person":     ((210, 170, 140), (240, 200, 170), "skin-tone gradient (portrait)"),
    "technology": (( 30,  30,  50), ( 50, 100, 200), "dark with blue highlights (tech)"),
    "art":        ((180,  80, 160), (240, 200,  60), "vivid contrast (painting-like)"),
    "indoor":     ((150, 110,  80), (200, 170, 130), "warm wood-brown (interior)"),
}

SIZE = 256  # pixels for synthetic images


def _make_synthetic(
    bg: tuple[int, int, int],
    stripe: tuple[int, int, int],
    category: str,
) -> Image.Image:
    """Generate a simple 256×256 synthetic image with gradient + pattern."""
    img = Image.new("RGB", (SIZE, SIZE), bg)
    draw = ImageDraw.Draw(img)

    # Horizontal gradient overlay
    for y in range(SIZE):
        t = y / SIZE
        r = int(bg[0] * (1 - t) + stripe[0] * t)
        g = int(bg[1] * (1 - t) + stripe[1] * t)
        b = int(bg[2] * (1 - t) + stripe[2] * t)
        draw.line([(0, y), (SIZE, y)], fill=(r, g, b))

    # Add a few characteristic shapes
    if category == "building":
        for x in range(0, SIZE, 32):
            draw.rectangle([x + 4, SIZE // 4, x + 20, SIZE - 8], fill=stripe)
    elif category == "sport":
        draw.line([(0, SIZE // 2), (SIZE, SIZE // 2)], fill=stripe, width=4)
        draw.line([(SIZE // 2, 0), (SIZE // 2, SIZE)], fill=stripe, width=4)
    elif category == "technology":
        for i in range(0, SIZE, 16):
            draw.rectangle([i, i, i + 8, i + 8], outline=stripe)
    elif category == "art":
        # Concentric circles (Van Gogh-ish swirl feel)
        for r_val in range(10, SIZE // 2, 20):
            draw.ellipse(
                [SIZE // 2 - r_val, SIZE // 2 - r_val,
                 SIZE // 2 + r_val, SIZE // 2 + r_val],
                outline=stripe, width=3,
            )

    # Slight blur to remove pixelation
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def _download(url: str, timeout: int = 15) -> Image.Image | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        print(f"    download failed: {exc}")
        return None


def prepare(anchors_dir: Path, synthetic_only: bool = False) -> None:
    anchors_dir.mkdir(parents=True, exist_ok=True)
    print(f"Anchor directory: {anchors_dir.resolve()}\n")

    for cat in _SYNTHETIC:   # iterate in a fixed order
        out_path = anchors_dir / f"{cat}.jpg"
        if out_path.exists():
            print(f"  {cat:12s}  already exists, skipping")
            continue

        img: Image.Image | None = None
        if not synthetic_only and cat in _URLS:
            print(f"  {cat:12s}  downloading …")
            img = _download(_URLS[cat])
            if img is not None:
                print(f"    downloaded ({img.size[0]}×{img.size[1]})")

        if img is None:
            bg, stripe, desc = _SYNTHETIC[cat]
            print(f"  {cat:12s}  generating synthetic ({desc})")
            img = _make_synthetic(bg, stripe, cat)

        img.save(out_path, quality=92)
        print(f"    saved → {out_path}")

    print(f"\nDone. {len(list(anchors_dir.glob('*.jpg')))} anchor images ready.")
    print(
        "\nTip: replace any .jpg in the anchors/ folder with a real photograph "
        "for better semantic matching."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up anchor image library for semantic_inversion.py"
    )
    parser.add_argument(
        "--anchors-dir", default="anchors",
        help="Where to save anchor images (default: anchors/ next to this script)",
    )
    parser.add_argument(
        "--synthetic-only", action="store_true",
        help="Skip downloads; generate synthetic placeholder images only",
    )
    args = parser.parse_args()

    anchors_dir = Path(args.anchors_dir)
    if not anchors_dir.is_absolute():
        anchors_dir = Path(__file__).parent / anchors_dir

    prepare(anchors_dir, synthetic_only=args.synthetic_only)


if __name__ == "__main__":
    main()
