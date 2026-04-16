"""
prepare_anchors.py — Set up the anchor image library for semantic_inversion.py.

What this script does
---------------------
1. Creates the anchors/ directory (default: semantic_init/anchors/).
2. For each category, generates one image via OpenRouter (OpenAI-compatible chat API
   + image modality), using prompts tailored to the category. Any failure raises
   (no silent fallback to synthetic).
3. With --synthetic-only only: writes PIL synthetic placeholders (no API, no key).

10 default categories
---------------------
  animal, building, vehicle, nature, food, sport, person, technology, art, indoor

Environment
-----------
  OPENROUTER_API_KEY or OPENAI_API_KEY — required for API generation (unless --synthetic-only).

Usage
-----
    export OPENROUTER_API_KEY=sk-or-...
    python prepare_anchors.py
    python prepare_anchors.py --anchors-dir /my/path
    python prepare_anchors.py --synthetic-only   # no network / no API
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import os
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# ─── Per-category image prompts (OpenRouter / Gemini image preview) ─────────
_PROMPTS: dict[str, str] = {
    "animal": (
        "Photorealistic close-up of a friendly dog in natural daylight, shallow depth of field."
    ),
    "building": (
        "Photorealistic modern city skyline with glass skyscrapers at golden hour."
    ),
    "vehicle": (
        "Photorealistic silver sedan car, three-quarter front view, clean studio-like lighting."
    ),
    "nature": (
        "Photorealistic mountain landscape at sunset, layered peaks, dramatic sky."
    ),
    "food": (
        "Photorealistic overhead shot of a colorful healthy meal on a wooden table."
    ),
    "sport": (
        "Photorealistic outdoor soccer field with green grass and white lines, sunny day."
    ),
    "person": (
        "Photorealistic portrait of a smiling adult, soft natural window light, neutral background."
    ),
    "technology": (
        "Photorealistic macro of a circuit board with chips and copper traces, cool lighting."
    ),
    "art": (
        "Oil painting style still life with fruit and vase, rich brushstrokes, museum quality."
    ),
    "indoor": (
        "Photorealistic cozy modern living room interior, warm ambient light, wide angle."
    ),
}

# ─── Synthetic fallback images ────────────────────────────────────────────────
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
    """Generate a simple 256x256 synthetic image with gradient + pattern."""
    img = Image.new("RGB", (SIZE, SIZE), bg)
    draw = ImageDraw.Draw(img)

    for y in range(SIZE):
        t = y / SIZE
        r = int(bg[0] * (1 - t) + stripe[0] * t)
        g = int(bg[1] * (1 - t) + stripe[1] * t)
        b = int(bg[2] * (1 - t) + stripe[2] * t)
        draw.line([(0, y), (SIZE, y)], fill=(r, g, b))

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
        for r_val in range(10, SIZE // 2, 20):
            draw.ellipse(
                [SIZE // 2 - r_val, SIZE // 2 - r_val,
                 SIZE // 2 + r_val, SIZE // 2 + r_val],
                outline=stripe, width=3,
            )

    return img.filter(ImageFilter.GaussianBlur(radius=2))


def _decode_data_url(data_url: str) -> bytes:
    """Decode data:image/...;base64,... payload."""
    comma = data_url.find(",")
    if comma == -1:
        raise ValueError("data URL missing comma separator")
    payload = data_url[comma + 1 :].strip()
    # URL-safe base64 sometimes uses - _ ; standard library handles padding
    try:
        return base64.b64decode(payload, validate=False)
    except binascii.Error as exc:
        raise ValueError(f"invalid base64 in data URL: {exc}") from exc


def _pil_from_openrouter_message(msg: object, *, category: str) -> Image.Image:
    """Extract first RGB image from assistant message; raises if missing or invalid."""
    images = getattr(msg, "images", None)
    if not images:
        raise RuntimeError(
            f"{category}: assistant message has no .images (model may not support image output)."
        )

    last_decode_err: Exception | None = None
    for image in images:
        url: str | None = None
        if isinstance(image, dict):
            iu = image.get("image_url")
            if isinstance(iu, dict):
                url = iu.get("url")
            elif isinstance(iu, str):
                url = iu
        else:
            iu = getattr(image, "image_url", None)
            if isinstance(iu, dict):
                url = iu.get("url")
            elif hasattr(iu, "url"):
                url = getattr(iu, "url", None)

        if not url or not isinstance(url, str):
            continue
        if url.startswith("data:"):
            try:
                raw = _decode_data_url(url)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as exc:
                last_decode_err = exc
                continue
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"{category}: HTTP {exc.code} while fetching image URL."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{category}: failed to fetch or decode image URL: {url[:120]!r}..."
            ) from exc

    if last_decode_err is not None:
        raise RuntimeError(
            f"{category}: could not decode any data URL in .images."
        ) from last_decode_err
    raise RuntimeError(
        f"{category}: .images present but no usable image_url url (empty or wrong shape)."
    )


def _generate_via_openrouter(
    *,
    category: str,
    prompt: str,
    base_url: str,
    model: str,
    api_key: str,
) -> Image.Image:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is required for API anchor generation (pip install openai)."
        ) from exc

    client = OpenAI(base_url=base_url, api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"modalities": ["image", "text"]},
    )
    msg = completion.choices[0].message
    return _pil_from_openrouter_message(msg, category=category)


def prepare(
    anchors_dir: Path,
    *,
    synthetic_only: bool = False,
    openrouter_base_url: str,
    openrouter_model: str,
) -> None:
    anchors_dir.mkdir(parents=True, exist_ok=True)
    print(f"Anchor directory: {anchors_dir.resolve()}\n")

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not synthetic_only:
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY or OPENAI_API_KEY must be set for API generation, "
                "or pass --synthetic-only for local PIL anchors."
            )
        api_key_str: str = api_key

    for cat in _SYNTHETIC:
        out_path = anchors_dir / f"{cat}.jpg"
        if out_path.exists():
            print(f"  {cat:12s}  already exists, skipping")
            continue

        if synthetic_only:
            bg, stripe, desc = _SYNTHETIC[cat]
            print(f"  {cat:12s}  synthetic ({desc})")
            img = _make_synthetic(bg, stripe, cat)
        else:
            print(f"  {cat:12s}  generating via OpenRouter ...")
            img = _generate_via_openrouter(
                category=cat,
                prompt=_PROMPTS[cat],
                base_url=openrouter_base_url,
                model=openrouter_model,
                api_key=api_key_str,
            )
            print(f"    got image ({img.size[0]}x{img.size[1]})")

        img.save(out_path, quality=92)
        print(f"    saved → {out_path}")

    print(f"\nDone. {len(list(anchors_dir.glob('*.jpg')))} anchor images ready.")
    print(
        "\nTip: replace any .jpg in anchors/ with your own photos for stronger semantic matching."
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
        help="Skip OpenRouter; use PIL synthetic images only",
    )
    parser.add_argument(
        "--openrouter-base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--openrouter-model",
        default="google/gemini-3.1-flash-image-preview",
        help="Model id on OpenRouter that supports image output",
    )
    args = parser.parse_args()

    anchors_dir = Path(args.anchors_dir)
    if not anchors_dir.is_absolute():
        anchors_dir = Path(__file__).parent / anchors_dir

    prepare(
        anchors_dir,
        synthetic_only=args.synthetic_only,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_model=args.openrouter_model,
    )


if __name__ == "__main__":
    main()
