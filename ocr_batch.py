"""
Batched OCR using ChandraOCR (mlx-community/chandra-bf16) on Apple Silicon.

This script performs efficient batched OCR inference using the ChandraOCR model
via mlx-vlm's batch_generate function. It processes images in batches, grouping
images by shape for optimal performance.

Features:
- Batch processing for high throughput on Apple Silicon
- Recursive folder scanning or manifest file input
- CSV output with resume capability
- Individual text file output per image

Usage:
  # From a directory (all images recursively)
  python ocr_batch.py --images-dir /path/to/images --output results.csv

  # From a manifest file (one image path per line)
  python ocr_batch.py --manifest paths.txt --output results.csv

  # Options
  python ocr_batch.py --images-dir /path/to/images --output results.csv --batch-size 8 --resume --verbose
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import time
from typing import List, Optional

from mlx_vlm import load, batch_generate

from prompt import PROMPT_MAPPING

MODEL_ID = "mlx-community/chandra-bf16"
DEFAULT_PROMPT = PROMPT_MAPPING["ocr"]


def collect_image_paths(
    images_dir: Optional[Path], manifest: Optional[Path]
) -> List[Path]:
    """Collect image paths from directory (recursive) or manifest file."""
    exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    paths: List[Path] = []

    if images_dir is not None:
        images_dir = Path(images_dir)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {images_dir}")
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p.resolve())
    elif manifest is not None:
        manifest = Path(manifest)
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        with open(manifest, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(Path(line).resolve())
    else:
        raise ValueError("Provide either --images-dir or --manifest")

    return sorted(paths)


def already_done_paths(output_csv: Path) -> set:
    """Set of image paths already in output CSV (for --resume)."""
    done = set()
    if not output_csv.exists():
        return done
    with open(output_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_path" not in (reader.fieldnames or []):
            return done
        for row in reader:
            p = row.get("image_path", "")
            if p:
                done.add(p)
    return done


def run_batched(
    image_paths: List[Path],
    output_csv: Path,
    prompt: str,
    max_tokens: int,
    batch_size: int,
    save_every: int,
    resume: bool,
    verbose: bool,
    max_size: Optional[int],
) -> None:
    """Process images in batches using mlx_vlm.batch_generate; save incrementally."""
    print(f"Loading model: {MODEL_ID}")
    model, processor = load(MODEL_ID)
    print("Model loaded successfully.")

    done = already_done_paths(output_csv) if resume else set()
    pending = [p for p in image_paths if str(p) not in done]
    if not pending and resume:
        print("All images already in output; nothing to do.")
        return
    if resume and len(done) > 0:
        print(f"Resuming: {len(done)} already in output, {len(pending)} to process.")

    write_header = not (resume and output_csv.exists())
    total_written = 0

    kwargs = {"verbose": verbose, "group_by_shape": True}
    if max_size is not None:
        kwargs["resize_shape"] = (max_size, max_size)

    # Open CSV file for writing results
    csv_file = open(output_csv, "a" if resume else "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["image_path", "status", "ocr_text"])
    if write_header:
        csv_writer.writeheader()

    batch_start_time = time()

    try:
        for start in range(0, len(pending), batch_size):
            chunk_paths = pending[start : start + batch_size]
            image_path_strs = [str(p) for p in chunk_paths]
            prompts = [prompt] * len(chunk_paths)

            try:
                response = batch_generate(
                    model,
                    processor,
                    images=image_path_strs,
                    prompts=prompts,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                texts = response.texts or [""] * len(chunk_paths)
                if len(texts) < len(chunk_paths):
                    texts = list(texts) + [""] * (len(chunk_paths) - len(texts))
                batch_error = None
            except Exception as e:
                batch_error = f"error:{type(e).__name__}:{e}"
                if verbose:
                    print(f"Batch error: {batch_error}")
                texts = [""] * len(chunk_paths)
                response = None

            for path_str, text in zip(image_path_strs, texts):
                if batch_error is not None:
                    status = batch_error
                elif text and text.strip():
                    status = "ok"
                else:
                    status = "error:empty"

                text_clean = (text or "").strip()

                # Write individual OCR result to text file
                output_txt_path = Path(path_str).with_suffix("").as_posix() + "_ocr.txt"
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(text_clean)

                # Write to CSV
                csv_writer.writerow({
                    "image_path": path_str,
                    "status": status,
                    "ocr_text": text_clean,
                })

                elapsed = time() - batch_start_time
                print(f"{path_str} DONE (elapsed: {elapsed:.1f}s)")
                total_written += 1

            # Flush CSV periodically
            csv_file.flush()

            if (start + len(chunk_paths)) % save_every == 0 or start + len(chunk_paths) >= len(
                pending
            ):
                print(f"Progress: {total_written}/{len(pending)}")

            if response is not None and response.stats and verbose:
                print(
                    f"  Prompt: {response.stats.prompt_tps:.1f} tok/s, "
                    f"Generation: {response.stats.generation_tps:.1f} tok/s"
                )
    finally:
        csv_file.close()

    print(f"Done. Results saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Batched OCR with ChandraOCR on Apple Silicon (mlx-vlm batch_generate)"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory to scan for images (recursive)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="File with one image path per line",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="OCR prompt (default: standard OCR prompt)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate per image (default: 8192)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Images per batch_generate call (default: 8)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Print progress every N images (default: 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already in --output CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-batch statistics",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        metavar="N",
        help="Resize longest side to N pixels (default: 1024)",
    )
    args = parser.parse_args()

    if (args.images_dir is None) == (args.manifest is None):
        parser.error("Provide exactly one of --images-dir or --manifest")

    paths = collect_image_paths(args.images_dir, args.manifest)
    print(f"Found {len(paths)} images")

    if not paths:
        print("No images to process.")
        sys.exit(0)

    if args.verbose:
        print(f"Configuration: {args}")

    run_batched(
        paths,
        args.output,
        args.prompt,
        args.max_tokens,
        args.batch_size,
        args.save_every,
        args.resume,
        args.verbose,
        args.max_size,
    )


if __name__ == "__main__":
    start_time = time()
    main()
    elapsed = time() - start_time
    print(f"Total time: {elapsed:.2f} seconds")
