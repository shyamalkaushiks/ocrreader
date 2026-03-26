"""
Spatial OCR — Table Extractor
-------------------------------
Third pipeline alongside paddleocr_extractor/ and opencv_grid_ocr/.

Key difference from the others
--------------------------------
paddleocr_extractor/  → PP-Structure whole-table HTML  (fast, ~70-80 %, merge issues)
opencv_grid_ocr/      → OpenCV H-lines + PaddleOCR text boxes  (good on images with visible row lines)
spatial_ocr/          → Pure PaddleOCR text boxes + Y/X spatial clustering  (no grid dependency)

The spatial approach never merges adjacent rows because row boundaries are
determined by PHYSICAL GAPS between text fragments, not by a neural network
deciding where `<tr>` tags go.

Usage:
    cd /path/to/excelproject
    ./paddleocr_extractor/venv/bin/python3 spatial_ocr/main.py --input enhanced_images --output spatial_output

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
      ./paddleocr_extractor/venv/bin/python3 spatial_ocr/main.py \\
      --input enhanced_images --output spatial_output --limit 40 --save-json
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from excel_writer import write_single_workbook, write_workbooks_by_column_count
from models import ImageTable, TableExtractionResult
from spatial_engine import extract_table_from_image, initialize_ocr

IMAGE_EXTENSIONS: frozenset = frozenset({".jpg", ".jpeg", ".png", ".webp"})


# ── argument parsing ───────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spatial OCR table extractor — PaddleOCR text boxes + Y/X clustering"
    )
    p.add_argument("--input",  "-i", default="../enhanced_images",
                   help="Input folder with images (default: ../enhanced_images)")
    p.add_argument("--s3-bucket", default="",
                   help="S3 bucket name to fetch images from (overrides --input)")
    p.add_argument("--s3-prefix", default="",
                   help="S3 key prefix for image objects")
    p.add_argument("--s3-region", default="",
                   help="AWS region for S3 client (optional)")
    p.add_argument("--s3-profile", default="",
                   help="AWS profile name to use (optional)")
    p.add_argument("--output", "-o", default="../spatial_output",
                   help="Output folder for xlsx files (default: ../spatial_output)")
    p.add_argument("--limit",  "-l", type=int, default=0,
                   help="Process only first N images (0 = all)")
    p.add_argument("--offset", type=int, default=0,
                   help="Skip first N images")
    p.add_argument("--single-xlsx", action="store_true",
                   help="One workbook with one sheet per column count")
    p.add_argument("--min-columns", type=int, default=3,
                   help="Skip images with fewer detected columns (default: 3)")
    p.add_argument("--save-json", action="store_true",
                   help="Save debug JSON per image under output/debug/")
    p.add_argument("--include-quality-warnings", action="store_true",
                   help="Keep all rows even if they triggered quality warnings")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Log every image")
    return p.parse_args()


# ── file helpers ───────────────────────────────────────────────────────────

def list_image_files(input_dir: str, offset: int, limit: int) -> List[str]:
    if not os.path.isdir(input_dir):
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])
    if not files:
        print(f"No images found in: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if offset > 0:
        files = files[offset:]
        print(f"offset: skipped {offset} image(s), remaining {len(files)}", file=sys.stderr)
    if limit > 0:
        files = files[:limit]
        print(f"limit: processing {len(files)} image(s)", file=sys.stderr)
    return files


def _create_s3_client(args: argparse.Namespace):
    session_kwargs: Dict[str, str] = {}
    if args.s3_profile:
        session_kwargs["profile_name"] = args.s3_profile
    session = boto3.Session(**session_kwargs)
    client_kwargs: Dict[str, str] = {}
    if args.s3_region:
        client_kwargs["region_name"] = args.s3_region
    return session.client("s3", **client_kwargs)


def _list_s3_image_keys(
    s3_client,
    bucket_name: str,
    prefix: str,
    offset: int,
    limit: int,
) -> List[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    keys: List[str] = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            extension = os.path.splitext(key)[1].lower()
            if extension in IMAGE_EXTENSIONS:
                keys.append(key)
    keys.sort()
    if offset > 0:
        keys = keys[offset:]
        print(f"offset: skipped {offset} image(s), remaining {len(keys)}", file=sys.stderr)
    if limit > 0:
        keys = keys[:limit]
        print(f"limit: processing {len(keys)} image(s)", file=sys.stderr)
    if not keys:
        print(
            f"No images found in s3://{bucket_name}/{prefix}",
            file=sys.stderr,
        )
        sys.exit(1)
    return keys


def _download_s3_images(
    s3_client,
    bucket_name: str,
    object_keys: List[str],
    destination_dir: str,
) -> List[str]:
    os.makedirs(destination_dir, exist_ok=True)
    local_paths: List[str] = []
    for object_key in object_keys:
        local_filename = os.path.basename(object_key)
        if not local_filename:
            continue
        local_path = os.path.join(destination_dir, local_filename)
        s3_client.download_file(bucket_name, object_key, local_path)
        local_paths.append(local_path)
    if not local_paths:
        print(
            f"No valid image objects downloaded from s3://{bucket_name}",
            file=sys.stderr,
        )
        sys.exit(1)
    return sorted(local_paths)


def resolve_input_files(args: argparse.Namespace) -> Tuple[List[str], str]:
    if not args.s3_bucket:
        return list_image_files(args.input, args.offset, args.limit), args.input
    try:
        s3_client = _create_s3_client(args)
        keys = _list_s3_image_keys(
            s3_client=s3_client,
            bucket_name=args.s3_bucket,
            prefix=args.s3_prefix,
            offset=args.offset,
            limit=args.limit,
        )
        cache_dir = os.path.join(args.output, "_s3_input_cache")
        local_paths = _download_s3_images(
            s3_client=s3_client,
            bucket_name=args.s3_bucket,
            object_keys=keys,
            destination_dir=cache_dir,
        )
        return local_paths, f"s3://{args.s3_bucket}/{args.s3_prefix}"
    except (BotoCoreError, ClientError) as err:
        print(f"Failed to fetch S3 input: {err}", file=sys.stderr)
        sys.exit(1)


def save_debug_json(debug_dir: str, image_path: str, result: TableExtractionResult) -> None:
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(debug_dir, f"{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)


def save_review_log(output_dir: str, review_lines: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "review_needed.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Images that need manual review.\n")
        f.write("# Full skip = nothing written to Excel.\n")
        f.write("# quality = bad rows skipped; good rows still in Excel.\n")
        f.write("# Options:\n")
        f.write("#   1. Re-run with --include-quality-warnings to force all rows\n\n")
        for line in review_lines:
            f.write(line + "\n")
    # Intentionally quiet during batch runs.


def _append_process_log_line(process_log_path: str, line: str) -> None:
    with open(process_log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_process_log(
    output_dir: str,
    process_lines: List[str],
) -> None:
    """
    Writes a per-image processing status log.

    This helps you see:
      - which images were written to Excel
      - which ones were skipped (few_cols / quality)
      - which ones errored
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "process_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Per-image processing log for spatial_ocr\n")
        f.write("# Format: index/total filename -> STATUS cols=X rows=Y\n\n")
        for line in process_lines:
            f.write(line + "\n")
    # Intentionally quiet during batch runs.


def save_snapshot(
    output_dir: str,
    by_cols: Dict[int, List[ImageTable]],
    single_xlsx: bool,
) -> None:
    if single_xlsx:
        write_single_workbook(output_dir, by_cols)
    else:
        write_workbooks_by_column_count(output_dir, by_cols)


# ── output decision ────────────────────────────────────────────────────────

def _resolve_output(
    result: TableExtractionResult,
    filename: str,
    args: argparse.Namespace,
) -> Tuple[Optional[TableExtractionResult], Optional[str], str]:
    """
    Returns (table_for_excel | None, review_log_line | None, status).
    status: "ok" | "few_cols" | "quality"
    """
    if result.column_count < args.min_columns or not result.rows:
        return None, f"{filename}  # reason: {result.column_count} col(s) — too few", "few_cols"

    if args.include_quality_warnings or not result.quality_warnings:
        return result, None, "ok"

    preview = "; ".join(result.quality_warnings[:2])
    log_line = f"{filename}  # quality: {preview}"
    return result, log_line, "quality"


# ── summary ────────────────────────────────────────────────────────────────

def print_summary(
    success: int, few_cols: int, errors: int,
    quality_flagged: int, total: int,
    elapsed: float, output_dir: str,
) -> None:
    print(f"\n{'=' * 55}")
    print(f"Done in {elapsed:.1f}s   ({total} images)")
    print(f"   Written to Excel     : {success}")
    if few_cols:
        print(f"   Too few columns     : {few_cols}")
    if quality_flagged:
        print(f"   Quality warnings    : {quality_flagged}  ← see review_needed.txt")
    if errors:
        print(f"   Errors              : {errors}")
    print(f"\nOutput folder: {output_dir}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)
    files, input_source = resolve_input_files(args)
    total = len(files)

    debug_dir = ""
    if args.save_json:
        debug_dir = os.path.join(args.output, "debug")
        os.makedirs(debug_dir, exist_ok=True)

    print("\nInitialising PaddleOCR (spatial mode — no PP-Structure)...")
    engine = initialize_ocr()
    print(f"Engine ready. Processing {total} images...\n")
    print(f"   Input:  {input_source}")
    print(f"   Output: {args.output}\n")

    by_cols: Dict[int, List[ImageTable]] = {}
    success = 0
    few_cols_skip = 0
    errors = 0
    quality_flagged = 0
    review_lines: List[str] = []
    process_log_path = os.path.join(args.output, "process_log.txt")
    with open(process_log_path, "w", encoding="utf-8") as f:
        f.write("# Per-image processing log for spatial_ocr\n")
        f.write("# Format: [index/total] filename -> STATUS (cols=X rows=Y)\n\n")
    start_time = time.time()

    for index, image_path in enumerate(files, 1):
        filename = os.path.basename(image_path)
        # Minimal terminal progress so you can see processing is alive.
        print(f"[{index}/{total}] Processing {filename} ...", flush=True)
        image_start_time = time.time()
        try:
            result = extract_table_from_image(engine, image_path)

            if debug_dir:
                save_debug_json(debug_dir, image_path, result)
                wrote_json_seconds = time.time() - image_start_time
                print(
                    f"[{index}/{total}] Saved debug JSON for {filename} "
                    f"(cols={result.column_count}, rows={len(result.rows)}) "
                    f"in {wrote_json_seconds:.1f}s",
                    flush=True,
                )

            excel_result, log_line, status = _resolve_output(result, filename, args)

            if log_line:
                review_lines.append(log_line)

            _append_process_log_line(
                process_log_path,
                f"[{index}/{total}] {filename} -> {status} "
                f"(cols={result.column_count}, rows={len(result.rows)})",
            )

            if status == "few_cols":
                few_cols_skip += 1
                if args.verbose:
                    print(f"  [{index}/{total}] {filename} -> few_cols ({result.column_count})")
                if not args.verbose:
                    print(f"  [{index}/{total}] {filename} -> few_cols (cols={result.column_count})", flush=True)
                continue

            if status == "quality":
                quality_flagged += 1
                print(
                    f"  [{index}/{total}] {filename} -> quality skip (cols={result.column_count}, rows={len(result.rows)})",
                    flush=True,
                )
                # Match paddleocr_extractor behavior:
                # quality/conflict images are logged for manual review, but are not written to Excel.
                continue

            target = excel_result if excel_result is not None else result
            by_cols.setdefault(target.column_count, []).append(
                ImageTable(
                    path=image_path,
                    column_count=target.column_count,
                    rows=target.rows,
                )
            )
            save_snapshot(args.output, by_cols, args.single_xlsx)
            success += 1
            image_total_seconds = time.time() - image_start_time
            print(
                f"  [{index}/{total}] Done {filename}: wrote Excel "
                f"(cols={target.column_count}, rows={len(target.rows)}, "
                f"elapsed={image_total_seconds:.1f}s)",
                flush=True,
            )

        except Exception as err:
            errors += 1
            review_lines.append(f"{filename}  # reason: error — {err}")
            _append_process_log_line(
                process_log_path,
                f"[{index}/{total}] {filename} -> ERROR (reason={err})",
            )

    elapsed = time.time() - start_time
    save_review_log(args.output, review_lines)
    print_summary(success, few_cols_skip, errors, quality_flagged, total, elapsed, args.output)

    if not by_cols:
        print("No images written to Excel — all skipped or failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
