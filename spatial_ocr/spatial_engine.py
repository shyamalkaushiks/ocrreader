"""
Spatial OCR Engine — Option B Implementation
---------------------------------------------
Uses PaddleOCR's raw text-detection output (bounding boxes + text) and
assembles the table purely from spatial positions, bypassing PP-Structure's
broken table-structure HTML entirely.

Why this is better than PP-Structure HTML
-----------------------------------------
PP-Structure decides ALL cell boundaries in one neural-network shot and outputs
an HTML <table>.  When it cannot see a thin grid line it merges multiple rows
or columns into one <td>.  By the time your Python code sees the HTML, the
structural damage is already done.

Here we work with raw OCR text boxes (every detected text fragment with its
pixel-level bounding box) and assemble rows and columns ourselves:

  1. Y-CLUSTERING  — group text boxes that are vertically close into the same
     row.  Gap threshold is adaptive (based on median text-box height), so it
     handles tables with varying row heights automatically.  Because we work
     with physical pixel positions, multi-row merges are impossible: a text
     box sitting at Y=350 and one at Y=700 will never land in the same row.

  2. X-CLUSTERING  — find consistent column X-positions by clustering all
     horizontal midpoints across all rows.  Each cluster = one column.  Because
     columns are defined by WHERE data sits horizontally, not by grid lines,
     this works even for borderless Excel tables.

  3. CELL ASSEMBLY — assign each text box to the nearest column in its row.
     Multiple boxes in the same (row, col) slot are joined with a space.

  4. UI-CHROME FILTER — rows that look like Excel UI (formula bar, column
     letters A-Z, sheet tabs, toolbars) are removed after assembly.
"""
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from models import TableExtractionResult

# ── upscaling ──────────────────────────────────────────────────────────────
_UPSCALE_TARGET_WIDTH = 2400
_UPSCALE_MAX_FACTOR = 2.5

# ── Y-clustering (row detection) ───────────────────────────────────────────
# A new row starts when a box's y-midpoint is more than this multiple of the
# median box height away from the running mean y-midpoint of the current row.
# Using y-midpoints (not y1/y2 edges) is essential because adjacent-row boxes
# physically overlap in pixel Y coordinates — their edges touch or cross — so
# an edge-gap comparison always produces a negative or near-zero number and
# never triggers a row split.  Mid-to-mid gaps between rows are ~25 px while
# within-row mid scatter is only 0–8 px, giving a clean separation point.
_ROW_GAP_FACTOR = 0.7

# ── X-clustering (column detection) ────────────────────────────────────────
# Two text-box X-midpoints belong to the same column if they are closer than
# this fraction of image width.  2 % keeps columns at least 48 px apart on a
# 2400 px wide image — fine for typical 5-8 column spreadsheets.
_COL_CLUSTER_FRACTION = 0.02   # 2 % of image width

# Wide boxes (spanning > this fraction of image width) are omitted from
# column detection — they are likely multi-column merged artifacts.
_MAX_BOX_WIDTH_FRACTION = 0.45

# A cluster must appear in at least this fraction of total midpoints to be
# considered a real column (not sporadic noise).  Lowered so narrow columns
# (e.g. serial-number column with few cells) are not discarded.
_MIN_COL_PRESENCE_FRACTION = 0.04

_MAX_COLS = 25

# ── quality warnings ───────────────────────────────────────────────────────
_SUSPICIOUSLY_LONG_CELL = 100

# ── UI-chrome filtering patterns ───────────────────────────────────────────
_EXCEL_COL_LETTER    = re.compile(r'^[A-Z]{1,2}$')
_EXCEL_TOOLBAR_KW    = frozenset({"merge", "wrap", "calibri", "callbri", "arial"})
_OFFICE_MENU_KW      = frozenset({
    "insert", "draw", "layout", "formula", "review", "view",
    "help", "wps", "pdf", "format", "data", "tools", "window",
})
_CELL_REF_RE         = re.compile(r'^[A-Za-z]{1,3}\d{1,7}$')
_TASKBAR_TEMP_RE     = re.compile(r'\b\d+[°º]?[CF]\b', re.IGNORECASE)
_WEATHER_WORDS       = frozenset({
    "clear", "cloudy", "sunny", "overcast", "rain", "rainy",
    "drizzle", "snow", "fog", "foggy", "haze", "mist", "windy",
})

_INDIAN_MOBILE_RE       = re.compile(r'^[6-9]\d{9}$')
_SERIAL_SPACE_PHONE_RE  = re.compile(r'^(\d{1,5})\s+([6-9]\d{9})(?:\s.*)?$')
_SERIAL_CONCAT_PHONE_RE = re.compile(r'^(\d{1,5})([6-9]\d{9})$')
_SERIAL_PHONE_TRIGGER   = 0.30

_PENDING_CHECK_DATA_RE = re.compile(
    r'pending[\s_]*check[\s_]*data',
    re.IGNORECASE,
)


# ── data class ─────────────────────────────────────────────────────────────

class _Box:
    """A single OCR-detected text fragment with its bounding box."""
    __slots__ = ("x1", "y1", "x2", "y2", "text")

    def __init__(self, x1: float, y1: float, x2: float, y2: float, text: str):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.text = text

    @property
    def x_mid(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def y_mid(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1


# ── public API ─────────────────────────────────────────────────────────────

def initialize_ocr():
    """
    Initialises PaddleOCR in full OCR mode (text detection + recognition).
    PP-Structure is deliberately NOT used — we assemble table structure
    ourselves via spatial clustering of raw text boxes.
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as err:
        raise RuntimeError(
            "PaddleOCR is not installed.\n"
            "Run:  pip install paddlepaddle paddleocr"
        ) from err

    try:
        return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except (TypeError, ValueError):
        return PaddleOCR(use_angle_cls=True, lang="en")


def extract_table_from_image(engine, image_path: str) -> TableExtractionResult:
    """
    Full pipeline:
      1. Load + upscale image.
      2. Run PaddleOCR → raw text boxes.
      3. Y-cluster → rows.
      4. X-cluster → columns.
      5. Assign boxes to (row, col) cells.
      6. Filter UI-chrome rows.
      7. Return TableExtractionResult.

    Raises:
        OSError:    if image cannot be read.
        ValueError: if no text is detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise OSError(f"Cannot read image: {image_path}")

    img = _upscale(img)
    img_h, img_w = img.shape[:2]

    boxes = _run_ocr(engine, img)
    if not boxes:
        raise ValueError("No text detected in image")

    row_groups   = _cluster_rows_y(boxes)
    col_centers  = _cluster_columns_x(boxes, img_w)

    if not col_centers:
        raise ValueError("Could not determine column positions")

    rows, warnings = _assemble_table(row_groups, col_centers, img_w)
    rows = _filter_ui_rows(rows)
    rows = _fix_serial_and_phone_concatenation(rows)

    if not rows:
        raise ValueError("All rows filtered as UI chrome")

    col_count = max(len(r) for r in rows) if rows else 0
    return TableExtractionResult(
        column_count=col_count,
        rows=rows,
        quality_warnings=warnings,
    )


# ── step 1: run PaddleOCR ──────────────────────────────────────────────────

def _run_ocr(engine, img: np.ndarray) -> List[_Box]:
    """Runs PaddleOCR and returns a flat list of _Box objects."""
    # Fast path: preserve original local behaviour.
    # If `cls` is supported, use the original parsing shape (`result[0]`).
    try:
        result_try = engine.ocr(img, cls=True)
    except TypeError:
        result_try = None
    except Exception:
        return []

    if isinstance(result_try, list) and result_try and result_try[0]:
        boxes: List[_Box] = []
        for item in result_try[0]:
            if not item or len(item) < 2:
                continue
            try:
                quad = item[0]
                tc = item[1]
                text = tc[0] if isinstance(tc, (list, tuple)) else str(tc)
                text = text.strip()
                if not text:
                    continue
                xs = [float(pt[0]) for pt in quad]
                ys = [float(pt[1]) for pt in quad]
                boxes.append(_Box(min(xs), min(ys), max(xs), max(ys), text))
            except (IndexError, TypeError, ValueError):
                continue
        if boxes:
            return boxes

    # Fallback: PaddleOCR API differs by version. Some versions don't accept `cls`
    # in `ocr()`; use safer argument sets and normalize output.
    result: object = None
    ocr_variants: Tuple[Dict[str, bool], ...] = (
        {},
        {"det": True, "rec": True},
    )
    for ocr_kwargs in ocr_variants:
        try:
            result = engine.ocr(img, **ocr_kwargs)
            break
        except TypeError:
            continue
        except Exception:
            return []

    if not result:
        return []

    items: List[object] = []
    if isinstance(result, list) and result:
        if isinstance(result[0], list):
            items = result[0] if result[0] is not None else []
        else:
            items = result

    if not items:
        return []

    boxes_fallback: List[_Box] = []
    for item in items:
        if not item or len(item) < 2:
            continue
        try:
            quad = item[0]
            tc = item[1]
            text = tc[0] if isinstance(tc, (list, tuple)) and tc else str(tc)
            text = text.strip()
            if not text:
                continue
            xs = [float(pt[0]) for pt in quad]
            ys = [float(pt[1]) for pt in quad]
            boxes_fallback.append(_Box(min(xs), min(ys), max(xs), max(ys), text))
        except (IndexError, TypeError, ValueError):
            continue

    return boxes_fallback


# ── step 2: Y-cluster into rows ────────────────────────────────────────────

def _cluster_rows_y(boxes: List[_Box]) -> List[List[_Box]]:
    """
    Groups text boxes into rows using an adaptive y-midpoint gap threshold.

    Boxes are sorted by y-midpoint.  A new row starts whenever a box's
    y-midpoint is more than _ROW_GAP_FACTOR × median_height away from the
    running mean y-midpoint of the current cluster.

    Why y-midpoints (not y1/y2 edges):
      In Excel screenshot tables the bounding boxes of text in adjacent rows
      physically OVERLAP vertically (the detected quad for row N+1 starts
      before the quad for row N ends).  Comparing box.y1 − current_group.y2
      therefore yields a negative or near-zero number and never fires, causing
      every box in the image to collapse into a single row.  Y-midpoint gaps
      between true rows are ~25 px; within-row midpoint scatter is only 0-8 px,
      giving a clean separation regardless of bounding-box overlap.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b.y_mid)

    heights = sorted([b.height for b in sorted_boxes])
    median_h = heights[len(heights) // 2]
    gap_threshold = max(6.0, median_h * _ROW_GAP_FACTOR)

    rows: List[List[_Box]] = []
    current: List[_Box] = [sorted_boxes[0]]
    current_sum_mid: float = sorted_boxes[0].y_mid

    for box in sorted_boxes[1:]:
        current_mean_mid = current_sum_mid / len(current)
        gap = box.y_mid - current_mean_mid
        if gap > gap_threshold:
            rows.append(current)
            current = [box]
            current_sum_mid = box.y_mid
        else:
            current.append(box)
            current_sum_mid += box.y_mid

    rows.append(current)
    return rows


# ── step 3: X-cluster into columns ────────────────────────────────────────

def _cluster_columns_x(boxes: List[_Box], img_width: int) -> List[float]:
    """
    Finds stable column X-center positions by clustering the X-midpoints
    of all narrow text boxes (excludes multi-column wide spans).

    Returns a sorted list of column center X values.
    """
    max_box_w   = img_width * _MAX_BOX_WIDTH_FRACTION
    cluster_gap = img_width * _COL_CLUSTER_FRACTION

    midpoints = sorted(
        b.x_mid for b in boxes if b.width < max_box_w
    )
    if not midpoints:
        return []

    clusters: List[List[float]] = []
    current_cluster: List[float] = [midpoints[0]]

    for mp in midpoints[1:]:
        if mp - current_cluster[-1] <= cluster_gap:
            current_cluster.append(mp)
        else:
            clusters.append(current_cluster)
            current_cluster = [mp]
    clusters.append(current_cluster)

    min_count = max(2, int(len(midpoints) * _MIN_COL_PRESENCE_FRACTION))
    kept = [c for c in clusters if len(c) >= min_count]

    if not kept or len(kept) > _MAX_COLS:
        return []

    return [sum(c) / len(c) for c in kept]


# ── step 4: assemble table ────────────────────────────────────────────────

def _assemble_table(
    row_groups: List[List[_Box]],
    col_centers: List[float],
    img_width: int,
) -> Tuple[List[List[str]], List[str]]:
    """
    Assigns each text box to (row, column) using nearest column center and
    builds the 2-D table.  Multiple boxes in the same cell are joined by
    a space (e.g. a name split across two OCR detections).
    """
    n_cols = len(col_centers)
    sorted_centers = sorted(col_centers)

    # Half-point boundaries between adjacent column centers
    boundaries: List[float] = [0.0]
    for i in range(len(sorted_centers) - 1):
        boundaries.append((sorted_centers[i] + sorted_centers[i + 1]) / 2.0)
    boundaries.append(float(img_width))

    rows: List[List[str]] = []
    warnings: List[str] = []

    for row_idx, group in enumerate(row_groups):
        cell_parts: Dict[int, List[str]] = defaultdict(list)
        for box in sorted(group, key=lambda b: b.x1):
            col_idx = _nearest_col(box.x_mid, boundaries)
            cell_parts[col_idx].append(box.text)

        row: List[str] = []
        for c in range(n_cols):
            text = " ".join(cell_parts.get(c, []))
            row.append(text)
            if len(text) > _SUSPICIOUSLY_LONG_CELL:
                warnings.append(
                    f"row {row_idx} col {c}: unusually long "
                    f"({len(text)} chars) — possible row boundary miss: "
                    f"{text[:80]!r}…"
                )

        rows.append(row)

    rows = [r for r in rows if any(c.strip() for c in r)]
    return rows, warnings


def _nearest_col(x_mid: float, boundaries: List[float]) -> int:
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= x_mid < boundaries[i + 1]:
            return i
    return len(boundaries) - 2


# ── step 5: filter UI-chrome rows ─────────────────────────────────────────

def _filter_ui_rows(rows: List[List[str]]) -> List[List[str]]:
    """
    Removes rows that are clearly Excel/WPS/OS UI artifacts rather than data.
    Replicates the logic of paddleocr_extractor's _filter_excel_ui_rows but
    works on already-assembled row lists instead of HTML cells.
    """
    filtered: List[List[str]] = []
    for row in rows:
        non_empty = [c.strip() for c in row if c.strip()]
        if not non_empty:
            continue
        if _is_formula_bar(non_empty):
            continue
        if _is_col_letter_row(non_empty):
            continue
        if _is_sheet_tab(non_empty):
            continue
        if _is_toolbar(non_empty):
            continue
        if _is_office_menu(non_empty):
            continue
        if _is_license_row(non_empty):
            continue
        if _is_status_bar(non_empty):
            continue
        if _is_taskbar(non_empty):
            continue
        if _is_pending_check_row(non_empty):
            continue
        filtered.append(row)
    return filtered


def _is_formula_bar(cells: List[str]) -> bool:
    if any("fx" in c for c in cells):
        return True
    if len(cells) <= 2 and any(_CELL_REF_RE.match(c) for c in cells):
        return True
    return False


def _is_col_letter_row(cells: List[str]) -> bool:
    tokens: List[str] = []
    for c in cells:
        tokens.extend(c.split())
    return bool(tokens) and all(_EXCEL_COL_LETTER.match(t) for t in tokens)


def _is_sheet_tab(cells: List[str]) -> bool:
    for c in cells:
        for tok in c.lower().split():
            if "sheet" in tok or re.match(r'^s[ph]ee[rt][l12]?$', tok):
                return True
    return False


def _is_toolbar(cells: List[str]) -> bool:
    return any(any(kw in c.lower() for kw in _EXCEL_TOOLBAR_KW) for c in cells)


def _is_office_menu(cells: List[str]) -> bool:
    full = " ".join(cells).lower()
    return sum(1 for kw in _OFFICE_MENU_KW if kw in full) >= 2


def _is_license_row(cells: List[str]) -> bool:
    full = " ".join(cells).lower()
    return "genuine" in full or "license" in full


def _is_status_bar(cells: List[str]) -> bool:
    full = " ".join(cells).lower()
    return "ready" in full or "accessibility" in full


def _is_taskbar(cells: List[str]) -> bool:
    if any(_TASKBAR_TEMP_RE.search(c) for c in cells):
        return True
    if len(cells) == 1 and cells[0].lower() in _WEATHER_WORDS:
        return True
    return False


def _is_pending_check_row(cells: List[str]) -> bool:
    """
    Filters the "Pending check Data" / "Pending_check Data" row that appears
    as a stray UI line in some Excel screenshots.
    """
    full = " ".join(cells).strip()
    return bool(_PENDING_CHECK_DATA_RE.search(full))


def _try_split_serial_phone(cell: str) -> Optional[tuple]:
    """
    Returns (serial, phone) if the cell contains a merged serial+phone, else None.

    Handles two layouts:
    - Space-separated : "1305 7592008515"   → ("1305", "7592008515")
    - Concatenated    : "13079886181723"    → ("1307", "9886181723")
                        "8427838928211"     → ("842",  "7838928211")
    """
    cell = cell.strip()
    m = _SERIAL_SPACE_PHONE_RE.match(cell)
    if m and _INDIAN_MOBILE_RE.match(m.group(2)):
        return m.group(1), m.group(2)
    digits = re.sub(r'\D+', '', cell)
    if len(digits) > 10:
        phone_part  = digits[-10:]
        serial_part = digits[:-10]
        if _INDIAN_MOBILE_RE.match(phone_part) and 1 <= len(serial_part) <= 5:
            return serial_part, phone_part
    return None


def _fix_serial_and_phone_concatenation(rows: List[List[str]]) -> List[List[str]]:
    """
    Splits col0 when serial and phone are merged into one cell.

    Works for ANY column count — 6-col tables (serial+phone | city | amount | …),
    8-col tables where col1 is already a long concatenation (empty serial col),
    etc.

    Two detection passes:
    A) col0 has serial+phone directly  →  expand col0 into [serial, phone]
    B) col0 is empty AND col1 is a >10-digit number with valid trailing mobile
       → split col1 into [serial, phone]  (the 8-col layout)

    Trigger: ≥30 % of rows must match before we commit to the split.
    """
    if not rows:
        return rows

    # ── Pass A: col0 contains merged serial+phone ─────────────────────────
    col0_cells = [row[0].strip() for row in rows if row]
    hits_a = sum(1 for c in col0_cells if _try_split_serial_phone(c) is not None)
    if hits_a >= len(col0_cells) * _SERIAL_PHONE_TRIGGER:
        fixed: List[List[str]] = []
        for row in rows:
            if not row:
                fixed.append(row)
                continue
            result = _try_split_serial_phone(row[0].strip())
            if result:
                serial, phone = result
                fixed.append([serial, phone] + list(row[1:]))
            else:
                fixed.append(["", row[0]] + list(row[1:]))
        return fixed

    # ── Pass B: col0 empty, col1 has serial+phone concatenated ────────────
    def _col1_split(row: List[str]) -> Optional[tuple]:
        if not row or len(row) < 2:
            return None
        if row[0].strip():
            return None
        digits = re.sub(r'\D+', '', row[1])
        if len(digits) <= 10:
            return None
        phone_part  = digits[-10:]
        serial_part = digits[:-10]
        if _INDIAN_MOBILE_RE.match(phone_part) and 1 <= len(serial_part) <= 5:
            return serial_part, phone_part
        return None

    hits_b = sum(1 for row in rows if _col1_split(row) is not None)
    if hits_b >= len(rows) * _SERIAL_PHONE_TRIGGER:
        fixed2: List[List[str]] = []
        for row in rows:
            result = _col1_split(row)
            if result:
                serial, phone = result
                row_out = list(row)
                row_out[0] = serial
                row_out[1] = phone
                fixed2.append(row_out)
            else:
                fixed2.append(list(row))
        return fixed2

    return rows


# ── helpers ────────────────────────────────────────────────────────────────

def _upscale(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= _UPSCALE_TARGET_WIDTH:
        return img
    factor = min(_UPSCALE_TARGET_WIDTH / w, _UPSCALE_MAX_FACTOR)
    return cv2.resize(
        img, (int(w * factor), int(h * factor)),
        interpolation=cv2.INTER_LANCZOS4,
    )
