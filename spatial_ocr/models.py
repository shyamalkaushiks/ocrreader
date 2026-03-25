"""
Data models shared across the PaddleOCR extractor pipeline.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class TableExtractionResult:
    """
    Holds the table data extracted from a single image.

    `quality_warnings` is a list of human-readable strings describing rows that
    could not be reliably repaired (e.g. two OCR rows merged into one cell).
    When non-empty the image is flagged in review_needed.txt for manual inspection.
    """
    column_count: int
    rows: List[List[str]]
    quality_warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImageTable:
    """Associates a source image path with its extracted table."""
    path: str
    column_count: int
    rows: List[List[str]]
