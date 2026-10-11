"""Shared table reading, interval merging and file fingerprints for processing."""
from __future__ import annotations
from typing import Iterable
import hashlib
import pandas as pd

def read_table(path, **options):
    return pd.read_csv(path, sep='\t', float_precision='round_trip', **options)

def sha256_file(path):
    """Return a fingerprint for cache keys and source records; do not verify it."""
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def merge_intervals_06ce9ba0(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent 0-based, half-open intervals."""
    merged: list[list[int]] = []
    for (start, end) in sorted(((int(a), int(b)) for (a, b) in intervals)):
        if start < 0 or end <= start:
            raise ValueError(f'Invalid half-open interval [{start}, {end})')
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for (start, end) in merged]
