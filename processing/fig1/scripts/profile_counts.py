"""Consolidated final processing: profile_counts.

One implementation per calculation; stages share common kernels.
See processing/README.md and stage_catalog.tsv for inputs and execution scope.
"""
from __future__ import annotations
import sys
from pathlib import Path
from functools import partial as _partial
_REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p/'processing/component_registry.json').is_file())
if str(_REPOSITORY) not in sys.path: sys.path.insert(0,str(_REPOSITORY))
from processing._runtime import component as _component, component_file as _component_file, PROCESSING_ROOT as _PROCESSING_ROOT, run_stages as _run_stages
from processing import common as _shared
import bisect
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
profiles_THRESHOLDS = {'Human': 0.001, 'Rhesus_macaque': 0.002, 'Mouse': 0.01}
profiles_PROFILE_BINS = 100
profiles_FLANK_BINS = 40
profiles_BODY_BINS = 20
profiles_FLANK_BIN_BP = 50
profiles_merge_intervals = _shared.merge_intervals_06ce9ba0

def profiles_build_profile_bins(catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gene in catalog.to_dict('records'):
        start = int(gene['start0'])
        end = int(gene['end'])
        strand = gene['strand']
        body_edges = np.rint(np.linspace(start, end, profiles_BODY_BINS + 1)).astype(int)
        if np.any(np.diff(body_edges) <= 0):
            raise ValueError(f"Gene body too short for {profiles_BODY_BINS} bins: {gene['stable_gene_id']}")
        genomic_intervals: list[tuple[int, int, str, int, int]] = []
        for index in range(profiles_FLANK_BINS):
            if strand == '+':
                a = start - (profiles_FLANK_BINS - index) * profiles_FLANK_BIN_BP
                b = start - (profiles_FLANK_BINS - index - 1) * profiles_FLANK_BIN_BP
            else:
                a = end + (profiles_FLANK_BINS - index - 1) * profiles_FLANK_BIN_BP
                b = end + (profiles_FLANK_BINS - index) * profiles_FLANK_BIN_BP
            genomic_intervals.append((a, b, 'upstream', -2000 + index * 50, -1950 + index * 50))
        for index in range(profiles_BODY_BINS):
            source_index = index if strand == '+' else profiles_BODY_BINS - index - 1
            a = int(body_edges[source_index])
            b = int(body_edges[source_index + 1])
            genomic_intervals.append((a, b, 'body', index * 50, (index + 1) * 50))
        for index in range(profiles_FLANK_BINS):
            if strand == '+':
                a = end + index * profiles_FLANK_BIN_BP
                b = end + (index + 1) * profiles_FLANK_BIN_BP
            else:
                a = start - (index + 1) * profiles_FLANK_BIN_BP
                b = start - index * profiles_FLANK_BIN_BP
            genomic_intervals.append((a, b, 'downstream', 1000 + index * 50, 1050 + index * 50))
        for (bin_index, (a, b, region, relative_start, relative_end)) in enumerate(genomic_intervals):
            if a < 0:
                raise ValueError(f"Negative profile coordinate for {gene['stable_gene_id']}")
            bin_id = f"{gene['species']}|{gene['stable_gene_id']}|{bin_index:03d}"
            rows.append({**gene, 'bin_id': bin_id, 'bin_index': bin_index, 'region': region, 'relative_start': relative_start, 'relative_end': relative_end, 'genomic_start': a, 'genomic_end': b, 'genomic_bin_bp': b - a})
    bins = pd.DataFrame(rows)
    expected = int(len(catalog) * profiles_PROFILE_BINS)
    if len(bins) != expected or bins['bin_id'].nunique() != expected:
        raise ValueError('Profile bin construction failed')
    return bins

def profiles_load_callable_index(path: Path) -> dict[str, tuple[list[int], list[int]]]:
    """Load a BED as merged per-chromosome start/end arrays."""
    raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open() as handle:
        for (line_number, line) in enumerate(handle, start=1):
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 3:
                raise ValueError(f'Malformed callable BED line {path}:{line_number}')
            raw[fields[0]].append((int(fields[1]), int(fields[2])))
    index: dict[str, tuple[list[int], list[int]]] = {}
    for (chrom, intervals) in raw.items():
        merged = profiles_merge_intervals(intervals)
        index[chrom] = ([start for (start, _) in merged], [end for (_, end) in merged])
    return index

def profiles_interval_callable_bp(index: dict[str, tuple[list[int], list[int]]], chrom: str, start: int, end: int) -> int:
    if chrom not in index:
        return 0
    (starts, ends) = index[chrom]
    interval_index = bisect.bisect_right(ends, start)
    total = 0
    while interval_index < len(starts) and starts[interval_index] < end:
        total += max(0, min(end, ends[interval_index]) - max(start, starts[interval_index]))
        interval_index += 1
    return total

def profiles_position_is_callable(index: dict[str, tuple[list[int], list[int]]], chrom: str, pos0: int) -> bool:
    if chrom not in index:
        return False
    (starts, ends) = index[chrom]
    interval_index = bisect.bisect_right(starts, pos0) - 1
    return interval_index >= 0 and pos0 < ends[interval_index]

def profiles_numeric_alt_value(value: Any, alt_index: int, n_alts: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if alt_index >= len(value):
            return None
        value = value[alt_index]
    elif n_alts != 1 and alt_index != 0:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None

def profiles_classify_gene_profile_direct(species: str, gene_bins: pd.DataFrame, callable_index: dict[str, tuple[list[int], list[int]]], vcf: Any) -> tuple[pd.DataFrame, Counter[str]]:
    """Classify one gene window without materializing genome-wide base maps."""
    chrom_values = gene_bins['chrom'].unique()
    if len(chrom_values) != 1:
        raise ValueError('A profile gene must occupy exactly one chromosome')
    chrom = str(chrom_values[0])
    genomic = gene_bins.sort_values('genomic_start')
    genomic_starts = genomic['genomic_start'].astype(int).tolist()
    genomic_ends = genomic['genomic_end'].astype(int).tolist()
    genomic_bin_ids = genomic['bin_id'].astype(str).tolist()
    window_start = min(genomic_starts)
    window_end = max(genomic_ends)
    callable_bp = {str(row.bin_id): profiles_interval_callable_bp(callable_index, chrom, int(row.genomic_start), int(row.genomic_end)) for row in gene_bins.itertuples()}
    position_alleles: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, str, str]] = set()
    qc: Counter[str] = Counter()
    for record in vcf.fetch(chrom, window_start, window_end):
        qc['fetched_record_gene_occurrences'] += 1
        pos0 = int(record.pos) - 1
        if not profiles_position_is_callable(callable_index, chrom, pos0):
            continue
        bin_index = bisect.bisect_right(genomic_starts, pos0) - 1
        if bin_index < 0 or pos0 >= genomic_ends[bin_index]:
            continue
        bin_id = genomic_bin_ids[bin_index]
        alts = tuple(record.alts or ())
        if not alts:
            continue
        an = profiles_numeric_alt_value(record.info.get('AN'), 0, 1)
        ac_value = record.info.get('AC')
        filter_pass = tuple(record.filter.keys()) == ('PASS',)
        for (alt_index, alt_value) in enumerate(alts):
            (ref, alt) = (str(record.ref), str(alt_value))
            key = (pos0, ref, alt)
            if key in seen:
                qc['duplicate_alt_alleles'] += 1
                continue
            seen.add(key)
            ac = profiles_numeric_alt_value(ac_value, alt_index, len(alts))
            af = None
            if ac is not None and an is not None and (an > 0):
                candidate = ac / an
                if math.isfinite(candidate) and 0 <= candidate <= 1:
                    af = candidate
            valid_snv = len(ref) == 1 and len(alt) == 1 and (ref in 'ACGT') and (alt in 'ACGT') and (ref != alt)
            position_alleles[pos0, bin_id].append({'filter_pass': filter_pass, 'valid_snv': valid_snv, 'af': af})
            qc['alt_allele_gene_occurrences'] += 1
    threshold = profiles_THRESHOLDS[species]
    mask_positions: Counter[str] = Counter()
    rare_positions: Counter[str] = Counter()
    rare_alt: Counter[str] = Counter()
    for ((_, bin_id), alleles) in position_alleles.items():
        has_excluded = any((not allele['filter_pass'] or not allele['valid_snv'] or allele['af'] is None or (allele['af'] > threshold) for allele in alleles))
        if has_excluded:
            mask_positions[bin_id] += 1
            qc['masked_position_gene_occurrences'] += 1
            continue
        count = sum((allele['filter_pass'] and allele['valid_snv'] and (allele['af'] is not None) and (allele['af'] <= threshold) for allele in alleles))
        if count:
            rare_positions[bin_id] += 1
            rare_alt[bin_id] += count
            qc['qualifying_position_gene_occurrences'] += 1
            qc['qualifying_alt_gene_occurrences'] += count
    result = gene_bins.copy()
    result['callable_bp'] = result['bin_id'].map(callable_bp).fillna(0).astype(int)
    result['mask_bp'] = result['bin_id'].map(mask_positions).fillna(0).astype(int)
    result['effective_callable_bp'] = result['callable_bp'] - result['mask_bp']
    result['rare_snv_position_count'] = result['bin_id'].map(rare_positions).fillna(0).astype(int)
    result['rare_snv_alt_count'] = result['bin_id'].map(rare_alt).fillna(0).astype(int)
    result['rare_snv_density_per_kb'] = np.where(result['effective_callable_bp'].gt(0), 1000 * result['rare_snv_alt_count'] / result['effective_callable_bp'], np.nan)
    if (result['effective_callable_bp'] < 0).any():
        raise ValueError(f'Negative effective callable bases for {species}')
    return (result, qc)

def _prepare_runtime():
    pass
