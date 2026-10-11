"""Consolidated final processing: extract_processed_signals.

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
import argparse
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pyBigWig
signal_CELL_ORDER = ('A549', 'H1', 'HeLa-S3', 'HepG2', 'K562', 'MCF-7')
signal_TARGET_ORDER = ('POLR2A', 'H3K27ac', 'H3K4me3')
signal_TARGET_WINDOWS = {'POLR2A': (-50, 400), 'H3K27ac': (-400, 200), 'H3K4me3': (-1000, 200)}
signal_PROFILE_START = -2000
signal_PROFILE_END = 1000
signal_PROFILE_BIN_SIZE = 25
signal_PROFILE_FLOOR_FC = 0.25
signal_METRIC_FIELDS = ('locus_id', 'display_name', 'family', 'expected_polymerase', 'annotation_class', 'chrom', 'start', 'end', 'strand', 'tss0', 'rnu2_array', 'cell_line', 'target', 'experiment_accession', 'signal_file_accession', 'peak_file_accession', 'window_rel_start', 'window_rel_end', 'window_genomic_start', 'window_genomic_end', 'window_bases', 'mean_fc', 'log2_mean_fc', 'max_fc', 'positive_fraction', 'peak_overlap', 'peak_count', 'peak_overlap_bp_max', 'peak_signal_value_max', 'peak_score_max')

def signal_track_sort_key(row: dict[str, str]) -> tuple[int, int]:
    return (signal_CELL_ORDER.index(row['cell_line']), signal_TARGET_ORDER.index(row['target']))

def signal_relative_window(tss0: int, strand: str, rel_start: int, rel_end: int) -> tuple[int, int]:
    if strand == '+':
        return (tss0 + rel_start, tss0 + rel_end)
    if strand == '-':
        return (tss0 - rel_end, tss0 - rel_start)
    raise ValueError(f'unexpected strand: {strand}')

class signal_PeakIndex:

    def __init__(self, path: Path) -> None:
        by_chrom: dict[str, list[tuple[int, int, float, float]]] = {}
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip() or line.startswith(('#', 'track', 'browser')):
                    continue
                fields = line.rstrip('\n').split('\t')
                by_chrom.setdefault(fields[0], []).append((int(fields[1]), int(fields[2]), float(fields[6]), float(fields[4])))
        self.data: dict[str, tuple[list[int], list[tuple[int, int, float, float]], list[int]]] = {}
        for (chrom, records) in by_chrom.items():
            records.sort(key=lambda item: (item[0], item[1]))
            starts = [item[0] for item in records]
            prefix_max_end: list[int] = []
            running = -1
            for item in records:
                running = max(running, item[1])
                prefix_max_end.append(running)
            self.data[chrom] = (starts, records, prefix_max_end)

    def query(self, chrom: str, start: int, end: int) -> dict[str, float | int | bool]:
        if chrom not in self.data:
            return {'peak_overlap': False, 'peak_count': 0, 'peak_overlap_bp_max': 0, 'peak_signal_value_max': 0.0, 'peak_score_max': 0.0}
        (starts, records, prefix_max_end) = self.data[chrom]
        index = bisect_left(starts, end) - 1
        overlaps: list[tuple[int, int, float, float]] = []
        while index >= 0 and prefix_max_end[index] > start:
            record = records[index]
            if record[1] > start:
                overlaps.append(record)
            index -= 1
        return {'peak_overlap': bool(overlaps), 'peak_count': len(overlaps), 'peak_overlap_bp_max': max((min(end, item[1]) - max(start, item[0]) for item in overlaps), default=0), 'peak_signal_value_max': max((item[2] for item in overlaps), default=0.0), 'peak_score_max': max((item[3] for item in overlaps), default=0.0)}

def signal_extract_track(signal_row: dict[str, str], peak_row: dict[str, str], candidates: list[dict[str, str]], profile_candidates: list[dict[str, str]]) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    target = signal_row['target']
    (rel_start, rel_end) = signal_TARGET_WINDOWS[target]
    signal_path = Path(signal_row['destination'])
    peak_path = Path(peak_row['destination'])
    peak_index = signal_PeakIndex(peak_path)
    bigwig = pyBigWig.open(str(signal_path))
    chrom_sizes = bigwig.chroms()
    metric_rows: list[dict[str, Any]] = []
    profile_matrix = np.zeros((len(profile_candidates), (signal_PROFILE_END - signal_PROFILE_START) // signal_PROFILE_BIN_SIZE), dtype=np.float32)
    try:
        for candidate in candidates:
            chrom = candidate['chrom']
            tss0 = int(candidate['tss0'])
            (genomic_start, genomic_end) = signal_relative_window(tss0, candidate['strand'], rel_start, rel_end)
            if chrom not in chrom_sizes or genomic_start < 0 or genomic_end > chrom_sizes[chrom]:
                raise ValueError(f"{candidate['locus_id']} target window is outside {chrom} bounds")
            values = np.asarray(bigwig.values(chrom, genomic_start, genomic_end, numpy=True), dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            mean_fc = float(values.mean())
            peak_metrics = peak_index.query(chrom, genomic_start, genomic_end)
            metric_rows.append({**{field: candidate[field] for field in ('locus_id', 'display_name', 'family', 'expected_polymerase', 'annotation_class', 'chrom', 'start', 'end', 'strand', 'tss0', 'rnu2_array')}, 'cell_line': signal_row['cell_line'], 'target': target, 'experiment_accession': signal_row['experiment_accession'], 'signal_file_accession': signal_row['file_accession'], 'peak_file_accession': peak_row['file_accession'], 'window_rel_start': rel_start, 'window_rel_end': rel_end, 'window_genomic_start': genomic_start, 'window_genomic_end': genomic_end, 'window_bases': len(values), 'mean_fc': mean_fc, 'log2_mean_fc': math.log2(mean_fc) if mean_fc > 0 else '', 'max_fc': float(values.max(initial=0.0)), 'positive_fraction': float(np.mean(values > 0)), **peak_metrics})
        number_of_bins = profile_matrix.shape[1]
        for (index, candidate) in enumerate(profile_candidates):
            chrom = candidate['chrom']
            tss0 = int(candidate['tss0'])
            (genomic_start, genomic_end) = signal_relative_window(tss0, candidate['strand'], signal_PROFILE_START, signal_PROFILE_END)
            stats = bigwig.stats(chrom, genomic_start, genomic_end, type='mean', nBins=number_of_bins, exact=True)
            means = np.asarray([0.0 if value is None else value for value in stats], dtype=float)
            if candidate['strand'] == '-':
                means = means[::-1]
            profile_matrix[index, :] = means.astype(np.float32)
    finally:
        bigwig.close()
    track_metadata = {'cell_line': signal_row['cell_line'], 'target': target, 'experiment_accession': signal_row['experiment_accession'], 'signal_file_accession': signal_row['file_accession'], 'peak_file_accession': peak_row['file_accession'], 'signal_path': str(signal_path), 'peak_path': str(peak_path), 'biological_replicates': signal_row['biological_replicates']}
    return (metric_rows, profile_matrix, track_metadata)

def signal_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('manifests/paper_inputs.manifest.tsv'))
    parser.add_argument('--candidates', type=Path, default=Path('results/candidates/snRNA_candidates.tsv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/signals'))
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()

def signal_main() -> int:
    args = signal_parse_args()
    with args.manifest.open(encoding='utf-8', newline='') as handle:
        manifest = list(csv.DictReader(handle, delimiter='\t'))
    with args.candidates.open(encoding='utf-8', newline='') as handle:
        candidates = list(csv.DictReader(handle, delimiter='\t'))
    profile_candidates = [row for row in candidates if row['expected_polymerase'] == 'Pol_II']
    signals = sorted([row for row in manifest if row['role'] == 'SIGNAL_FC_BW'], key=signal_track_sort_key)
    peaks_by_experiment = {row['experiment_accession']: row for row in manifest if row['role'] == 'CONSENSUS_PEAKS'}
    if len(signals) != 18 or len(peaks_by_experiment) != 18:
        raise SystemExit('expected 18 signal tracks and 18 peak files')
    tasks = [(signal, peaks_by_experiment[signal['experiment_accession']]) for signal in signals]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = list(executor.map(lambda pair: signal_extract_track(pair[0], pair[1], candidates, profile_candidates), tasks))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = [row for result in results for row in result[0]]
    metric_path = args.output_dir / 'locus_signal_metrics.tsv'
    with metric_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=signal_METRIC_FIELDS, delimiter='\t')
        writer.writeheader()
        writer.writerows(all_metrics)
    profile_stack = np.stack([result[1] for result in results], axis=1)
    profile_path = args.output_dir / 'tss_profiles_25bp.npz'
    np.savez_compressed(profile_path, mean_fc=profile_stack, locus_ids=np.asarray([row['locus_id'] for row in profile_candidates]), display_names=np.asarray([row['display_name'] for row in profile_candidates]), families=np.asarray([row['family'] for row in profile_candidates]), track_labels=np.asarray([f"{result[2]['cell_line']}|{result[2]['target']}" for result in results]), relative_bin_starts=np.arange(signal_PROFILE_START, signal_PROFILE_END, signal_PROFILE_BIN_SIZE))
    metadata = {'coordinate_system': 'GRCh38, 0-based half-open; profiles are strand-oriented', 'candidate_count': len(candidates), 'profile_candidate_count': len(profile_candidates), 'metric_row_count': len(all_metrics), 'target_windows_relative_to_tss': signal_TARGET_WINDOWS, 'profile': {'relative_start': signal_PROFILE_START, 'relative_end': signal_PROFILE_END, 'bin_size': signal_PROFILE_BIN_SIZE, 'number_of_bins': profile_stack.shape[2], 'stored_values': 'linear ENCODE fold change over control, mean per bin', 'recommended_display_transform': f'log2(max(mean_fc, {signal_PROFILE_FLOOR_FC})); display floor is not used for classification'}, 'tracks': [result[2] for result in results]}
    metadata_path = args.output_dir / 'signal_extraction_metadata.json'
    with metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(f'Candidates: {len(candidates):,}')
    print(f'Pol-II-expected profile candidates: {len(profile_candidates):,}')
    print(f'Tracks: {len(results)}')
    print(f'Metric rows: {len(all_metrics):,}')
    print(f'Profile shape: {profile_stack.shape}')
    print(f'Metrics: {metric_path}')
    print(f'Profiles: {profile_path}')
    return 0

def signal_entrypoint():
    signal_main()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    _run_stages({'signal': 'signal_entrypoint'}, globals(), setup=_prepare_runtime)
