"""Consolidated final processing: classify_snrna_loci.

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
import csv
from pathlib import Path
from typing import Any
import pandas as pd
human_CELL_ORDER = ('A549', 'H1', 'HeLa-S3', 'HepG2', 'K562', 'MCF-7')
human_TARGETS = ('POLR2A', 'H3K27ac', 'H3K4me3')
human_DEFAULT_FC_THRESHOLD = 2.0
human_CLASS_ORDER = ('active_strong', 'active_supported', 'pol2_only_ambiguous', 'peak_only_below_fc', 'signal_only_no_peak', 'array_unresolved_processed', 'no_processed_evidence', 'not_evaluated_polIII')

def human_cell_string(values: set[str]) -> str:
    return ','.join((cell for cell in human_CELL_ORDER if cell in values))

def human_classify_locus(candidate: dict[str, str], strict_cells: dict[str, set[str]], mean_positive_cells: dict[str, set[str]], peak_cells: dict[str, set[str]]) -> tuple[str, str, set[str], set[str]]:
    pol2 = strict_cells['POLR2A']
    all_three = pol2 & strict_cells['H3K27ac'] & strict_cells['H3K4me3']
    pol2_with_histone = pol2 & (strict_cells['H3K27ac'] | strict_cells['H3K4me3'])
    if candidate['expected_polymerase'] != 'Pol_II':
        return ('not_evaluated_polIII', 'U6/U6atac loci are expected to use Pol III and are outside the Pol-II active-promoter classifier.', all_three, pol2_with_histone)
    if candidate['rnu2_array'] == 'true':
        return ('array_unresolved_processed', 'RNU2-1 tandem-array locus cannot be assigned reliably from standard-GRCh38 processed tracks.', all_three, pol2_with_histone)
    if all_three:
        return ('active_strong', 'At least one cell line has mean FC>2 and a consensus peak for POLR2A, H3K27ac and H3K4me3 in the paper-defined windows.', all_three, pol2_with_histone)
    if pol2_with_histone:
        return ('active_supported', 'At least one cell line has strict POLR2A evidence and one concordant active-promoter histone mark.', all_three, pol2_with_histone)
    if pol2:
        return ('pol2_only_ambiguous', 'Strict POLR2A evidence is present without concordant H3K27ac or H3K4me3 support in the same cell line.', all_three, pol2_with_histone)
    if peak_cells['POLR2A']:
        return ('peak_only_below_fc', 'A POLR2A consensus peak overlaps the window, but mean fold change does not exceed 2 in any cell line.', all_three, pol2_with_histone)
    if mean_positive_cells['POLR2A']:
        return ('signal_only_no_peak', 'POLR2A mean fold change exceeds 2 without a consensus peak; this is compatible with weak signal or mapping artefact.', all_three, pol2_with_histone)
    return ('no_processed_evidence', 'Neither strict POLR2A evidence nor a POLR2A consensus peak was detected in the six processed datasets.', all_three, pol2_with_histone)

def human_build_classification(candidates: list[dict[str, str]], metrics: pd.DataFrame, sequence_qc: dict[str, dict[str, str]], annotation_context: dict[str, dict[str, str]], threshold: float) -> list[dict[str, Any]]:
    by_locus = {locus_id: group for (locus_id, group) in metrics.groupby('locus_id')}
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        locus_id = candidate['locus_id']
        group = by_locus[locus_id]
        strict_cells: dict[str, set[str]] = {}
        mean_positive_cells: dict[str, set[str]] = {}
        peak_cells: dict[str, set[str]] = {}
        max_mean_fc: dict[str, float] = {}
        for target in human_TARGETS:
            target_rows = group[group['target'] == target]
            strict_cells[target] = set(target_rows.loc[(target_rows['mean_fc'] > threshold) & target_rows['peak_overlap'], 'cell_line'])
            mean_positive_cells[target] = set(target_rows.loc[target_rows['mean_fc'] > threshold, 'cell_line'])
            peak_cells[target] = set(target_rows.loc[target_rows['peak_overlap'], 'cell_line'])
            max_mean_fc[target] = float(target_rows['mean_fc'].max())
        (classification, reason, all_three, pol2_with_histone) = human_classify_locus(candidate, strict_cells, mean_positive_cells, peak_cells)
        qc = sequence_qc[locus_id]
        context = annotation_context[locus_id]
        preferred_name = qc['paper_alias'] or candidate['display_name']
        review_flags: list[str] = []
        if candidate['rnu2_array'] == 'true':
            review_flags.append('RNU2_ARRAY')
        if qc['candidate_set_mapping_risk'] != 'low':
            review_flags.append('CANDIDATE_SET_LOW_UNIQUENESS')
        if context['nearby_tss_confounding_risk'] == 'high':
            review_flags.append('OTHER_GENE_TSS_WITHIN_500BP')
        if classification == 'pol2_only_ambiguous':
            review_flags.append('POL2_WITHOUT_HISTONE')
        if classification == 'signal_only_no_peak':
            review_flags.append('SIGNAL_WITHOUT_PEAK')
        if classification == 'peak_only_below_fc':
            review_flags.append('PEAK_BELOW_FC')
        if classification in {'array_unresolved_processed', 'pol2_only_ambiguous', 'signal_only_no_peak', 'peak_only_below_fc'} or review_flags:
            review_priority = 'high'
        elif classification == 'active_supported' or context['nearby_tss_confounding_risk'] == 'moderate':
            review_priority = 'moderate'
        else:
            review_priority = 'low'
        output.append({**candidate, 'preferred_name': preferred_name, 'paper_alias': qc['paper_alias'], 'classification': classification, 'classification_reason': reason, 'fc_threshold': threshold, 'pol2_strict_cells': human_cell_string(strict_cells['POLR2A']), 'h3k27ac_strict_cells': human_cell_string(strict_cells['H3K27ac']), 'h3k4me3_strict_cells': human_cell_string(strict_cells['H3K4me3']), 'all_three_same_cells': human_cell_string(all_three), 'pol2_plus_histone_same_cells': human_cell_string(pol2_with_histone), 'pol2_mean_only_cells': human_cell_string(mean_positive_cells['POLR2A']), 'pol2_peak_cells': human_cell_string(peak_cells['POLR2A']), 'max_pol2_mean_fc': max_mean_fc['POLR2A'], 'max_h3k27ac_mean_fc': max_mean_fc['H3K27ac'], 'max_h3k4me3_mean_fc': max_mean_fc['H3K4me3'], 'candidate_set_mapping_risk': qc['candidate_set_mapping_risk'], 'mature_unique_50mer_fraction': qc['mature_unique_50mer_fraction'], 'promoter_unique_50mer_fraction': qc['promoter_unique_50mer_fraction'], 'exact_mature_sequence_copy_count': qc['exact_mature_sequence_copy_count'], 'nearest_other_gene_tss_distance_bp': context['nearest_other_gene_tss_distance_bp'], 'nearest_other_gene_name': context['nearest_other_gene_name'], 'nearest_other_gene_type': context['nearest_other_gene_type'], 'other_gene_tss_within_500bp': context['other_gene_tss_within_500bp'], 'overlapping_other_gene_bodies': context['overlapping_other_gene_bodies'], 'nearby_tss_confounding_risk': context['nearby_tss_confounding_risk'], 'review_priority': review_priority, 'review_flags': ';'.join(review_flags)})
    return output

def human_write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

def human_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=Path('results/candidates/snRNA_candidates.tsv'))
    parser.add_argument('--metrics', type=Path, default=Path('results/signals/locus_signal_metrics.tsv'))
    parser.add_argument('--sequence-qc', type=Path, default=Path('results/candidates/sequence_mappability_proxy.tsv'))
    parser.add_argument('--annotation-context', type=Path, default=Path('results/candidates/annotation_context.tsv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/classification'))
    return parser.parse_args()

def human_main() -> int:
    args = human_parse_args()
    with args.candidates.open(encoding='utf-8', newline='') as handle:
        candidates = list(csv.DictReader(handle, delimiter='\t'))
    metrics = pd.read_csv(args.metrics, sep='\t', low_memory=False)
    with args.sequence_qc.open(encoding='utf-8', newline='') as handle:
        sequence_qc = {row['locus_id']: row for row in csv.DictReader(handle, delimiter='\t')}
    with args.annotation_context.open(encoding='utf-8', newline='') as handle:
        annotation_context = {row['locus_id']: row for row in csv.DictReader(handle, delimiter='\t')}
    if metrics.shape[0] != len(candidates) * 18:
        raise SystemExit('signal metric table does not contain 18 tracks per candidate')
    classifications = human_build_classification(candidates, metrics, sequence_qc, annotation_context, human_DEFAULT_FC_THRESHOLD)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    human_write_tsv(args.output_dir / 'snRNA_locus_classification.tsv', classifications)
    human_write_tsv(args.output_dir / 'active_loci.tsv', [row for row in classifications if row['classification'] in {'active_strong', 'active_supported'}])
    human_write_tsv(args.output_dir / 'ambiguous_loci.tsv', [row for row in classifications if row['classification'] in {'pol2_only_ambiguous', 'peak_only_below_fc', 'signal_only_no_peak', 'array_unresolved_processed'}])
    summary_rows: list[dict[str, Any]] = []
    for family in ('U1', 'U2', 'U4', 'U5', 'U6', 'U11', 'U12', 'U4ATAC', 'U6ATAC'):
        for classification in human_CLASS_ORDER:
            count = sum((row['family'] == family and row['classification'] == classification for row in classifications))
            if count:
                summary_rows.append({'family': family, 'classification': classification, 'loci': count})
    human_write_tsv(args.output_dir / 'classification_summary.tsv', summary_rows)

def human_entrypoint():
    human_main()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    _run_stages({'human': 'human_entrypoint'}, globals(), setup=_prepare_runtime)
