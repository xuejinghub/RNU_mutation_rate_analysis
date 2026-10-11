"""Consolidated final processing: analyze_splicernu_annotation_comparison.

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
from collections import Counter
import math
from pathlib import Path
import shutil
import subprocess
import pandas as pd
catalogue_ROOT = (_PROCESSING_ROOT / 'fig1/gene_family/scripts/analyze_splicernu_annotation_comparison.py').parents[1]
catalogue_OUT = catalogue_ROOT / 'results/splicernu_annotation_comparison'
catalogue_TMP = catalogue_ROOT / 'tmp/splicernu_annotation_comparison_20260805/analysis'
catalogue_CLASSIFICATION = catalogue_ROOT / 'results/classification/snRNA_locus_classification.tsv'
catalogue_CANDIDATES = catalogue_ROOT / 'results/candidates/snRNA_candidates.tsv'
catalogue_SEQUENCE_QC = catalogue_ROOT / 'results/candidates/sequence_mappability_proxy.tsv'
catalogue_EXPRESSION = catalogue_ROOT / 'GSE126797_TPM_matrix_all_datasets.csv'
catalogue_RARE = catalogue_ROOT / 'rare_mutation_rate'
catalogue_RNU_ANN = catalogue_RARE / 'rnus_ann'
catalogue_NAME_MAP = catalogue_RNU_ANN / 'hg38.gencodeV44.allRNUs_ensemblID_gene_name.list'
catalogue_ELIGIBILITY_TRACK = Path('/public4/home/jiangzj/analysis/mtr/histone_mutation_explore/all_species_summary/data/variant_data/human/hg38.genmap.k150e1_unique.cov_16-48.bed')
catalogue_A3B_ROOT = Path('/public5/home/xuejing/RNU/APOBEC3B')
catalogue_BROAD_CLASSES = ('spliceRNUs_minusU6', 'RNUs_U6')
catalogue_AUTOSOMES = {f'chr{i}' for i in range(1, 23)}

def catalogue_stable_id(value: object) -> str:
    text = '' if value is None else str(value)
    return text.split('.', 1)[0] if text else ''

def catalogue_read_bed(path: Path, class_name: str | None=None) -> pd.DataFrame:
    frame = pd.read_csv(path, sep='\t', header=None, names=['chrom', 'start', 'end', 'versioned_gene_id'], dtype={'chrom': str, 'start': int, 'end': int, 'versioned_gene_id': str})
    frame['stable_gene_id'] = frame['versioned_gene_id'].map(catalogue_stable_id)
    if class_name is not None:
        frame['broad_class'] = class_name
    frame['length_bp'] = frame['end'] - frame['start']
    return frame

def catalogue_write_bed(frame: pd.DataFrame, path: Path, id_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, ['chrom', 'start', 'end', id_column]].to_csv(path, sep='\t', header=False, index=False)

def catalogue_bedtools_intersect(a_path: Path, b_path: Path, output_path: Path) -> None:
    bedtools = shutil.which('bedtools')
    if bedtools is None:
        raise RuntimeError('bedtools is required but was not found')
    with output_path.open('w', encoding='utf-8') as handle:
        completed = subprocess.run([bedtools, 'intersect', '-a', str(a_path), '-b', str(b_path)], stdout=handle, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f'bedtools intersect failed: {completed.stderr}')

def catalogue_read_intersection(path: Path, id_name: str) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=['chrom', 'start', 'end', id_name, 'eligible_bp'])
    frame = pd.read_csv(path, sep='\t', header=None, names=['chrom', 'start', 'end', id_name], dtype={'chrom': str, 'start': int, 'end': int, id_name: str})
    frame['eligible_bp'] = frame['end'] - frame['start']
    if (frame['eligible_bp'] <= 0).any():
        raise AssertionError(f'Non-positive eligible interval in {path}')
    return frame

def catalogue_interval_counter(frame: pd.DataFrame, id_column: str) -> Counter[tuple[object, ...]]:
    return Counter((tuple(row) for row in frame.loc[:, ['chrom', 'start', 'end', id_column]].itertuples(index=False, name=None)))

def catalogue_evidence_status(stable: str, expression_index: set[str], all_missing: set[str]) -> str:
    if not stable:
        return 'NOT_EVALUATED_NO_STABLE_ID'
    if stable not in expression_index:
        return 'NOT_EVALUATED_NOT_IN_MATRIX'
    if stable in all_missing:
        return 'NOT_EVALUATED_ALL_MISSING'
    return 'EVALUATED'

def catalogue_main() -> int:
    catalogue_OUT.mkdir(parents=True, exist_ok=True)
    catalogue_TMP.mkdir(parents=True, exist_ok=True)
    classification = pd.read_csv(catalogue_CLASSIFICATION, sep='\t', dtype=str).fillna('')
    classification['stable_gene_id'] = classification['gencode_gene_id'].map(catalogue_stable_id)
    classification_by_stable = classification.loc[classification['stable_gene_id'] != ''].copy()
    if classification_by_stable['stable_gene_id'].duplicated().any():
        raise AssertionError('Non-empty GENCODE stable IDs are duplicated in classification')
    strict = classification.loc[(classification['expected_polymerase'] == 'Pol_II') & (classification['pol2_strict_cells'] != '')].copy()
    if len(strict) != 49:
        raise AssertionError(f'Expected 49 strict POLR2A loci, observed {len(strict)}')
    if strict['locus_id'].duplicated().any():
        raise AssertionError('Strict locus IDs are not unique')
    candidates = pd.read_csv(catalogue_CANDIDATES, sep='\t', dtype=str).fillna('')
    candidates['stable_gene_id'] = candidates['gencode_gene_id'].map(catalogue_stable_id)
    candidate_by_stable = candidates.loc[candidates['stable_gene_id'] != ''].copy()
    if candidate_by_stable['stable_gene_id'].duplicated().any():
        raise AssertionError('GENCODE stable IDs are duplicated in the candidate table')
    sequence_qc = pd.read_csv(catalogue_SEQUENCE_QC, sep='\t')
    sequence_qc = sequence_qc.merge(candidates[['locus_id', 'stable_gene_id']], on='locus_id', how='left', validate='one_to_one')
    name_map = pd.read_csv(catalogue_NAME_MAP, sep='\t', header=None, names=['stable_gene_id', 'gencode_v44_gene_name', 'source_subfamily'], dtype=str)
    if name_map['stable_gene_id'].duplicated().any():
        raise AssertionError('GENCODE v44 name map has duplicated stable IDs')
    broad_frames: list[pd.DataFrame] = []
    broad_filtered_source_frames: list[pd.DataFrame] = []
    for class_name in catalogue_BROAD_CLASSES:
        broad_frames.append(catalogue_read_bed(catalogue_RNU_ANN / f'hg38.gencodeV44.{class_name}_transcript.bed', class_name))
        broad_filtered_source_frames.append(catalogue_read_bed(catalogue_RNU_ANN / f'hg38.gencodeV44.{class_name}_transcript.no_blacklist.autosome.bed', class_name))
    broad = pd.concat(broad_frames, ignore_index=True)
    broad_filtered_source = pd.concat(broad_filtered_source_frames, ignore_index=True)
    if len(broad) != 200 or broad['stable_gene_id'].nunique() != 200:
        raise AssertionError('Expected 200 unique GENCODE v44 spliceRNU/U6 gene IDs')
    broad = broad.merge(name_map, on='stable_gene_id', how='left', validate='one_to_one')
    if broad['gencode_v44_gene_name'].isna().any():
        raise AssertionError('GENCODE name mapping is incomplete for broad200')
    broad_full_bed = catalogue_OUT / 'broad200_full.bed'
    strict_full_bed = catalogue_OUT / 'strict49_full.bed'
    broad_intersection_bed = catalogue_OUT / 'broad200_eligibility_segments.bed'
    strict_intersection_bed = catalogue_OUT / 'strict49_eligibility_segments.bed'
    catalogue_write_bed(broad, broad_full_bed, 'versioned_gene_id')
    strict_bed_frame = strict[['chrom', 'start', 'end', 'locus_id']].copy()
    strict_bed_frame[['start', 'end']] = strict_bed_frame[['start', 'end']].astype(int)
    catalogue_write_bed(strict_bed_frame, strict_full_bed, 'locus_id')
    catalogue_bedtools_intersect(broad_full_bed, catalogue_ELIGIBILITY_TRACK, broad_intersection_bed)
    catalogue_bedtools_intersect(strict_full_bed, catalogue_ELIGIBILITY_TRACK, strict_intersection_bed)
    broad_segments = catalogue_read_intersection(broad_intersection_bed, 'versioned_gene_id')
    broad_segments['stable_gene_id'] = broad_segments['versioned_gene_id'].map(catalogue_stable_id)
    broad_segments['broad_class'] = broad_segments['stable_gene_id'].map(broad.set_index('stable_gene_id')['broad_class'])
    strict_segments = catalogue_read_intersection(strict_intersection_bed, 'locus_id')
    source_intervals_match = catalogue_interval_counter(broad_segments, 'versioned_gene_id') == catalogue_interval_counter(broad_filtered_source, 'versioned_gene_id')
    if not source_intervals_match:
        raise AssertionError('Regenerated broad eligibility intervals differ from source files')
    broad_eligible = broad_segments.groupby('stable_gene_id', as_index=False).agg(eligible_bp=('eligible_bp', 'sum'), eligible_segments=('eligible_bp', 'size'))
    strict_eligible = strict_segments.groupby('locus_id', as_index=False).agg(eligible_bp_strict_coordinate=('eligible_bp', 'sum'), eligible_segments_strict_coordinate=('eligible_bp', 'size'))
    broad_current = broad.merge(classification_by_stable, on='stable_gene_id', how='left', validate='one_to_one', suffixes=('_v44', ''))
    if broad_current['locus_id'].isna().any():
        raise AssertionError('Every broad200 stable ID should map to the current candidate set')
    broad_current['candidate_start'] = pd.to_numeric(broad_current['start'], errors='raise')
    broad_current['candidate_end'] = pd.to_numeric(broad_current['end'], errors='raise')
    broad_current['coordinate_exact_current_vs_v44'] = (broad_current['chrom_v44'] == broad_current['chrom']) & (broad_current['start_v44'] == broad_current['candidate_start']) & (broad_current['end_v44'] == broad_current['candidate_end'])
    if not broad_current['coordinate_exact_current_vs_v44'].all():
        raise AssertionError('GENCODE v44 broad coordinates do not exactly match current candidates')
    expression = pd.read_csv(catalogue_EXPRESSION)
    sample_columns = [column for column in expression.columns if column != 'gene_id']
    expression[sample_columns] = expression[sample_columns].apply(pd.to_numeric, errors='coerce')
    if expression['gene_id'].duplicated().any():
        raise AssertionError('Expression matrix gene IDs are duplicated')
    expression['median_tpm'] = expression[sample_columns].median(axis=1, skipna=True)
    expression['mean_tpm'] = expression[sample_columns].mean(axis=1, skipna=True)
    expression['detected_samples_gt0'] = (expression[sample_columns] > 0).sum(axis=1)
    expression['detected_fraction_gt0'] = expression['detected_samples_gt0'] / len(sample_columns)
    expression = expression.rename(columns={'gene_id': 'stable_gene_id'})
    expression_index = set(expression['stable_gene_id'])
    all_missing = set(expression.loc[expression[sample_columns].isna().all(axis=1), 'stable_gene_id'])
    expression_metrics = expression[['stable_gene_id', 'median_tpm', 'mean_tpm', 'detected_samples_gt0', 'detected_fraction_gt0']]
    a3b_frames: list[pd.DataFrame] = []
    for class_name in catalogue_BROAD_CLASSES:
        path = catalogue_A3B_ROOT / f'hg38.gencodeV44.{class_name}_transcript.no_blacklist.A3B_signal_summary.tsv'
        frame = pd.read_csv(path, sep='\t', header=None, names=['stable_gene_id', 'a3b_ssdna_rep2_sum', 'a3b_ssdna_rep1_sum', 'a3b_chip_mean_per_bp', 'a3b_ssdna_mean_per_bp'])
        frame['a3b_source_class'] = class_name
        a3b_frames.append(frame)
    a3b = pd.concat(a3b_frames, ignore_index=True)
    if a3b['stable_gene_id'].duplicated().any():
        raise AssertionError('A3B summary has duplicate stable IDs')
    if set(a3b['stable_gene_id']) != set(broad_eligible['stable_gene_id']):
        raise AssertionError('A3B coverage differs from broad eligibility coverage')
    strict_ids = set(strict.loc[strict['stable_gene_id'] != '', 'stable_gene_id'])
    broad_ids = set(broad['stable_gene_id'])
    membership_rows: list[dict[str, object]] = []
    current_index = classification_by_stable.set_index('stable_gene_id', drop=False)
    sequence_index = sequence_qc.set_index('locus_id', drop=False)
    expression_lookup = expression_metrics.set_index('stable_gene_id')
    a3b_lookup = a3b.set_index('stable_gene_id')
    broad_eligible_lookup = broad_eligible.set_index('stable_gene_id')
    strict_eligible_lookup = strict_eligible.set_index('locus_id')

    def append_record(*, current: pd.Series, in_broad: bool, broad_row: pd.Series | None) -> None:
        stable = str(current['stable_gene_id'])
        locus_id = str(current['locus_id'])
        if in_broad and broad_row is not None:
            chrom = str(broad_row['chrom'])
            start = int(broad_row['start'])
            end = int(broad_row['end'])
            versioned = str(broad_row['versioned_gene_id'])
            broad_class = str(broad_row['broad_class'])
            name = str(broad_row['gencode_v44_gene_name'])
            coordinate_source = 'GENCODE_v44'
        else:
            chrom = str(current['chrom'])
            start = int(current['start'])
            end = int(current['end'])
            versioned = str(current['gencode_gene_id'])
            broad_class = ''
            name = str(current['display_name'])
            coordinate_source = str(current['source_combination'])
        if in_broad and stable in broad_eligible_lookup.index:
            eligible_bp = int(broad_eligible_lookup.loc[stable, 'eligible_bp'])
            eligible_segments = int(broad_eligible_lookup.loc[stable, 'eligible_segments'])
        elif locus_id in strict_eligible_lookup.index:
            eligible_bp = int(strict_eligible_lookup.loc[locus_id, 'eligible_bp_strict_coordinate'])
            eligible_segments = int(strict_eligible_lookup.loc[locus_id, 'eligible_segments_strict_coordinate'])
        else:
            eligible_bp = 0
            eligible_segments = 0
        exp_status = catalogue_evidence_status(stable, expression_index, all_missing)
        if exp_status == 'EVALUATED':
            expr = expression_lookup.loc[stable]
            median_tpm = float(expr['median_tpm'])
            mean_tpm = float(expr['mean_tpm'])
            detected_samples = int(expr['detected_samples_gt0'])
            detected_fraction = float(expr['detected_fraction_gt0'])
        else:
            median_tpm = math.nan
            mean_tpm = math.nan
            detected_samples = math.nan
            detected_fraction = math.nan
        if stable and stable in a3b_lookup.index:
            a3b_status = 'EVALUATED'
            a3b_row = a3b_lookup.loc[stable]
            chip = float(a3b_row['a3b_chip_mean_per_bp'])
            ssdna = float(a3b_row['a3b_ssdna_mean_per_bp'])
        else:
            a3b_status = 'NOT_EVALUATED_OUTSIDE_V44_ELIGIBILITY_SCOPE'
            chip = math.nan
            ssdna = math.nan
        seq = sequence_index.loc[locus_id]
        is_strict = bool(current['expected_polymerase'] == 'Pol_II' and current['pol2_strict_cells'] != '')
        if is_strict:
            evidence_group = 'Strict POLR2A'
        elif broad_class == 'spliceRNUs_minusU6':
            evidence_group = 'Other annotated Pol II'
        elif broad_class == 'RNUs_U6':
            evidence_group = 'Annotated U6/U6ATAC'
        else:
            evidence_group = 'Other'
        chromosome_scope = 'autosome' if chrom in catalogue_AUTOSOMES else 'chrX' if chrom == 'chrX' else 'other'
        row = {'record_key': stable if in_broad else locus_id, 'in_strict49': is_strict, 'in_broad200': in_broad, 'evidence_group': evidence_group, 'broad_class': broad_class, 'locus_id': locus_id, 'stable_gene_id': stable, 'versioned_gene_id': versioned, 'display_name': name, 'family': str(current['family']), 'spliceosome': str(current['spliceosome']), 'expected_polymerase': str(current['expected_polymerase']), 'chrom': chrom, 'start': start, 'end': end, 'length_bp': end - start, 'chromosome_scope': chromosome_scope, 'coordinate_source': coordinate_source, 'annotation_class': str(current['annotation_class']), 'is_pseudogene_annotation': str(current['annotation_class']) == 'gencode_pseudogene', 'classification': str(current['classification']), 'pol2_strict': bool(current['pol2_strict_cells']), 'pol2_plus_histone_same_cell': bool(current['pol2_plus_histone_same_cells']), 'all_three_same_cell': bool(current['all_three_same_cells']), 'max_pol2_mean_fc': pd.to_numeric(current['max_pol2_mean_fc'], errors='coerce'), 'max_h3k27ac_mean_fc': pd.to_numeric(current['max_h3k27ac_mean_fc'], errors='coerce'), 'max_h3k4me3_mean_fc': pd.to_numeric(current['max_h3k4me3_mean_fc'], errors='coerce'), 'eligible_bp': eligible_bp, 'eligible_segments': eligible_segments, 'eligible_fraction': eligible_bp / (end - start), 'expression_status': exp_status, 'median_tpm': median_tpm, 'mean_tpm': mean_tpm, 'detected_samples_gt0': detected_samples, 'detected_fraction_gt0': detected_fraction, 'a3b_status': a3b_status, 'a3b_chip_mean_per_bp': chip, 'a3b_ssdna_mean_per_bp': ssdna, 'mature_unique_50mer_fraction': float(seq['mature_unique_50mer_fraction']), 'promoter_unique_50mer_fraction': float(seq['promoter_unique_50mer_fraction']), 'exact_mature_sequence_copy_count': int(seq['exact_mature_sequence_copy_count']), 'candidate_mapping_risk': str(current['candidate_set_mapping_risk'])}
        membership_rows.append(row)
    broad_lookup = broad.set_index('stable_gene_id', drop=False)
    for stable in broad['stable_gene_id']:
        append_record(current=current_index.loc[stable], in_broad=True, broad_row=broad_lookup.loc[stable])
    strict_only = strict.loc[~strict['stable_gene_id'].isin(broad_ids)].copy()
    for (_, current) in strict_only.iterrows():
        append_record(current=current, in_broad=False, broad_row=None)
    membership = pd.DataFrame(membership_rows)
    if len(membership) != 205 or membership['record_key'].duplicated().any():
        raise AssertionError('Expected a 205-locus union with unique record keys')
    if int(membership['in_strict49'].sum()) != 49 or int(membership['in_broad200'].sum()) != 200:
        raise AssertionError('Membership counts do not recover strict49 and broad200')

    def assign_tier(row: pd.Series) -> tuple[str, bool, str]:
        if row['in_strict49']:
            if row['eligible_bp'] <= 0:
                return ('EXCLUDE_NO_ELIGIBLE_BASES', False, 'No overlap with the existing GenMap/coverage eligibility track')
            if row['chromosome_scope'] != 'autosome':
                return ('SEPARATE_ACTIVE_POL2_CHRX', False, 'Sex chromosome requires a separate mutation denominator and ploidy model')
            if not row['in_broad200']:
                reason = 'Rfam-only coordinate lacks a stable GENCODE gene ID' if not row['stable_gene_id'] else 'Strict current locus is absent from the GENCODE v44 broad set'
                return ('SENSITIVITY_ACTIVE_POL2_COORDINATE_ONLY', False, reason)
            return ('PRIMARY_ACTIVE_POL2_AUTOSOMAL', True, 'Strict POLR2A evidence, GENCODE v44 identity, autosomal eligible bases')
        if row['eligible_bp'] <= 0:
            return ('EXCLUDE_NO_ELIGIBLE_BASES', False, 'No overlap with the existing GenMap/coverage eligibility track')
        if row['broad_class'] == 'spliceRNUs_minusU6':
            if row['chromosome_scope'] != 'autosome':
                return ('SEPARATE_OTHER_POL2_CHRX', False, 'Sex chromosome requires a separate mutation denominator and ploidy model')
            return ('SENSITIVITY_OTHER_POL2_AUTOSOMAL', False, 'GENCODE Pol-II-family locus without strict processed POLR2A evidence')
        if row['broad_class'] == 'RNUs_U6':
            if row['chromosome_scope'] != 'autosome':
                return ('SEPARATE_POLIII_CHRX', False, 'Pol III and sex-chromosome effects both require separate modeling')
            return ('SEPARATE_POLIII_U6_AUTOSOMAL', False, 'U6/U6ATAC are Pol III expected and must not be pooled with Pol II loci')
        return ('EXCLUDE_UNRESOLVED', False, 'Unresolved scope')
    tiers = membership.apply(assign_tier, axis=1, result_type='expand')
    tiers.columns = ['recommended_tier', 'include_primary_active_pol2', 'tier_reason']
    membership = pd.concat([membership, tiers], axis=1)
    membership.to_csv(catalogue_OUT / 'harmonized_locus_evidence.tsv', sep='\t', index=False, na_rep='NA')
    membership.to_csv(catalogue_OUT / 'recommended_tiered_rnu_loci.tsv', sep='\t', index=False, na_rep='NA')
    membership.loc[membership['in_strict49'] & ~membership['in_broad200']].to_csv(catalogue_OUT / 'strict49_only_loci.tsv', sep='\t', index=False, na_rep='NA')
    membership.loc[membership['chromosome_scope'] == 'chrX'].to_csv(catalogue_OUT / 'chrx_loci.tsv', sep='\t', index=False, na_rep='NA')
    membership.loc[membership['recommended_tier'] == 'EXCLUDE_NO_ELIGIBLE_BASES'].to_csv(catalogue_OUT / 'excluded_no_eligible_bases_loci.tsv', sep='\t', index=False, na_rep='NA')

def catalogue_entrypoint():
    catalogue_main()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    _run_stages({'catalogue': 'catalogue_entrypoint'}, globals(), setup=_prepare_runtime)
