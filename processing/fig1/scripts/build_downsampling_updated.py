"""Consolidated final processing: build_downsampling_updated.

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
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import pysam
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import importlib.util
import os
import sys
from scipy.stats import mannwhitneyu
genotypes_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/primate/transpecies_compare/scripts/calculate_shuf2000_downsample_panel_c_folds.py').parents[1]
genotypes_DEFAULT_INDEX_DIR = genotypes_ROOT / 'tmp/shuf_downsample_20260901/indexes'
genotypes_SOURCE_VCF_DIR = Path('/public5/home/xuejing/population_mutation_rate/HGDP/v3.1.2')
genotypes_SOURCE_VCF_TEMPLATE = 'gnomad.genomes.v3.1.2.hgdp_tgp.chr{chrom}.vcf.bgz'
genotypes_INDEX_URL_TEMPLATE = 'https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1.2/vcf/genomes/gnomad.genomes.v3.1.2.hgdp_tgp.chr{chrom}.vcf.bgz.tbi'
genotypes_PARENT_SAMPLE_LIST = Path('/public5/home/xuejing/RNU/human/variants/1kG/gnomad.genomes.v3.1.2.hgdp_1kg_noRelated_sample.shuf2000.list')
genotypes_BASES = set('ACGT')
genotypes_AF_SCENARIOS = {'fixed_af_1e-3': 0.001, 'rhesus_af_2e-3': 0.002, 'mouse_af_1e-2': 0.01}
genotypes_merge_intervals = _shared.merge_intervals_06ce9ba0
genotypes_sha256 = _shared.sha256_file

def genotypes_load_parent_samples() -> list[str]:
    samples = [line.strip() for line in genotypes_PARENT_SAMPLE_LIST.read_text().splitlines()]
    samples = [sample for sample in samples if sample]
    if len(samples) != 2000 or len(set(samples)) != 2000:
        raise ValueError(f'Expected 2,000 unique parent samples; observed {len(samples)} rows and {len(set(samples))} unique IDs')
    return samples

def genotypes_genotype_counts(record: pysam.VariantRecord, sample_names: list[str], n_alts: int) -> tuple[list[int], int]:
    ac = [0] * n_alts
    an = 0
    for sample_name in sample_names:
        genotype = record.samples[sample_name].get('GT')
        if genotype is None:
            continue
        for allele in genotype:
            if allele is None:
                continue
            allele_index = int(allele)
            an += 1
            if allele_index > n_alts:
                raise ValueError(f'GT allele {allele_index} exceeds {n_alts} ALT alleles at {record.chrom}:{record.pos}')
            if allele_index > 0:
                ac[allele_index - 1] += 1
    return (ac, an)

def genotypes_fetch_cohort_alleles(cohorts: dict[str, list[str]], position_to_genes: dict[tuple[str, int], tuple[str, ...]], intervals_by_chrom: dict[str, list[tuple[int, int]]], index_dir: Path) -> tuple[dict[str, dict[tuple[str, int], list[dict[str, Any]]]], pd.DataFrame, pd.DataFrame]:
    alleles: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = {cohort: defaultdict(list) for cohort in cohorts}
    seen: dict[str, set[tuple[str, int, str, str]]] = {cohort: set() for cohort in cohorts}
    cohort_qc: dict[str, Counter[str]] = {cohort: Counter() for cohort in cohorts}
    index_rows: list[dict[str, Any]] = []
    for chrom_number in range(1, 23):
        chrom = f'chr{chrom_number}'
        if chrom not in intervals_by_chrom:
            continue
        vcf_path = genotypes_SOURCE_VCF_DIR / genotypes_SOURCE_VCF_TEMPLATE.format(chrom=chrom_number)
        index_path = index_dir / (vcf_path.name + '.tbi')
        if not vcf_path.exists():
            raise FileNotFoundError(vcf_path)
        if not index_path.exists():
            raise FileNotFoundError(f'Missing {index_path}; download from {genotypes_INDEX_URL_TEMPLATE.format(chrom=chrom_number)}')
        index_stat = index_path.stat()
        index_rows.append({'chrom': chrom, 'source_vcf': str(vcf_path), 'source_vcf_size_bytes': vcf_path.stat().st_size, 'index_path': str(index_path), 'index_url': genotypes_INDEX_URL_TEMPLATE.format(chrom=chrom_number), 'index_size_bytes': index_stat.st_size, 'index_sha256': genotypes_sha256(index_path)})
        print(f'[{chrom}] fetching {len(intervals_by_chrom[chrom]):,} intervals', flush=True)
        with pysam.VariantFile(str(vcf_path), index_filename=str(index_path)) as vcf:
            header_samples = set(vcf.header.samples)
            missing = sorted(set(cohorts['shuf2000_parent']) - header_samples)
            if missing:
                raise ValueError(f'{chrom} lacks parent samples: {missing[:20]}')
            for (start, end) in intervals_by_chrom[chrom]:
                for record in vcf.fetch(chrom, start, end):
                    position = (chrom, int(record.pos) - 1)
                    if position not in position_to_genes:
                        continue
                    alt_values = tuple(record.alts or ())
                    if not alt_values:
                        continue
                    ref = str(record.ref)
                    has_snp = any((len(ref) == 1 and len(str(alt)) == 1 for alt in alt_values))
                    if not has_snp:
                        continue
                    filter_pass = tuple(record.filter.keys()) == ('PASS',)
                    for (cohort, sample_names) in cohorts.items():
                        cohort_qc[cohort]['source_snp_records_examined'] += 1
                        (ac_values, an) = genotypes_genotype_counts(record, sample_names, len(alt_values))
                        if sum(ac_values) < 1:
                            continue
                        cohort_qc[cohort]['records_retained_after_c1'] += 1
                        for (alt_index, alt_value) in enumerate(alt_values):
                            alt = str(alt_value)
                            key = (chrom, position[1], ref, alt)
                            if key in seen[cohort]:
                                cohort_qc[cohort]['duplicate_alt_alleles'] += 1
                                continue
                            seen[cohort].add(key)
                            ac = int(ac_values[alt_index])
                            af = ac / an if an > 0 else None
                            valid_snv = len(ref) == 1 and len(alt) == 1 and (ref in genotypes_BASES) and (alt in genotypes_BASES) and (ref != alt)
                            alleles[cohort][position].append({'ref': ref, 'alt': alt, 'ac': ac, 'an': an, 'af': af, 'filter_pass': filter_pass, 'valid_snv': valid_snv})
                            cohort_qc[cohort]['retained_alt_alleles'] += 1
    qc_rows = []
    for (cohort, counts) in cohort_qc.items():
        qc_rows.append({'cohort': cohort, 'sample_count': len(cohorts[cohort]), **dict(counts), 'positions_with_retained_record': len(alleles[cohort])})
    return (alleles, pd.DataFrame(qc_rows), pd.DataFrame(index_rows))

def genotypes_classify_density(cohort: str, scenario: str, cohort_alleles: dict[tuple[str, int], list[dict[str, Any]]], position_to_genes: dict[tuple[str, int], tuple[str, ...]], current: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rare_alt_by_gene: Counter[str] = Counter()
    rare_position_by_gene: Counter[str] = Counter()
    mask_position_by_gene: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    qualifying_positions = 0
    qualifying_alt_alleles = 0
    masked_positions = 0
    for (position, position_alleles) in cohort_alleles.items():
        exclusion_reasons: set[str] = set()
        for allele in position_alleles:
            if not allele['filter_pass']:
                exclusion_reasons.add('non_PASS')
            if not allele['valid_snv']:
                exclusion_reasons.add('non_snv_or_invalid_alt')
            if scenario in genotypes_AF_SCENARIOS:
                af = allele['af']
                if af is None or not math.isfinite(af):
                    exclusion_reasons.add('missing_or_invalid_AF')
                elif af > genotypes_AF_SCENARIOS[scenario]:
                    exclusion_reasons.add('AF_above_threshold')
            elif scenario == 'ac_le_4_sensitivity':
                if allele['an'] <= 0:
                    exclusion_reasons.add('missing_or_invalid_AN')
                elif allele['ac'] > 4:
                    exclusion_reasons.add('AC_above_4')
            else:
                raise ValueError(f'Unknown scenario: {scenario}')
        genes = position_to_genes[position]
        if exclusion_reasons:
            masked_positions += 1
            for reason in exclusion_reasons:
                reasons[reason] += 1
            for gene_id in genes:
                mask_position_by_gene[gene_id] += 1
            continue
        alt_count = len(position_alleles)
        qualifying_positions += 1
        qualifying_alt_alleles += alt_count
        for gene_id in genes:
            rare_position_by_gene[gene_id] += 1
            rare_alt_by_gene[gene_id] += alt_count
    result = current[['stable_gene_id', 'display_name', 'analysis_group', 'exon_union_bp', 'callable_bp']].copy()
    result.insert(0, 'scenario', scenario)
    result.insert(0, 'cohort', cohort)
    result['sample_count'] = int(cohort.removeprefix('shuf').split('_')[0]) if cohort != 'shuf2000_parent' else 2000
    result['mask_bp'] = result['stable_gene_id'].map(mask_position_by_gene).fillna(0).astype(int)
    result['effective_callable_bp'] = result['callable_bp'] - result['mask_bp']
    result['rare_snv_position_count'] = result['stable_gene_id'].map(rare_position_by_gene).fillna(0).astype(int)
    result['rare_snv_alt_count'] = result['stable_gene_id'].map(rare_alt_by_gene).fillna(0).astype(int)
    result['rare_snv_density_per_kb'] = np.where(result['effective_callable_bp'].gt(0), 1000.0 * result['rare_snv_alt_count'] / result['effective_callable_bp'], np.nan)
    if (result['effective_callable_bp'] < 0).any():
        raise ValueError(f'Negative effective callable bp in {cohort}, {scenario}')
    qc = {'cohort': cohort, 'scenario': scenario, 'sample_count': int(result['sample_count'].iloc[0]), 'positions_with_retained_record': len(cohort_alleles), 'qualifying_positions': qualifying_positions, 'qualifying_alt_alleles': qualifying_alt_alleles, 'masked_positions': masked_positions, 'classification_closed': qualifying_positions + masked_positions == len(cohort_alleles), 'mask_reason_counts_json': json.dumps(dict(sorted(reasons.items())))}
    return (result, qc)
subsets_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/build_figS4_independent_20261005.py').parents[1]
subsets_TMP = subsets_ROOT / 'tmp/figS4_independent_20261005'
subsets_task = _component('revise_fig1_gene_rules_20261004')
subsets_OUT = subsets_ROOT / 'data/processed/figS4_independent_20261005'
subsets_DEL = subsets_ROOT / 'deliverables/figS4_independent_20261005'
subsets_OLD = subsets_task.OUT / 'supplements/FigS4/FigS4_four_group_density.tsv.gz'
subsets_CONTROL = subsets_task.TR / 'final_result/fig1/revised_human_autosomal_control_20260916/human_autosomal_control_1000.tsv'
subsets_MASK = subsets_task.TR / 'hg38.genmap.k150e1_unique.cov_16-48.bed'
subsets_SOURCE = _component_file('calculate_shuf2000_downsample_panel_c_folds')
subsets_original = _component('calculate_shuf2000_downsample_panel_c_folds')
subsets_G = subsets_task.GROUPS
subsets_SIZES = [500, 200, 100]
subsets_SEEDS = {500: 2026100501, 200: 2026100502, 100: 2026100503}
subsets_AF = {500: 0.002, 200: 0.01, 100: 0.02}
subsets_LABELS = dict(zip(subsets_G, ['POLII spliceRNUs', 'U6/U6ATAC', 'RNU pseudogenes', 'Control']))
(subsets_COHORTS, subsets_POSITIONS, subsets_INTERVALS, subsets_SIGNATURE) = ({}, {}, {}, {})
subsets_read = _shared.read_table

def subsets_save(frame, name):
    frame.to_csv(subsets_OUT / name, sep='\t', index=False, na_rep='NA')

def subsets_fetch_chromosome(chrom):
    cache = subsets_TMP / f'{chrom}.alleles.tsv.gz'
    marker = subsets_TMP / f'{chrom}.signature.json'
    if cache.exists() and marker.exists() and (json.loads(marker.read_text()) == subsets_SIGNATURE):
        return (subsets_read(cache), subsets_read(subsets_TMP / f'{chrom}.qc.tsv'), subsets_read(subsets_TMP / f'{chrom}.sources.tsv'))
    vcf = subsets_original.SOURCE_VCF_DIR / subsets_original.SOURCE_VCF_TEMPLATE.format(chrom=chrom[3:])
    index = subsets_original.DEFAULT_INDEX_DIR / (vcf.name + '.tbi')
    with pysam.VariantFile(str(vcf), index_filename=str(index)) as handle:
        assert set(subsets_original.load_parent_samples()) <= set(handle.header.samples)
    request = {'shuf2000_parent': [], **subsets_COHORTS}
    (alleles, qc, sources) = subsets_original.fetch_cohort_alleles(request, subsets_POSITIONS, {chrom: subsets_INTERVALS[chrom]}, subsets_original.DEFAULT_INDEX_DIR)
    rows = pd.DataFrame([dict(cohort=k, chrom=c, position0=p, **a) for (k, mapping) in alleles.items() for ((c, p), entries) in mapping.items() for a in entries])
    rows.to_csv(cache, sep='\t', index=False)
    qc = qc[qc.cohort.ne('shuf2000_parent')].copy()
    qc['chrom'] = chrom
    qc.to_csv(subsets_TMP / f'{chrom}.qc.tsv', sep='\t', index=False)
    sources.to_csv(subsets_TMP / f'{chrom}.sources.tsv', sep='\t', index=False)
    marker.write_text(json.dumps(subsets_SIGNATURE, indent=2) + '\n')
    print(f'{chrom}: finished {len(rows)} retained ALT entries', flush=True)
    return (rows, qc, sources)

def subsets_main():
    global subsets_COHORTS, subsets_POSITIONS, subsets_INTERVALS, subsets_SIGNATURE
    subsets_OUT.mkdir(parents=True, exist_ok=True)
    subsets_DEL.mkdir(parents=True, exist_ok=True)
    parent = subsets_original.load_parent_samples()
    old = subsets_read(subsets_OLD)
    base = old[old.dataset.eq('shuf500_r1')].copy()
    human = subsets_task.catalogs()['Human']
    controls = subsets_read(subsets_CONTROL).rename(columns={'neutral_region_id': 'stable_gene_id'}).set_index('stable_gene_id')
    coordinates = []
    for row in base.itertuples():
        is_control = row.analysis_group == subsets_G[-1]
        c = controls.loc[row.stable_gene_id] if is_control else human.loc[row.stable_gene_id]
        coordinates.append(dict(stable_gene_id=row.stable_gene_id, display_name=row.display_name, analysis_group=row.analysis_group, chrom=c.chrom, start=int(c.start), end=int(c.end), strand='.' if is_control else c.strand, exon_union_bp=int(c.end) - int(c.start), callable_bp=int(row.callable_bp)))
    targets = pd.DataFrame(coordinates)
    assert targets.groupby('analysis_group').size().to_dict() == dict(zip(subsets_G, [41, 116, 1362, 1000]))
    engine = subsets_task.profile_engine()
    ci = engine.load_callable_index(subsets_MASK)
    (positions, intervals) = (defaultdict(list), defaultdict(list))
    for r in targets.itertuples():
        assert engine.interval_callable_bp(ci, r.chrom, r.start, r.end) == r.callable_bp, r.stable_gene_id
        intervals[r.chrom].append((r.start, r.end))
        for pos in range(r.start, r.end):
            if engine.position_is_callable(ci, r.chrom, pos):
                positions[r.chrom, pos].append(r.stable_gene_id)
    subsets_POSITIONS = dict(positions)
    subsets_INTERVALS = {k: subsets_original.merge_intervals(v) for (k, v) in intervals.items()}
    subsets_save(targets, 'Human_fixed_locus_coordinates.tsv')
    memberships = []
    for n in subsets_SIZES:
        selected = set(np.random.default_rng(subsets_SEEDS[n]).choice(parent, size=n, replace=False))
        cohort = f'shuf{n}_r1'
        subsets_COHORTS[cohort] = [s for s in parent if s in selected]
        (subsets_OUT / f'human_random{n}_samples.list').write_text('\n'.join(subsets_COHORTS[cohort]) + '\n')
        for s in subsets_COHORTS[cohort]:
            memberships.append(dict(cohort=cohort, sample_size=n, draw=1, random_seed=subsets_SEEDS[n], sample_id=s, parent_sample_order=parent.index(s) + 1, sampling='Independent simple random sampling without replacement'))
    membership = pd.DataFrame(memberships)
    subsets_save(membership, 'FigS4_sample_membership.tsv')
    overlaps = []
    for (a, b) in combinations(subsets_SIZES, 2):
        (x, y) = (set(subsets_COHORTS[f'shuf{a}_r1']), set(subsets_COHORTS[f'shuf{b}_r1']))
        assert not (x <= y or y <= x)
        overlaps.append(dict(sample_size_1=a, sample_size_2=b, shared_individuals=len(x & y), nested=False))
    subsets_save(pd.DataFrame(overlaps), 'FigS4_sample_overlap.tsv')
    old_members = subsets_read(subsets_task.TR / 'final_result/figS4/random500_200_100_no_overlap_20260928/FigS4_sample_membership.tsv')
    subsets_COHORTS['shuf500_validation'] = old_members.loc[old_members.cohort.eq('shuf500_r1'), 'sample_id'].tolist()
    subsets_SIGNATURE = dict(seeds=subsets_SEEDS, cohort_ids=subsets_COHORTS, old_sha256=subsets_original.sha256(subsets_OLD), targets_sha256=subsets_original.sha256(subsets_OUT / 'Human_fixed_locus_coordinates.tsv'), policy='unchanged')
    subsets_SIGNATURE = json.loads(json.dumps(subsets_SIGNATURE))
    with ProcessPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(subsets_fetch_chromosome, sorted(subsets_INTERVALS, key=lambda c: int(c[3:]))))
    entries = pd.concat([c[0] for c in chunks], ignore_index=True)
    subsets_save(entries, 'Human_retained_alleles.tsv.gz')
    subsets_save(pd.concat([c[1] for c in chunks], ignore_index=True), 'fetch_qc.tsv')
    subsets_save(pd.concat([c[2] for c in chunks], ignore_index=True), 'genotype_source_manifest.tsv')
    alleles = {k: defaultdict(list) for k in subsets_COHORTS}
    for row in entries.to_dict('records'):
        (k, c, p) = (row.pop('cohort'), row.pop('chrom'), int(row.pop('position0')))
        if pd.isna(row['af']):
            row['af'] = None
        alleles[k][c, p].append(row)
    subsets_original.AF_SCENARIOS['rat_af_2e-2'] = 0.02
    scenarios = {500: 'rhesus_af_2e-3', 200: 'mouse_af_1e-2', 100: 'rat_af_2e-2'}
    (validation, _) = subsets_original.classify_density('shuf500_validation', scenarios[500], alleles['shuf500_validation'], subsets_POSITIONS, targets)
    cols = ['callable_bp', 'mask_bp', 'effective_callable_bp', 'rare_snv_alt_count', 'rare_snv_position_count', 'rare_snv_density_per_kb']
    a = validation.set_index('stable_gene_id').loc[base.stable_gene_id, cols]
    b = base.set_index('stable_gene_id').loc[base.stable_gene_id, cols]
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=0)
    subsets_save(pd.DataFrame([dict(column=c, max_absolute_difference=float((a[c] - b[c]).abs().max())) for c in cols]), 'Previous_random500_recount_validation.tsv')
    (parts, exclusions, qcs) = ([], [], [])
    for (i, n) in enumerate(subsets_SIZES):
        cohort = f'shuf{n}_r1'
        (density, qc) = subsets_original.classify_density(cohort, scenarios[n], alleles[cohort], subsets_POSITIONS, targets)
        qcs.append(qc)
        density = density.merge(targets[['stable_gene_id', 'chrom', 'start', 'end', 'strand']], on='stable_gene_id', validate='one_to_one')
        density['dataset'] = cohort
        (density['panel'], density['column_order']) = ('abc'[i], 1)
        (density['organism'], density['assembly'], density['af_threshold']) = ('Human', 'GRCh38', subsets_AF[n])
        density['display_dataset'] = f'Human random{n}'
        density['source_label'] = f'One independent sample of {n} unrelated HGDP+TGP individuals; seed {subsets_SEEDS[n]}'
        density['gene_set_label'] = density.analysis_group.map(subsets_LABELS)
        excluded = density[density.effective_callable_bp.lt(50)].copy()
        excluded['filter_reason'] = 'Effective callable length <50 bp'
        exclusions.append(excluded)
        parts.append(density[density.effective_callable_bp.ge(50)])
    nonhuman = old[~old.dataset.str.startswith('shuf')].copy()
    full = pd.concat([*parts, nonhuman], ignore_index=True)
    assert not full.duplicated(['dataset', 'stable_gene_id']).any()
    assert full.effective_callable_bp.ge(50).all()
    np.testing.assert_allclose(full.rare_snv_density_per_kb, 1000 * full.rare_snv_alt_count / full.effective_callable_bp, rtol=1e-13)
    (means, tests) = ([], [])
    for (dataset, frame) in full.groupby('dataset', sort=False):
        cm = frame.loc[frame.analysis_group.eq(subsets_G[-1]), 'rare_snv_density_per_kb'].mean()
        for g in subsets_G:
            v = frame.loc[frame.analysis_group.eq(g), 'rare_snv_density_per_kb']
            means.append(dict(dataset=dataset, analysis_group=g, n=len(v), mean_density=v.mean(), median_density=v.median(), zero_count=int(v.eq(0).sum()), control_mean_density=cm, mean_density_relative_to_control=v.mean() / cm))
        for (g1, g2) in combinations(subsets_G, 2):
            (x, y) = [frame.loc[frame.analysis_group.eq(g), 'rare_snv_density_per_kb'] for g in [g1, g2]]
            test = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic', use_continuity=True)
            tests.append(dict(dataset=dataset, group_1=g1, group_2=g2, n_1=len(x), n_2=len(y), mean_density_fold_change_1_over_2=x.mean() / y.mean(), mann_whitney_u=test.statistic, mann_whitney_p=test.pvalue))
    subsets_save(full, 'FigS4_four_group_density.tsv.gz')
    subsets_save(pd.DataFrame(means), 'FigS4_control_relative_means.tsv')
    subsets_save(pd.DataFrame(tests), 'FigS4_pairwise_unadjusted_P.tsv')
    subsets_save(pd.concat(exclusions, ignore_index=True), 'Human_subset_callable_exclusions.tsv')
    subsets_save(pd.DataFrame(qcs), 'classification_qc.tsv')
    report = dict(status='PASS', parent_pool_size=2000, seeds=subsets_SEEDS, draws_per_size=1, sampling='Independent sampling without replacement within each size; overlap allowed across sizes', cohort_overlaps=overlaps, previous_random500_all_loci_reproduced=True, fixed_gene_sets=[41, 116, 1362], fixed_intergenic_controls=1000, current_rat_pseudogenes=847, nonhuman_rows_unchanged=len(nonhuman), human_excluded_loci=sum((len(x) for x in exclusions)), variant_policy='Frozen record-level AC>=1; all retained ALT entries, including AC=0 siblings; position masking unchanged', command='python scripts/build_figS4_independent_20261005.py')
    (subsets_OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    paths = [subsets_OLD, subsets_CONTROL, subsets_MASK, subsets_SOURCE, subsets_original.PARENT_SAMPLE_LIST, subsets_ROOT / 'deliverables/human_snrna_metadata_20261003/Human_snRNA_metadata.tsv']
    subsets_save(pd.DataFrame([dict(path=str(p), sha256=subsets_original.sha256(p)) for p in paths]), 'input_manifest.tsv')
    print(json.dumps(report, indent=2), flush=True)

def subsets_entrypoint():
    subsets_main()

def _prepare_runtime():
    subsets_TMP.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(subsets_TMP)
    sys.pycache_prefix = str(subsets_TMP / 'pycache')

if __name__ == '__main__':
    _run_stages({'subsets': 'subsets_entrypoint'}, globals(), setup=_prepare_runtime)
