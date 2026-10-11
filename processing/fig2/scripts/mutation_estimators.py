"""Consolidated final processing: mutation_estimators.

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
from pathlib import Path
import hashlib
import json
import math
import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import musical
ROOT = _PROCESSING_ROOT / 'fig2'
HUMAN_POPULATION = os.environ.get('FIG2_HUMAN_POPULATION', 'ASJ').upper()
HUMAN_SHORT_LABEL = 'shuf2000' if HUMAN_POPULATION == 'SHUF2000' else HUMAN_POPULATION
HUMAN_POPULATION_LABEL = f'Human {HUMAN_SHORT_LABEL}'
GROUP_ORDER = ['Pol II splice-RNUs', 'U6', 'Pseudo-RNUs']
GENE_COUNTS = {('Human', 'Pol II splice-RNUs'): 41, ('Human', 'U6'): 116, ('Human', 'Pseudo-RNUs'): 1323, ('Macaque', 'Pol II splice-RNUs'): 28, ('Macaque', 'U6'): 193, ('Macaque', 'Pseudo-RNUs'): 1214, ('Mouse', 'Pol II splice-RNUs'): 39, ('Mouse', 'U6'): 63, ('Mouse', 'Pseudo-RNUs'): 1007}

def gene_set_label(species: str, group: str, *, stacked: bool=False) -> str:
    return {'Pol II splice-RNUs': 'POLII\nspliceRNUs' if stacked else 'POLII spliceRNUs', 'U6': 'U6/\nU6ATAC' if stacked else 'U6/U6ATAC', 'Pseudo-RNUs': 'PseudoRNUs'}[group]
HUMAN_LEGEND_LABELS = [gene_set_label('Human', group) for group in GROUP_ORDER]
HUMAN_STACKED_LABELS = [gene_set_label('Human', group, stacked=True) for group in GROUP_ORDER]
HUMAN_RESULT_SUMMARY_PATH = ROOT / 'data' / 'processed' / 'figure2_fig1_human_populations_20260825' / 'human_population_result_summary.tsv'
_human_result_summary = pd.read_csv(HUMAN_RESULT_SUMMARY_PATH, sep='\t')
_human_result_summary = _human_result_summary[_human_result_summary['population'].eq(HUMAN_POPULATION)].set_index('group').loc[GROUP_ORDER]
HUMAN_INPUTS = {group: ROOT / str(_human_result_summary.at[group, 'result_path']) for group in GROUP_ORDER}
HUMAN_SHA256 = {group: str(_human_result_summary.at[group, 'result_sha256']) for group in GROUP_ORDER}
HUMAN_RETAINED = {group: int(_human_result_summary.at[group, 'retained_rare_snv_records']) for group in GROUP_ORDER}
HUMAN_MASKED = {group: int(_human_result_summary.at[group, 'masked_variant_opportunities']) for group in GROUP_ORDER}
HUMAN_MAX_AF = float(_human_result_summary['max_af'].iloc[0])
HUMAN_VCF_PATH = ROOT / str(_human_result_summary['vcf_path'].iloc[0])
HUMAN_BED_PATHS = {group: ROOT / str(_human_result_summary.at[group, 'bed_path']) for group in GROUP_ORDER}
HUMAN_GENOME_FASTA = Path('/public/home/jiangzj/tools/annovar/humandb/hg38_seq/hg38.fa')
REFERENCE_3MER = Path('/public4/home/jiangzj/analysis/mtr/histone_mutation_explore/homosapi_hg38/mechanism-related-explore/mutation-spectrum-analysis/hg38.3-mer-count.txt')
COSMIC_SIGNATURES = Path('/public4/home/jiangzj/analysis/mtr/histone_mutation_explore/homosapi_hg38/mechanism-related-explore/mutation-spectrum-analysis/Signatures/assets/COSMIC_sigs.txt')
REFERENCE_SHA256 = '057e251e982269c6815a42b777af4abafc6476d327a9acce345b92d005812389'
COSMIC_SHA256 = '9e0d761a537187fa5056bb1e62af607efc103b9101e1480b12c2522bf261e10f'
sha256 = _shared.sha256_file

def wilson_interval(count: int, total: int) -> tuple[float, float]:
    if total <= 0 or count < 0 or count > total:
        raise ValueError(f'Invalid binomial count: {count}/{total}')
    z = 1.959963984540054
    proportion = count / total
    denominator = 1 + z ** 2 / total
    center = (proportion + z ** 2 / (2 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2)) / denominator
    return (max(0.0, center - half_width), min(1.0, center + half_width))

def bh_adjust(pvalues) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind='stable')
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result

def safe_fisher_after_rounding(cells: list[float]):
    values = np.asarray(cells, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        return (math.nan, math.nan, np.full((2, 2), -1, dtype=int))
    rounded = np.rint(values).astype(np.int64).reshape(2, 2)
    if rounded.sum() == 0:
        return (math.nan, math.nan, rounded)
    (odds_ratio, pvalue) = fisher_exact(rounded, alternative='two-sided')
    return (float(odds_ratio), float(pvalue), rounded)
for (group, path) in HUMAN_INPUTS.items():
    if not path.is_file():
        raise FileNotFoundError(f'Missing {HUMAN_POPULATION} input: {group} -> {path}')
for path in (REFERENCE_3MER, COSMIC_SIGNATURES):
    if not path.is_file():
        raise FileNotFoundError(f'Missing reference: {path}')
MUTATION_ORDER = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
COMPLEMENT_BASE = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

def historical_asymmetry_score(transcribed_density, nontranscribed_density):
    if transcribed_density == 0 and nontranscribed_density > 0:
        return -math.inf
    if nontranscribed_density == 0 and transcribed_density > 0:
        return math.inf
    if transcribed_density <= 0 or nontranscribed_density <= 0:
        return math.nan
    return -math.log2(nontranscribed_density / transcribed_density)

def analyze_panel_a():
    three_mer_rows = []
    for group in GROUP_ORDER:
        path = HUMAN_INPUTS[group]
        payload = json.loads(path.read_text(encoding='utf-8'))
        contexts = payload['3mer']['context']
        variants = payload['3mer']['variant']
        if len(contexts) != 64 or len(variants) != 192:
            raise ValueError(f'Unexpected 3-mer schema: {path}')
        for (variant, record) in variants.items():
            reference = variant[:3]
            total_site = int(contexts[reference])
            masked = int(record['mask'])
            raw_count = int(record['count'])
            callable_site = total_site - masked
            if callable_site <= 0 or raw_count > callable_site:
                raise ValueError(f'Invalid opportunity for {variant} in {path}')
            hazard = -math.log1p(-raw_count / callable_site)
            corrected_mutation = hazard * total_site
            one_mer = f'{variant[1]}>{variant[-2]}'
            if 'CG' in reference:
                one_mer += '(CpG)'
            three_mer_rows.append({'group': group, 'one_mer_mutation': one_mer, 'total_site': total_site, 'masked_count': masked, 'raw_count': raw_count, 'poisson_hazard': hazard, 'poisson_corrected_mutation': corrected_mutation, 'source_path': str(path.relative_to(ROOT)), 'source_sha256': sha256(path)})
    one_mer = pd.DataFrame(three_mer_rows).groupby(['group', 'one_mer_mutation', 'source_path', 'source_sha256'], as_index=False)[['total_site', 'masked_count', 'raw_count', 'poisson_corrected_mutation']].sum()
    one_mer['corrected_density'] = one_mer['poisson_corrected_mutation'] / one_mer['total_site']
    indexed = one_mer.set_index(['group', 'one_mer_mutation'])
    (asymmetry_rows, density_rows) = ([], [])
    for group in GROUP_ORDER:
        for mutation in MUTATION_ORDER:
            (reference_base, alternate_base) = mutation.split('>')
            partner = f'{COMPLEMENT_BASE[reference_base]}>{COMPLEMENT_BASE[alternate_base]}'
            transcribed = indexed.loc[group, mutation]
            nontranscribed = indexed.loc[group, partner]
            score = historical_asymmetry_score(float(transcribed['corrected_density']), float(nontranscribed['corrected_density']))
            fisher_cells = [float(transcribed['poisson_corrected_mutation']), float(transcribed['total_site'] - transcribed['poisson_corrected_mutation']), float(nontranscribed['poisson_corrected_mutation']), float(nontranscribed['total_site'] - nontranscribed['poisson_corrected_mutation'])]
            (odds_ratio, pvalue, rounded) = safe_fisher_after_rounding(fisher_cells)
            asymmetry_rows.append({'population': HUMAN_POPULATION_LABEL, 'max_af': HUMAN_MAX_AF, 'group': group, 'mutation': mutation, 'paired_mutation': partner, 'site_class': 'non-CpG', 'transcribed_total_site': int(transcribed['total_site']), 'transcribed_raw_count': int(transcribed['raw_count']), 'transcribed_poisson_corrected_mutation': float(transcribed['poisson_corrected_mutation']), 'transcribed_density': float(transcribed['corrected_density']), 'nontranscribed_total_site': int(nontranscribed['total_site']), 'nontranscribed_raw_count': int(nontranscribed['raw_count']), 'nontranscribed_poisson_corrected_mutation': float(nontranscribed['poisson_corrected_mutation']), 'nontranscribed_density': float(nontranscribed['corrected_density']), 'notebook_asymmetry_score_minus_log2_nontrans_over_trans': score, 'fisher_rounded_a': int(rounded[0, 0]), 'fisher_rounded_b': int(rounded[0, 1]), 'fisher_rounded_c': int(rounded[1, 0]), 'fisher_rounded_d': int(rounded[1, 1]), 'fisher_odds_ratio': odds_ratio, 'fisher_p_two_sided_raw': pvalue, 'significant_raw_p_0.05': bool(pvalue < 0.05)})
            density_rows.append({'population': HUMAN_POPULATION_LABEL, 'max_af': HUMAN_MAX_AF, 'group': group, 'mutation_pair': f'{mutation}&{partner}', 'site_class': 'non-CpG', 'notebook_merged_density': (float(transcribed['poisson_corrected_mutation']) + float(nontranscribed['poisson_corrected_mutation'])) / (float(transcribed['total_site']) + float(nontranscribed['total_site']))})
            if reference_base == 'C':
                transcribed_cpg = indexed.loc[group, f'{mutation}(CpG)']
                nontranscribed_cpg = indexed.loc[group, f'{partner}(CpG)']
                density_rows.append({'population': HUMAN_POPULATION_LABEL, 'max_af': HUMAN_MAX_AF, 'group': group, 'mutation_pair': f'{mutation}&{partner}', 'site_class': 'CpG', 'notebook_merged_density': (float(transcribed_cpg['poisson_corrected_mutation']) + float(nontranscribed_cpg['poisson_corrected_mutation'])) / (float(transcribed_cpg['total_site']) + float(nontranscribed_cpg['total_site']))})
    asymmetry = pd.DataFrame(asymmetry_rows)
    density = pd.DataFrame(density_rows)
    if len(asymmetry) != 18 or len(density) != 27:
        raise AssertionError('Panel a row count changed')
    pol2_ct = asymmetry.query("group == 'Pol II splice-RNUs' and mutation == 'C>T'").iloc[0]
    if not np.isfinite(float(pol2_ct['fisher_p_two_sided_raw'])):
        raise AssertionError('Panel a Pol II C>T Fisher test is not finite')
    for frame in (one_mer, asymmetry, density):
        frame['full_fig1_gene_count'] = frame['group'].map({group: GENE_COUNTS['Human', group] for group in GROUP_ORDER})
    return (pd.DataFrame(three_mer_rows), one_mer, asymmetry, density)
BASES = ('A', 'C', 'G', 'T')
SUBSTITUTION_ORDER = ('C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G')
SBS96_LABELS = tuple((f'{left}[{substitution}]{right}' for substitution in SUBSTITUTION_ORDER for left in BASES for right in BASES))
COMPLEMENT_TABLE = str.maketrans('ACGT', 'TGCA')
BASE_SIGNATURES = ['SBS1', 'SBS5']
TARGET_SIGNATURES = ['SBS2', 'SBS13']

def reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(COMPLEMENT_TABLE)[::-1]

def load_reference_3mer_counts(path: Path):
    counts = {}
    with path.open(encoding='utf-8') as handle:
        for (line_number, line) in enumerate(handle, start=1):
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 2:
                raise ValueError(f'{path}:{line_number}: expected >=2 fields')
            kmer = fields[0].upper()
            if len(kmer) != 3 or any((base not in 'ACGT' for base in kmer)):
                if line_number == 1:
                    continue
                raise ValueError(f'Invalid 3-mer: {kmer}')
            counts[kmer] = float(fields[1])
    if len(counts) != 64:
        raise ValueError(f'Expected 64 reference 3-mers, found {len(counts)}')
    return counts

def sbs96_components(label: str):
    (left, reference, alternate, right) = (label[0], label[2], label[4], label[-1])
    return (left + reference + right, left + alternate + right)

def legacy_result_to_sbs96(payload, reference_3mer):
    contexts = payload['3mer']['context']
    variants = payload['3mer']['variant']
    rows = []
    for label in SBS96_LABELS:
        (ref_kmer, alt_kmer) = sbs96_components(label)
        (ref_rc, alt_rc) = (reverse_complement(ref_kmer), reverse_complement(alt_kmer))
        (forward_key, reverse_key) = (f'{ref_kmer}>{alt_kmer}', f'{ref_rc}>{alt_rc}')
        raw_count = float(variants[forward_key]['count'] + variants[reverse_key]['count'])
        masked_count = float(variants[forward_key]['mask'] + variants[reverse_key]['mask'])
        target_opportunity = float(contexts[ref_kmer] + contexts[ref_rc])
        effective_opportunity = target_opportunity - masked_count
        reference_opportunity = float(reference_3mer[ref_kmer] + reference_3mer[ref_rc])
        if effective_opportunity < 0:
            raise ValueError(f'Negative effective opportunity: {label}')
        mutation_rate = raw_count / effective_opportunity if effective_opportunity else 0.0
        if mutation_rate >= 1:
            raise ValueError(f'Poisson hazard undefined for {label}')
        poisson_hazard = -math.log1p(-mutation_rate) if mutation_rate else 0.0
        rows.append({'MutationType': label, 'substitution': label[2:5], 'raw_count': raw_count, 'masked_count': masked_count, 'target_opportunity': target_opportunity, 'effective_opportunity': effective_opportunity, 'mutation_rate': mutation_rate, 'poisson_hazard': poisson_hazard, 'reference_opportunity': reference_opportunity, 'scaled_weight': poisson_hazard * reference_opportunity})
    profile = pd.DataFrame(rows)
    observed_total = float(profile['raw_count'].sum())
    scaled_total = float(profile['scaled_weight'].sum())
    profile['raw_fraction'] = profile['raw_count'] / observed_total
    profile['normalized_fraction'] = profile['scaled_weight'] / scaled_total
    profile['pseudo_count'] = profile['normalized_fraction'] * observed_total
    profile['normalization'] = 'notebook'
    matrix = profile.set_index('MutationType')[['pseudo_count']]
    return (profile, matrix)

def load_cosmic_signature_matrix(path: Path):
    signatures = pd.read_csv(path, sep='\t', index_col=0)
    signatures.index = signatures.index.astype(str)
    if signatures.index.has_duplicates or not set(SBS96_LABELS).issubset(signatures.index):
        raise ValueError('Invalid COSMIC SBS96 matrix')
    signatures = signatures.loc[list(SBS96_LABELS)].apply(pd.to_numeric)
    if (signatures < 0).any().any() or (signatures.sum(axis=0) <= 0).any():
        raise ValueError('Invalid COSMIC signature values')
    return signatures.div(signatures.sum(axis=0), axis=1)

def build_sbs96_analysis():
    reference_counts = load_reference_3mer_counts(REFERENCE_3MER)
    (detailed_parts, matrix_columns, summary_rows) = ([], {}, [])
    for group in GROUP_ORDER:
        path = HUMAN_INPUTS[group]
        payload = json.loads(path.read_text(encoding='utf-8'))
        (profile, one_column) = legacy_result_to_sbs96(payload, reference_counts)
        retained = int(round(profile['raw_count'].sum()))
        masked = int(round(profile['masked_count'].sum()))
        if retained != HUMAN_RETAINED[group] or masked != HUMAN_MASKED[group]:
            raise ValueError(f'{HUMAN_POPULATION} SBS96 totals changed for {group}: {retained}, {masked}')
        detailed = profile.copy()
        detailed.insert(0, 'population', HUMAN_POPULATION_LABEL)
        detailed.insert(1, 'max_af', HUMAN_MAX_AF)
        detailed.insert(2, 'group', group)
        detailed.insert(3, 'full_fig1_gene_count', GENE_COUNTS['Human', group])
        detailed['source_path'] = str(path.relative_to(ROOT))
        detailed['source_sha256'] = sha256(path)
        detailed_parts.append(detailed)
        matrix_columns[group] = one_column.iloc[:, 0]
        summary_rows.append({'population': HUMAN_POPULATION_LABEL, 'max_af': HUMAN_MAX_AF, 'group': group, 'full_fig1_gene_count': GENE_COUNTS['Human', group], 'retained_variants': retained, 'masked_variants': masked, 'total_variants': retained + masked, 'source_path': str(path.relative_to(ROOT)), 'source_sha256': sha256(path)})
    matrix = pd.DataFrame(matrix_columns)
    matrix.index.name = 'Mutation Types'
    profile_table = pd.concat(detailed_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    if matrix.shape != (96, 3) or not np.allclose(matrix.sum(), [HUMAN_RETAINED[g] for g in GROUP_ORDER]):
        raise AssertionError('SBS96 matrix validation failed')
    return (matrix, profile_table, summary)

def cosine_similarity(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else math.nan

def fit_signature_set(matrix, signatures, threshold=0.005, method='likelihood_bidirectional', connected_signatures=True):
    (exposures, _) = musical.refit.refit(matrix.astype(float), signatures.astype(float), thresh=threshold, method=method, connected_sigs=connected_signatures)
    exposures = exposures.reindex(index=signatures.columns, columns=matrix.columns).fillna(0)
    reconstruction = signatures @ exposures
    statistics_rows = []
    for sample in matrix.columns:
        observed = matrix[sample].to_numpy(float)
        fitted = reconstruction[sample].to_numpy(float)
        residual = observed - fitted
        statistics_rows.append({'sample': sample, 'engine': 'musical', 'cosine_similarity': cosine_similarity(observed, fitted), 'observed_total': float(observed.sum()), 'reconstructed_total': float(fitted.sum()), 'l1_residual': float(np.abs(residual).sum()), 'rmse': float(np.sqrt(np.mean(residual ** 2))), 'active_signatures': int((exposures[sample] > 0).sum())})
    normalized = exposures.div(exposures.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    return (exposures, normalized, reconstruction, pd.DataFrame(statistics_rows))

def run_signature_models(matrix, signatures):
    definitions = {'base': BASE_SIGNATURES, 'base_apobec': BASE_SIGNATURES + TARGET_SIGNATURES, 'all_cosmic': signatures.columns.tolist()}
    (statistics_parts, activity_rows, reconstruction_parts) = ([], [], [])
    for (model_name, signature_names) in definitions.items():
        (exposures, normalized, reconstruction, statistics) = fit_signature_set(matrix, signatures[signature_names])
        statistics.insert(1, 'model', model_name)
        statistics.insert(2, 'candidate_signature_count', len(signature_names))
        statistics_parts.append(statistics)
        for signature in exposures.index:
            for group in exposures.columns:
                activity_rows.append({'group': group, 'model': model_name, 'signature': signature, 'activity': float(exposures.at[signature, group]), 'proportion': float(normalized.at[signature, group])})
        rec = reconstruction.copy()
        rec.insert(0, 'Mutation Types', rec.index)
        rec = rec.melt(id_vars='Mutation Types', var_name='group', value_name='reconstructed_pseudo_count')
        rec.insert(1, 'model', model_name)
        reconstruction_parts.append(rec)
    return (pd.concat(statistics_parts, ignore_index=True), pd.DataFrame(activity_rows), pd.concat(reconstruction_parts, ignore_index=True))

def targeted_permutation_test(matrix, signatures, permutations=1000, seed=42):
    requested = BASE_SIGNATURES + TARGET_SIGNATURES
    alternatives = sorted(set(signatures.columns) - set(requested))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        (_, _, _, target_stats) = fit_signature_set(matrix, signatures[requested])
    warning_count = len(caught)
    target_cosine = target_stats.set_index('sample')['cosine_similarity']
    rng = random.Random(seed)
    null_rows = []
    for permutation in range(permutations):
        sampled = rng.sample(alternatives, len(TARGET_SIGNATURES))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            (_, _, _, stats) = fit_signature_set(matrix, signatures[BASE_SIGNATURES + sampled])
        warning_count += len(caught)
        for row in stats.itertuples(index=False):
            null_rows.append({'permutation': permutation, 'sample': row.sample, 'cosine_similarity': row.cosine_similarity, 'sampled_signatures': ','.join(sampled)})
    null = pd.DataFrame(null_rows)
    summary_rows = []
    for sample in matrix.columns:
        values = null.loc[null['sample'].eq(sample), 'cosine_similarity'].to_numpy(float)
        exceedances = int((values >= target_cosine[sample]).sum())
        summary_rows.append({'sample': sample, 'target_cosine': float(target_cosine[sample]), 'null_mean': float(np.mean(values)), 'null_95th_percentile': float(np.percentile(values, 95)), 'exceedances': exceedances, 'permutations': permutations, 'empirical_p': exceedances / permutations, 'pvalue_method': 'notebook', 'seed': seed})
    return (pd.DataFrame(summary_rows), null, {'captured_warnings': warning_count})

def _scalar_info(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        return value[0]
    return value

def analyze_shuf2000_unique_sites(group):
    """Count unique rare-variant sites after site-level masking.

    The legacy result format stores counts and masks per alternative allele.  At
    shuf2000 density, summing three possible substitutions can exceed the number
    of nucleotide positions, violating Wilson/Fisher binomial assumptions.  For
    panels d/e only, collapse records by position within each fixed Figure 1 BED
    occurrence; exclude a position if any record is masked, and otherwise count
    it once when one or more retained rare alleles are present.
    """
    if HUMAN_POPULATION != 'SHUF2000':
        raise RuntimeError('Unique-site aggregation is only defined for SHUF2000')
    if group in SHUF_SITE_CONTEXT_CACHE:
        return SHUF_SITE_CONTEXT_CACHE[group]
    from cyvcf2 import VCF
    import pysam
    complement = str.maketrans('ACGT', 'TGCA')
    rows = {context: {'context': context, 'total_sites': 0, 'masked_sites': 0, 'rare_variant_sites': 0} for left in 'ATCG' for center in 'ATCG' for right in 'ATCG' for context in [left + center + right]}
    vcf = VCF(str(HUMAN_VCF_PATH))
    fasta = pysam.FastaFile(str(HUMAN_GENOME_FASTA))
    try:
        for (line_number, line) in enumerate(HUMAN_BED_PATHS[group].read_text(encoding='utf-8').splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split('\t')
            if len(fields) < 4 or fields[3] not in {'+', '-'}:
                raise ValueError(f'Invalid strand BED row {line_number}: {HUMAN_BED_PATHS[group]}')
            (chrom, start_text, end_text, strand) = fields[:4]
            (start, end) = (int(start_text), int(end_text))
            sequence = fasta.fetch(chrom, start - 1, end + 1).upper()
            if strand == '-':
                sequence = sequence.translate(complement)[::-1]
            site_state = {}
            for record in vcf(f'{chrom}:{start + 1}-{end}'):
                if len(record.REF) != 1 or not record.ALT or any((len(alt) != 1 for alt in record.ALT)):
                    continue
                ac = _scalar_info(record.INFO.get('AC'))
                an = _scalar_info(record.INFO.get('AN'))
                if ac is None or an in (None, 0):
                    allele_frequency = 0.0
                else:
                    allele_frequency = float(ac) / float(an)
                state = site_state.setdefault(int(record.POS), {'rare': False, 'masked': False})
                if record.FILTER is not None or allele_frequency > HUMAN_MAX_AF:
                    state['masked'] = True
                else:
                    state['rare'] = True
            interval_length = end - start
            if len(sequence) != interval_length + 2:
                raise AssertionError(f'Reference fetch length changed: {chrom}:{start}-{end}')
            for offset in range(interval_length):
                context = sequence[offset:offset + 3]
                if context not in rows:
                    continue
                position = start + offset + 1 if strand == '+' else end - offset
                state = site_state.get(position, {'rare': False, 'masked': False})
                rows[context]['total_sites'] += 1
                if state['masked']:
                    rows[context]['masked_sites'] += 1
                elif state['rare']:
                    rows[context]['rare_variant_sites'] += 1
    finally:
        vcf.close()
        fasta.close()
    payload = json.loads(HUMAN_INPUTS[group].read_text(encoding='utf-8'))
    expected_contexts = {key: int(value) for (key, value) in payload['3mer']['context'].items()}
    observed_contexts = {key: int(value['total_sites']) for (key, value) in rows.items()}
    if observed_contexts != expected_contexts:
        raise AssertionError(f'SHUF2000 site contexts do not reproduce the frozen result for {group}')
    for row in rows.values():
        row['callable_sites'] = row['total_sites'] - row['masked_sites']
        if row['rare_variant_sites'] > row['callable_sites']:
            raise AssertionError(f'Invalid unique-site binomial count for {group}: {row}')
        row['group'] = group
        row['aggregation'] = 'unique position; exclude site if any ALT is masked'
    SHUF_SITE_CONTEXT_CACHE[group] = rows
    return rows

def summarize_shuf2000_site_contexts(group, contexts_to_include):
    rows = analyze_shuf2000_unique_sites(group)
    selected = [rows[context] for context in contexts_to_include]
    context_positions = sum((row['total_sites'] for row in selected))
    masked = sum((row['masked_sites'] for row in selected))
    count = sum((row['rare_variant_sites'] for row in selected))
    callable_sites = context_positions - masked
    (low, high) = wilson_interval(count, callable_sites)
    return {'included_contexts': ','.join(sorted(contexts_to_include)), 'variant_count': count, 'reference_context_positions': context_positions, 'masked_variant_opportunities': masked, 'callable_sites': callable_sites, 'rare_snv_rate': count / callable_sites, 'rare_snv_percent': 100 * count / callable_sites, 'wilson_95ci_low_percent': 100 * low, 'wilson_95ci_high_percent': 100 * high, 'aggregation': 'unique position; exclude site if any ALT is masked'}

def parse_strict_tcw(path):
    if HUMAN_POPULATION == 'SHUF2000' and path in HUMAN_INPUTS.values():
        group = next((group for (group, input_path) in HUMAN_INPUTS.items() if input_path == path))
        data = summarize_shuf2000_site_contexts(group, {'TCA', 'TCT'})
        return {'tcw_count': data['variant_count'], 'tcw_context': data['reference_context_positions'], 'tcw_mask': data['masked_variant_opportunities'], 'tcw_callable': data['callable_sites'], 'aggregation': data['aggregation']}
    payload = json.loads(path.read_text(encoding='utf-8'))
    (contexts, variants) = (payload['3mer']['context'], payload['3mer']['variant'])
    if len(contexts) != 64 or len(variants) != 192:
        raise ValueError(f'Unexpected 3-mer schema: {path}')
    records = [record for (change, record) in variants.items() if change.split('>', 1)[0] in {'TCA', 'TCT'}]
    if len(records) != 6:
        raise ValueError(f'Expected six strict-TCW substitutions: {path}')
    tcw_context = int(contexts['TCA']) + int(contexts['TCT'])
    tcw_count = sum((int(record['count']) for record in records))
    tcw_mask = sum((int(record['mask']) for record in records))
    tcw_callable = tcw_context - tcw_mask
    return {'tcw_count': tcw_count, 'tcw_context': tcw_context, 'tcw_mask': tcw_mask, 'tcw_callable': tcw_callable, 'aggregation': 'legacy mutation-record sum'}

def analyze_panel_d():
    rows = []
    for species in SPECIES_ORDER:
        for group in GROUP_ORDER:
            key = (species, group)
            path = CROSS_SPECIES_INPUTS[key]
            stats = parse_strict_tcw(path)
            observed = (stats['tcw_count'], stats['tcw_context'], stats['tcw_mask'], stats['tcw_callable'])
            if key in EXPECTED_TCW and observed != EXPECTED_TCW[key]:
                raise AssertionError(f'TCW regression changed for {key}: {observed}')
            (low, high) = wilson_interval(stats['tcw_count'], stats['tcw_callable'])
            rows.append({'species': species, 'group': group, 'full_fig1_gene_count': GENE_COUNTS[species, group], **stats, 'tcw_rate': stats['tcw_count'] / stats['tcw_callable'], 'tcw_percent': 100 * stats['tcw_count'] / stats['tcw_callable'], 'wilson_95ci_low_percent': 100 * low, 'wilson_95ci_high_percent': 100 * high, 'source_path': str(path.relative_to(ROOT)), 'source_sha256': sha256(path)})
    data = pd.DataFrame(rows)
    indexed = data.set_index(['species', 'group'])
    tests = []
    for species in SPECIES_ORDER:
        target = indexed.loc[species, 'Pol II splice-RNUs']
        for control_name in CONTROL_ORDER:
            control = indexed.loc[species, control_name]
            (target_count, target_total) = (int(target['tcw_count']), int(target['tcw_callable']))
            (control_count, control_total) = (int(control['tcw_count']), int(control['tcw_callable']))
            (odds_ratio, pvalue) = fisher_exact([[target_count, target_total - target_count], [control_count, control_total - control_count]], alternative='two-sided')
            (target_rate, control_rate) = (target_count / target_total, control_count / control_total)
            tests.append({'species': species, 'target': 'Pol II splice-RNUs', 'control': control_name, 'target_full_fig1_gene_count': GENE_COUNTS[species, 'Pol II splice-RNUs'], 'control_full_fig1_gene_count': GENE_COUNTS[species, control_name], 'target_count': target_count, 'target_callable': target_total, 'control_count': control_count, 'control_callable': control_total, 'target_percent': 100 * target_rate, 'control_percent': 100 * control_rate, 'fold_change_pol2_over_control': target_rate / control_rate if control_rate else math.inf, 'odds_ratio': float(odds_ratio), 'fisher_p_two_sided': float(pvalue)})
    tests = pd.DataFrame(tests)
    tests['bh_q_six_planned_tests'] = bh_adjust(tests['fisher_p_two_sided'])
    tests['significant_bh_0.05'] = tests['bh_q_six_planned_tests'] < 0.05
    return (data, tests)

def _prepare_runtime():
    pass
