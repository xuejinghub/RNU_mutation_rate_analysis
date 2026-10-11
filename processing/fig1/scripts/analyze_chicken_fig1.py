"""Consolidated final processing: analyze_chicken_fig1.

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
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import random
import subprocess
import shutil
import numpy as np
import pandas as pd
import pysam
from scipy.stats import mannwhitneyu
chicken_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/analyze_chicken_fig1.py').parents[1]
chicken_TMP = chicken_ROOT / 'tmp/chicken_fig1_20260924'
chicken_CLASS = chicken_ROOT / 'results/chicken_snrna_GRCg7b_complement_20260924'
chicken_OUT = chicken_ROOT / 'data/processed/fig1_chicken_GRCg7b_complement_20260924'
chicken_BASE = chicken_ROOT / 'primate/data/processed/fig1_chimpanzee_syntenic_pol2_af0p01_20260923'
chicken_VCF = chicken_ROOT / 'chicken_variants_191ind.GQ20_missing.vcf.gz'
chicken_CALL = chicken_ROOT / 'GRCg7b.genmap.k150e1_unique.cov_52-156.bed'
chicken_FA = chicken_TMP / 'GRCg7b.fa'
chicken_GTF = chicken_ROOT / 'data/raw/Gallus_gallus.GRCg7b.115.gtf.gz'
chicken_GROUPS = ('pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes', 'genome_wide_neutral')
chicken_AF_MAX = 0.01
chicken_MIN_AN = 300
chicken_MIN_EFFECTIVE = 50
chicken_N_CONTROLS = 1000
chicken_sha = _shared.sha256_file

def chicken_merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for (s, e) in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]))
        else:
            out.append((s, e))
    return out

def chicken_load_bed(path: Path) -> dict[str, tuple[list[int], list[int]]]:
    raw = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if line.startswith('#') or not line.strip():
                continue
            f = line.split('\t')
            (chrom, s, e) = (f[0], int(f[1]), int(f[2]))
            assert 0 <= s < e, line
            raw[chrom].append((s, e))
    return {c: ([x[0] for x in m], [x[1] for x in m]) for (c, rows) in raw.items() if (m := chicken_merge(rows))}

def chicken_overlap_bp(index: dict, chrom: str, s: int, e: int) -> int:
    (begins, ends) = index.get(chrom, ([], []))
    i = bisect_right(ends, s)
    total = 0
    while i < len(begins) and begins[i] < e:
        total += max(0, min(e, ends[i]) - max(s, begins[i]))
        i += 1
    return total

def chicken_within(index: dict, chrom: str, pos0: int) -> bool:
    (begins, ends) = index.get(chrom, ([], []))
    i = bisect_right(begins, pos0) - 1
    return i >= 0 and pos0 < ends[i]

def chicken_all_genes_exclusion() -> dict:
    raw = defaultdict(list)
    with gzip.open(chicken_GTF, 'rt') as handle:
        for line in handle:
            f = line.split('\t')
            if len(f) == 9 and f[2] == 'gene' and f[0].isdigit() and (1 <= int(f[0]) <= 39):
                raw[f[0]].append((max(0, int(f[3]) - 1 - 10000), int(f[4]) + 10000))
    return {c: ([x[0] for x in m], [x[1] for x in m]) for (c, rows) in raw.items() if (m := chicken_merge(rows))}

def chicken_variant_info(rec, seq: str, region_start: int) -> dict:
    reasons = []
    if len(rec.ref) != 1 or rec.ref not in 'ACGT' or any((len(a) != 1 or a not in 'ACGT' for a in rec.alts or ())):
        reasons.append('noncanonical')
    if not region_start <= rec.start < region_start + len(seq) or seq[rec.start - region_start] != rec.ref:
        reasons.append('reference_mismatch')
    filters = set(rec.filter.keys())
    if filters not in (set(), {'PASS'}):
        reasons.append('site_filter')
    if rec.qual is None or rec.qual < 30:
        reasons.append('QUAL_lt30')
    for (key, op, threshold) in (('QD', 'min', 2), ('MQ', 'min', 40), ('FS', 'max', 60), ('SOR', 'max', 3), ('ReadPosRankSum', 'min', -8)):
        val = rec.info.get(key)
        if val is None or (op == 'min' and val < threshold) or (op == 'max' and val > threshold):
            reasons.append(f'{key}_failed_or_missing')
    an = rec.info.get('AN')
    ac = rec.info.get('AC')
    if an is None or an < chicken_MIN_AN or an > 382 or (not ac) or any((v is None or v < 0 for v in ac)):
        reasons.append('AC_AN_invalid_or_low')
    return dict(pos0=rec.start, ref=rec.ref, alts=tuple(rec.alts or ()), ac=tuple(ac or ()), an=an, reasons=reasons)

def chicken_fetch_site_calls(vf, fa, chrom: str, s: int, e: int) -> tuple[dict, Counter]:
    seq = fa.fetch(chrom, s, e).upper()
    records = defaultdict(list)
    qc = Counter()
    for rec in vf.fetch(chrom, s, e):
        if not s <= rec.start < e:
            continue
        records[rec.start].append(chicken_variant_info(rec, seq, s))
        qc['records'] += 1
    calls = {}
    for (pos, siblings) in records.items():
        reasons = {reason for r in siblings for reason in r['reasons']}
        refs = {r['ref'] for r in siblings}
        ans = {r['an'] for r in siblings}
        alt_ac = {}
        for r in siblings:
            for (alt, ac) in zip(r['alts'], r['ac']):
                if alt in alt_ac and alt_ac[alt] != ac:
                    reasons.add('discordant_split_ALT')
                alt_ac[alt] = ac
        if len(refs) != 1 or len(ans) != 1 or (not all((x is not None for x in ans))):
            reasons.add('discordant_split_site')
        if not reasons:
            an = next(iter(ans))
            ref_ac = an - sum(alt_ac.values())
            if ref_ac < 0:
                reasons.add('AC_sum_gt_AN')
            else:
                counts = {'REF': ref_ac, **alt_ac}
                major = max(counts.values())
                if sum((ac == major for ac in counts.values())) != 1:
                    reasons.add('ambiguous_major_allele')
                nonmajor = {key: ac for (key, ac) in counts.items() if 0 < ac < major}
                if any((ac / an > chicken_AF_MAX for ac in nonmajor.values())):
                    reasons.add('nonmajor_AF_gt_threshold')
                if not reasons:
                    rare_ref = int('REF' in nonmajor)
                    rare_alt = sum((key != 'REF' for key in nonmajor))
                    calls[pos] = (False, rare_ref, rare_alt, '')
                    qc['counted_rare_REF'] += rare_ref
                    qc['counted_rare_ALT'] += rare_alt
        if reasons:
            calls[pos] = (True, 0, 0, ';'.join(sorted(reasons)))
            qc.update(reasons)
    return (calls, qc)

def chicken_interval_stats(fa, chrom: str, s: int, e: int, calls: dict, callable_index: dict) -> dict:
    seq = fa.fetch(chrom, s, e).upper()
    cbp = chicken_overlap_bp(callable_index, chrom, s, e)
    non_acgt_callable = sum((chicken_within(callable_index, chrom, s + j) and b not in 'ACGT' for (j, b) in enumerate(seq)))
    mask = rare_ref = rare_alt = positions = 0
    for (pos, (excluded, ref_count, alt_count, _)) in calls.items():
        if s <= pos < e and chicken_within(callable_index, chrom, pos) and (seq[pos - s] in 'ACGT'):
            if excluded:
                mask += 1
            else:
                rare_ref += ref_count
                rare_alt += alt_count
                positions += int(ref_count + alt_count > 0)
    effective = cbp - non_acgt_callable - mask
    assert effective >= 0
    count = rare_ref + rare_alt
    return dict(region_bp=e - s, callable_bp=cbp - non_acgt_callable, mask_bp=mask, effective_callable_bp=effective, effective_unmasked_bp=np.nan, rare_snv_alt_count=count, rare_snv_position_count=positions, rare_ref_count=rare_ref, rare_alt_count=rare_alt, rare_snv_allele_count=count, rare_snv_density_per_kb=1000 * count / effective if effective else np.nan)

def chicken_controls(fa, vf, callable_index: dict) -> tuple[list[dict], Counter]:
    excluded = chicken_all_genes_exclusion()
    chroms = [str(i) for i in range(1, 40) if str(i) in fa.references]
    weights = [fa.get_reference_length(c) for c in chroms]
    rng = random.Random(20260924)
    (selected, used) = ([], defaultdict(list))
    qc = Counter()
    for attempt in range(1, 2000001):
        chrom = rng.choices(chroms, weights=weights)[0]
        s = rng.randrange(0, fa.get_reference_length(chrom) - 150)
        e = s + 150
        if chicken_overlap_bp(excluded, chrom, s, e):
            continue
        if chicken_overlap_bp({chrom: ([x[0] for x in used[chrom]], [x[1] for x in used[chrom]])}, chrom, s, e):
            continue
        if chicken_overlap_bp(callable_index, chrom, s, e) < 100:
            continue
        if any((x not in 'ACGT' for x in fa.fetch(chrom, s, e).upper())):
            continue
        (calls, site_qc) = chicken_fetch_site_calls(vf, fa, chrom, s, e)
        stats = chicken_interval_stats(fa, chrom, s, e, calls, callable_index)
        if stats['effective_callable_bp'] < chicken_MIN_EFFECTIVE:
            continue
        unit_id = f'chicken_control_{len(selected) + 1:04d}'
        selected.append(dict(chrom=chrom, start=s, end=e, strand='+', unit_id=unit_id, display_name=unit_id, analysis_group=chicken_GROUPS[3]))
        used[chrom] = chicken_merge(used[chrom] + [(s, e)])
        qc.update(site_qc)
        if len(selected) == chicken_N_CONTROLS:
            qc['candidate_attempts'] = attempt
            break
    assert len(selected) == chicken_N_CONTROLS
    return (selected, qc)

def chicken_profile_bins(gene: dict) -> list[tuple[int, int, str, int, int]]:
    (s, e, strand) = (gene['start'], gene['end'], gene['strand'])
    edges = np.rint(np.linspace(s, e, 21)).astype(int)
    assert np.all(np.diff(edges) > 0)
    bins = []
    for k in range(40):
        (a, b) = (s - 2000 + 50 * k, s - 1950 + 50 * k) if strand == '+' else (e + 1950 - 50 * k, e + 2000 - 50 * k)
        bins.append((a, b, 'upstream', -2000 + 50 * k, -1950 + 50 * k))
    for k in range(20):
        j = k if strand == '+' else 19 - k
        bins.append((int(edges[j]), int(edges[j + 1]), 'body', k * 50, (k + 1) * 50))
    for k in range(40):
        (a, b) = (e + 50 * k, e + 50 * (k + 1)) if strand == '+' else (s - 50 * (k + 1), s - 50 * k)
        bins.append((a, b, 'downstream', 1000 + 50 * k, 1050 + 50 * k))
    return bins

def chicken_main() -> None:
    chicken_OUT.mkdir(parents=True, exist_ok=True)
    chicken_TMP.mkdir(parents=True, exist_ok=True)
    if not chicken_FA.is_file():
        source = chicken_ROOT / 'data/raw/Gallus_gallus.GRCg7b.dna.toplevel.fa.gz'
        with gzip.open(source, 'rb') as src, chicken_FA.open('wb') as dest:
            shutil.copyfileobj(src, dest)
    if not Path(str(chicken_FA) + '.fai').is_file():
        subprocess.run(['samtools', 'faidx', str(chicken_FA)], check=True)
    if not Path(str(chicken_VCF) + '.csi').is_file():
        subprocess.run(['bcftools', 'index', '-c', str(chicken_VCF)], check=True)
    catalogue = pd.read_csv(chicken_CLASS / 'all_ensembl_snRNA_82.tsv', sep='\t', dtype={'chrom': str})
    allowed = catalogue[catalogue.analysis_group.isin(chicken_GROUPS[:3])].copy()
    assert allowed.chrom.str.fullmatch('[1-9][0-9]*').all()
    fa = pysam.FastaFile(str(chicken_FA))
    callable_index = chicken_load_bed(chicken_CALL)
    assert fa.get_reference_length('1') == 196449156
    with pysam.VariantFile(str(chicken_VCF)) as vf:
        assert not vf.header.samples and vf.header.contigs['1'].length == fa.get_reference_length('1')
        (controls_list, control_qc) = chicken_controls(fa, vf, callable_index)
        pd.DataFrame(controls_list).to_csv(chicken_OUT / 'chicken_control_1000.tsv', sep='\t', index=False)
        all_units = allowed.to_dict('records') + controls_list
        (densities, bin_rows, variant_qc) = ([], [], Counter())
        for r in all_units:
            (chrom, s, e) = (str(r['chrom']), int(r['start']), int(r['end']))
            flank = 0 if r['analysis_group'] == chicken_GROUPS[3] else 2000
            assert s - flank >= 0 and e + flank <= fa.get_reference_length(chrom)
            (calls, qc) = chicken_fetch_site_calls(vf, fa, chrom, s - flank, e + flank)
            variant_qc.update(qc)
            vals = chicken_interval_stats(fa, chrom, s, e, calls, callable_index)
            densities.append(dict(species='Chicken', assembly='GRCg7b', analysis_group=r['analysis_group'], unit_id=r['unit_id'] if 'unit_id' in r else r['gene_id'], display_name=r['display_name'] if 'display_name' in r else r['gene_name'], chrom=chrom, start=s, end=e, strand=r['strand'], **vals, af_threshold=chicken_AF_MAX, denominator_basis='GRCg7b_callable_ACGT_minus_position_masks', source_note='191 chickens; GQ20 AC/AN; AN>=300; site QC; AF<=0.01; callable BED'))
            if r['analysis_group'] == chicken_GROUPS[3]:
                continue
            for (k, (a, b, region, rs, re_)) in enumerate(chicken_profile_bins(r)):
                values = chicken_interval_stats(fa, chrom, a, b, calls, callable_index)
                bin_rows.append(dict(species='Chicken', analysis_group=r['analysis_group'], unit_id=r['gene_id'], bin_index=k, region=region, relative_start=rs, relative_end=re_, chrom=chrom, genomic_start=a, genomic_end=b, **values))
    fa.close()
    raw_density = pd.DataFrame(densities)
    raw_density.to_csv(chicken_OUT / 'chicken_all_candidate_density_before_callable_filter.tsv', sep='\t', index=False)
    keep = raw_density[(raw_density.analysis_group == chicken_GROUPS[3]) | (raw_density.analysis_group == chicken_GROUPS[2]) & (raw_density.effective_callable_bp > 0) | raw_density.analysis_group.isin(chicken_GROUPS[:2]) & (raw_density.effective_callable_bp >= chicken_MIN_EFFECTIVE)].copy()
    assert len(keep[keep.analysis_group == chicken_GROUPS[3]]) == chicken_N_CONTROLS
    valid_ids = set(keep.unit_id)
    gb = pd.DataFrame(bin_rows)
    gb = gb[gb.unit_id.isin(valid_ids)]
    gb.to_csv(chicken_OUT / 'chicken_profile_gene_bins.tsv.gz', sep='\t', index=False)
    keep.to_csv(chicken_OUT / 'chicken_gene_locus_density.tsv', sep='\t', index=False)
    control = keep[keep.analysis_group == chicken_GROUPS[3]]
    control_mean = float(control.rare_snv_density_per_kb.mean())
    control_pooled = float(1000 * control.rare_snv_alt_count.sum() / control.effective_callable_bp.sum())
    assert control_mean > 0 and control_pooled > 0
    bases = {name: pd.read_csv(chicken_BASE / name, sep='\t', float_precision='round_trip') for name in ('panel_a_gene_locus_density.tsv', 'panel_a_pairwise_statistics.tsv', 'panel_b_control_mean_ratios.tsv', 'panel_c_profile_control_ratios.tsv', 'control_denominators.tsv')}
    density = pd.concat([bases['panel_a_gene_locus_density.tsv'], keep], ignore_index=True)
    stats = []
    for other in chicken_GROUPS[1:]:
        a = keep[keep.analysis_group == chicken_GROUPS[0]].rare_snv_density_per_kb
        b = keep[keep.analysis_group == other].rare_snv_density_per_kb
        if len(a) and len(b):
            res = mannwhitneyu(a, b, alternative='two-sided', method='asymptotic', use_continuity=True)
            stats.append(dict(species='Chicken', group_1=chicken_GROUPS[0], group_2=other, n1=len(a), n2=len(b), U=res.statistic, p_value=res.pvalue, mean_density_fold=a.mean() / b.mean() if b.mean() else np.nan))
    stats = pd.concat([bases['panel_a_pairwise_statistics.tsv'], pd.DataFrame(stats)], ignore_index=True)
    bars = []
    for g in chicken_GROUPS:
        sub = keep[keep.analysis_group == g]
        if len(sub):
            m = float(sub.rare_snv_density_per_kb.mean())
            bars.append(dict(species='Chicken', analysis_group=g, n_units=len(sub), mean_density_per_kb=m, mean_density_ratio_to_control=m / control_mean))
    bars = pd.concat([bases['panel_b_control_mean_ratios.tsv'], pd.DataFrame(bars)], ignore_index=True)
    prof = gb.groupby(['species', 'analysis_group', 'bin_index', 'region', 'relative_start', 'relative_end'], as_index=False).agg(gene_count=('unit_id', 'size'), contributing_genes=('effective_callable_bp', lambda x: int((x > 0).sum())), total_rare_snv_alt_count=('rare_snv_alt_count', 'sum'), total_effective_callable_bp=('effective_callable_bp', 'sum'), rare_REF=('rare_ref_count', 'sum'), rare_ALT=('rare_alt_count', 'sum'))
    prof['pooled_density_per_kb'] = np.where(prof.total_effective_callable_bp > 0, 1000 * prof.total_rare_snv_alt_count / prof.total_effective_callable_bp, np.nan)
    prof['density_ratio_to_control'] = prof.pooled_density_per_kb / control_pooled
    prof['gene_set_label'] = prof.analysis_group.map({chicken_GROUPS[0]: 'POLII spliceRNUs', chicken_GROUPS[1]: 'U6/U6ATAC', chicken_GROUPS[2]: 'PseudoRNUs'})
    profile = pd.concat([bases['panel_c_profile_control_ratios.tsv'], prof], ignore_index=True)
    cd = pd.concat([bases['control_denominators.tsv'], pd.DataFrame([dict(species='Chicken', control_mean_density_per_kb=control_mean, control_pooled_density_per_kb=control_pooled, denominator_column='effective_callable_bp')])], ignore_index=True)
    outputs = dict(panel_a_gene_locus_density=density, panel_a_pairwise_statistics=stats, panel_b_control_mean_ratios=bars, panel_c_profile_control_ratios=profile, control_denominators=cd)
    for (name, frame) in outputs.items():
        original = bases[name + '.tsv']
        pd.testing.assert_frame_equal(original.reset_index(drop=True), frame.loc[frame.species.ne('Chicken'), original.columns].reset_index(drop=True), check_exact=True, check_dtype=False)
        frame.to_csv(chicken_OUT / (name + '.tsv'), sep='\t', index=False)
    summary = dict(chicken_plotted_groups=keep.groupby('analysis_group').size().to_dict(), chicken_all_candidates=raw_density.groupby('analysis_group').size().to_dict(), functional_candidates_below_50bp=int(len(raw_density[raw_density.analysis_group.isin(chicken_GROUPS[:2]) & (raw_density.effective_callable_bp < chicken_MIN_EFFECTIVE)])), pseudogene_zero_effective_callable=int(len(raw_density[(raw_density.analysis_group == chicken_GROUPS[2]) & (raw_density.effective_callable_bp == 0)])), controls=chicken_N_CONTROLS, control_mean=control_mean, control_pooled=control_pooled, variant_qc=dict(variant_qc), control_variant_qc=dict(control_qc), frozen_six_species_exact=True, input_manifest={str(p.relative_to(chicken_ROOT)): dict(size=p.stat().st_size, sha256=chicken_sha(p)) for p in (chicken_VCF, chicken_CALL, chicken_FA, chicken_GTF, chicken_CLASS / 'all_ensembl_snRNA_82.tsv')}, variant_source_limitation='sites-only VCF: no per-sample genotype/GQ recheck; no direct per-base AN mask')
    (chicken_OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

def chicken_entrypoint():
    chicken_main()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    _run_stages({'chicken': 'chicken_entrypoint'}, globals(), setup=_prepare_runtime)
