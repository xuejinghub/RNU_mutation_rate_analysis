"""Consolidated final processing: analyze_baboon_updated.

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
from processing.fig1.chicken.scripts import resample_baboon_controls_random as _baboon_controls
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import argparse
import bisect
import hashlib
import json
import os
import subprocess
import numpy as np
import pandas as pd
import pysam
from scipy.stats import mannwhitneyu
from collections import defaultdict
import gzip
import importlib.util
import io
import shutil
sites_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/primate/scripts/analyze_fig1_baboon_af0p02.py').parents[1]
sites_TMP = sites_ROOT / 'tmp/fig1_baboon_af0p02_complement_20260921'
sites_OUT = sites_ROOT / 'data/processed/fig1_baboon_af0p02_complement_20260921'
sites_PREVIOUS = sites_ROOT / 'data/processed/fig1_baboon_20260921'
sites_INTEGRITY_TMP = sites_ROOT / 'tmp/fig1_baboon_20260921'
sites_BASE = sites_ROOT / 'transpecies_compare/data/processed/fig1_rat_no_callable_20260921'
sites_ANNOT = sites_ROOT / 'results/baboon_snrna_gene_sets_20260921'
sites_RAW = sites_ROOT / 'data/raw/baboon_Panubis1.0'
sites_FA = sites_ROOT / 'tmp/baboon_snrna_20260921/Panubis1.0.fa'
sites_GROUPS = ['pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes', 'genome_wide_neutral']
sites_SEED_POOL = 20260923
sites_SEED_SELECT = 20260924

def sites_chrommap():
    result = {}
    for line in (sites_RAW / 'assembly_report.txt').read_text().splitlines():
        if line.startswith('#'):
            continue
        a = line.split('\t')
        if a[1] == 'assembled-molecule' and a[2] in {str(i) for i in range(1, 21)}:
            assert a[5] == '='
            result[a[2]] = a[6]
    assert len(result) == 20
    return result

def sites_merge(intervals):
    result = []
    for (s, e) in sorted(intervals):
        if result and s <= result[-1][1]:
            result[-1] = (result[-1][0], max(e, result[-1][1]))
        else:
            result.append((s, e))
    return result

def sites_overlaps(intervals, s, e):
    i = bisect.bisect_left(intervals, (e,))
    return i > 0 and intervals[i - 1][1] > s

def sites_source_for(i):
    source = sites_ROOT / f'baboon/chr{i}_SNP.vcf.gz'
    if not source.exists():
        source = sites_ROOT / f'data/raw/baboon_variants/chr{i}_SNP.vcf.gz'
    idx = sites_ROOT / f'baboon/chr{i}_SNP.vcf.gz.tbi'
    if not idx.exists():
        idx = sites_ROOT / f'data/raw/baboon_variants/chr{i}_SNP.vcf.gz.tbi'
    if not idx.exists():
        idx = sites_TMP / f'chr{i}_SNP.vcf.gz.tbi'
    return (source, idx)

def sites_prepare():
    import shutil
    cmap = sites_chrommap()
    fa = pysam.FastaFile(str(sites_FA))
    lengths = {str(i): fa.get_reference_length(cmap[str(i)]) for i in range(1, 21)}
    universe = pd.read_csv(sites_ANNOT / 'all_annotated_snRNA.autosomal.tsv', sep='\t', keep_default_na=False)
    pol = pd.read_csv(sites_ANNOT / 'PolII_one_to_one_candidates.autosomal.tsv', sep='\t', keep_default_na=False)
    u6 = pd.read_csv(sites_ANNOT / 'U6_U6ATAC_one_to_one_candidates.autosomal.tsv', sep='\t', keep_default_na=False)
    polids = set(pol.baboon_gene_id)
    u6ids = set(u6.baboon_gene_id)
    allids = set(universe.baboon_gene_id)
    assert len(universe) == len(allids) == 1681 and len(polids) == 23 and (len(u6ids) == 88)
    assert polids.isdisjoint(u6ids) and polids | u6ids <= allids
    pseudoids = allids - polids - u6ids
    assert len(pseudoids) == 1570
    genes = []
    for r in universe.itertuples():
        assert r.gene_biotype == 'snRNA'
        group = sites_GROUPS[0] if r.baboon_gene_id in polids else sites_GROUPS[1] if r.baboon_gene_id in u6ids else sites_GROUPS[2]
        genes.append(dict(chrom=str(r.ensembl_chrom), refseq_chrom=r.refseq_chrom, start=int(r.start0), end=int(r.end), strand=r.strand, unit_id=r.baboon_gene_id, display_name=r.human_gene_name or r.gene_name or r.baboon_gene_id, human_gene_id=r.human_gene_id, family=r.human_family or r.annotation_family, annotation_family=r.annotation_family, analysis_group=group, gene_biotype='snRNA', assignment_method='user_defined_annotated_snRNA_complement' if group == sites_GROUPS[2] else r.synteny_assignment_method))
    genes = pd.DataFrame(genes)
    assert genes.groupby('analysis_group').size().to_dict() == dict(zip(sites_GROUPS[:3], [23, 88, 1570]))
    controls = pd.read_csv(sites_PREVIOUS / 'baboon_control_1000.tsv', sep='\t', dtype={'chrom': str}, float_precision='round_trip')
    controls.to_csv(sites_OUT / 'baboon_control_1000.tsv', sep='\t', index=False)
    shutil.copyfile(sites_PREVIOUS / 'baboon_control_10000_candidates.tsv', sites_OUT / 'baboon_control_10000_candidates.tsv')
    catalog = pd.concat([genes, controls], ignore_index=True)
    catalog.to_csv(sites_OUT / 'baboon_catalog.tsv', sep='\t', index=False)
    genes[genes.analysis_group.eq(sites_GROUPS[2])].to_csv(sites_OUT / 'baboon_operational_pseudogenes_1570.tsv', sep='\t', index=False)
    genes.to_csv(sites_OUT / 'baboon_snRNA_gene_membership_1681.tsv', sep='\t', index=False)
    genes.groupby(['analysis_group', 'annotation_family']).size().rename('n_genes').reset_index().to_csv(sites_OUT / 'baboon_group_family_counts.tsv', sep='\t', index=False)
    for group in sites_GROUPS:
        with (sites_OUT / (group + '.Panubis1.0.autosomal.bed')).open('w') as f:
            for r in catalog[catalog.analysis_group.eq(group)].itertuples():
                f.write(f'{r.chrom}\t{r.start}\t{r.end}\t{r.unit_id}\t0\t{r.strand}\n')
    regions = defaultdict(list)
    for r in catalog.itertuples():
        flank = 0 if r.analysis_group == sites_GROUPS[-1] else 2000
        assert 0 <= r.start - flank < r.end + flank <= lengths[r.chrom]
        regions[r.chrom].append((int(r.start) - flank, int(r.end) + flank))
    for c in lengths:
        with (sites_TMP / f'chr{c}.targets.bed').open('w') as f:
            for (start, end) in sites_merge(regions[c]):
                f.write(f'{c}\t{start}\t{end}\n')
    fa.close()
    (sites_OUT / 'analysis_contract.json').write_text(json.dumps(dict(assembly='Panubis1.0', annotation='Ensembl115', autosomes=list(lengths), groups=sites_GROUPS, group_sizes=[23, 88, 1570, 1000], n_samples=110, genotype_GQ_min=20, AF_threshold=0.02, callable_mask=False, minimum_length_filter=False, control_pool_seed=sites_SEED_POOL, control_select_seed=sites_SEED_SELECT, control_distance_from_all_genes=10000, control_length=150, controls_reused_from=str(sites_PREVIOUS / 'baboon_control_1000.tsv'), pseudogene_group='user_defined_all_annotated_autosomal_snRNA_minus_PolII_and_U6_syntenic_sets', pseudogene_definition_is_operational=True, U7_included_in_complement=True, annotation_universe=1681), indent=2) + '\n')
    print('Prepared', catalog.groupby('analysis_group').size().to_dict(), flush=True)

def sites_extract_chrom(i):
    (source, idx) = sites_source_for(i)
    chrom = str(i)
    if not source.exists():
        return dict(chrom=chrom, status='MISSING')
    before = source.stat()
    sig = dict(source=str(source.resolve()), bytes=before.st_size, mtime_ns=before.st_mtime_ns, target_sha256=hashlib.sha256((sites_TMP / f'chr{i}.targets.bed').read_bytes()).hexdigest())
    marker = sites_TMP / f'chr{i}.extract_complete.json'
    dest = sites_TMP / f'chr{i}.GQ20.snvs.vcf.gz'
    if marker.exists() and dest.exists() and (json.loads(marker.read_text())['signature'] == sig):
        return json.loads(marker.read_text())
    assert idx.exists(), idx
    rawfile = sites_TMP / f'chr{i}.targets.raw.bcf'
    records = 0
    with pysam.VariantFile(str(source), index_filename=str(idx)) as vf:
        assert len(vf.header.samples) == 110
        assert 'GQ' in vf.header.formats
        added_info = []
        for tag in ['CLASS', 'MA', 'MAF', 'rsID']:
            if tag not in vf.header.info:
                vf.header.info.add(tag, '.', 'String', 'Undeclared GVM source field preserved as uninterpreted text; not used in this analysis')
                added_info.append(tag)
        with pysam.VariantFile(str(rawfile), 'wb', header=vf.header) as out:
            for line in (sites_TMP / f'chr{i}.targets.bed').read_text().splitlines():
                (c, s, e) = line.split('\t')
                (s, e) = (int(s), int(e))
                for rec in vf.fetch(c, s, e):
                    if s <= rec.start < e:
                        out.write(rec)
                        records += 1
    partial = sites_TMP / f'chr{i}.partial.vcf.gz'
    commands = [['bcftools', '+setGT', str(rawfile), '-Ou', '--', '-t', 'q', '-n', '.', '-i', 'FMT/GQ<20'], ['bcftools', '+fill-tags', '-Ou', '--', '-t', 'AC,AN,AF'], ['bcftools', 'norm', '-m', '-any', '-Ou'], ['bcftools', 'view', '-G', '-v', 'snps', '-i', 'INFO/AC>0', '-Oz', '-o', str(partial)]]
    procs = []
    previous = None
    with (sites_TMP / f'chr{i}.extract.log').open('w') as log:
        log.write(json.dumps(commands) + '\n')
        log.flush()
        for (j, cmd) in enumerate(commands):
            p = subprocess.Popen(cmd, stdin=previous.stdout if previous else None, stdout=subprocess.PIPE if j < len(commands) - 1 else log, stderr=log)
            if previous:
                previous.stdout.close()
            procs.append(p)
            previous = p
        codes = [p.wait() for p in procs]
        if any(codes):
            raise RuntimeError((chrom, codes))
    os.replace(partial, dest)
    subprocess.run(['bcftools', 'index', '-f', str(dest)], check=True)
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    record = dict(chrom=chrom, status='PASS', signature=sig, index=str(idx), raw_target_records=records, derived_header_INFO_declarations_added=added_info)
    marker.write_text(json.dumps(record, indent=2) + '\n')
    print('Extracted chr' + chrom, records, flush=True)
    return record

def sites_extract():
    with ThreadPoolExecutor(max_workers=4) as ex:
        records = list(ex.map(sites_extract_chrom, range(1, 21)))
    (sites_OUT / 'extraction_manifest.json').write_text(json.dumps(records, indent=2) + '\n')

def sites_classify_positions(chrom, af_max):
    positions = defaultdict(dict)
    qc = Counter()
    iupac = set('ACGT')
    with pysam.VariantFile(str(sites_TMP / f'chr{chrom}.GQ20.snvs.vcf.gz')) as vf:
        for rec in vf:
            qc['retained_snv_records'] += 1
            for (i, alt) in enumerate(rec.alts or []):
                ac = rec.info['AC'][i]
                an = rec.info.get('AN', 0)
                af = ac / an if an and 0 <= ac <= an else None
                reasons = []
                if tuple(rec.filter.keys()) != ('PASS',):
                    reasons.append('non_PASS')
                if not (len(rec.ref) == len(alt) == 1 and rec.ref in iupac and (alt in iupac) and (rec.ref != alt)):
                    reasons.append('noncanonical_SNV')
                if af is None:
                    reasons.append('invalid_AC_AN')
                elif af > af_max:
                    reasons.append('AF_above_threshold')
                key = (rec.ref, alt)
                value = (ac, an, af, reasons)
                if key in positions[rec.start]:
                    assert positions[rec.start][key] == value, 'Discordant duplicate variant'
                    qc['identical_duplicate_records'] += 1
                positions[rec.start][key] = value
    rare = []
    counts = []
    masked = []
    audit = []
    for (pos, alts) in sorted(positions.items()):
        reasons = {reason for a in alts.values() for reason in a[3]}
        if reasons:
            masked.append(pos)
        else:
            rare.append(pos)
            counts.append(len(alts))
        for ((ref, alt), a) in alts.items():
            audit.append((chrom, pos + 1, ref, alt, *a[:3], ';'.join(sorted(reasons)), not bool(reasons)))
        qc.update(reasons)
    pd.DataFrame(audit, columns=['chrom', 'pos1', 'ref', 'alt', 'AC', 'AN', 'AF', 'position_mask_reason', 'counted']).to_csv(sites_OUT / f'chr{chrom}.classified_target_alleles.tsv.gz', sep='\t', index=False)
    return (np.array(rare, dtype=int), np.r_[0, np.cumsum(counts)], np.array(masked, dtype=int), dict(qc))

def sites_interval_counts(s, e, seq, positions):
    (rare, prefix, mask, _) = positions
    (left, right) = np.searchsorted(rare, [s, e])
    (ml, mr) = np.searchsorted(mask, [s, e])
    non = sum((x not in 'ACGT' for x in seq))
    bad = sum((seq[int(pos) - s] in 'ACGT' for pos in mask[ml:mr]))
    assert all((seq[int(pos) - s] in 'ACGT' for pos in rare[left:right]))
    effective = e - s - non - bad
    alt = int(prefix[right] - prefix[left])
    return dict(region_bp=e - s, non_acgt_bp=non, mask_bp=bad, effective_unmasked_bp=effective, rare_snv_alt_count=alt, rare_snv_position_count=int(right - left), rare_snv_density_per_kb=1000 * alt / effective if effective else np.nan)

def sites_count(af_max):
    manifest = json.loads((sites_OUT / 'extraction_manifest.json').read_text())
    assert len(manifest) == 20 and all((r['status'] == 'PASS' for r in manifest)), 'All 20 chromosomes required'
    cat = pd.read_csv(sites_OUT / 'baboon_catalog.tsv', sep='\t', dtype={'chrom': str})
    positions = {str(i): sites_classify_positions(str(i), af_max) for i in range(1, 21)}
    fa = pysam.FastaFile(str(sites_FA))
    density = []
    bins = []
    for r in cat.itertuples():
        (s, e) = (int(r.start), int(r.end))
        seq = fa.fetch(r.refseq_chrom, s, e).upper()
        vals = sites_interval_counts(s, e, seq, positions[r.chrom])
        assert vals['effective_unmasked_bp'] > 0
        density.append(dict(species='Baboon', assembly='Panubis1.0', analysis_group=r.analysis_group, unit_id=r.unit_id, display_name=r.display_name, chrom=r.chrom, start=s, end=e, strand=r.strand, **vals, callable_bp=np.nan, effective_callable_bp=np.nan, af_threshold=af_max, denominator_basis='reference_ACGT_minus_position_masks; no callable filter', source_note='110 samples; GQ20; PolII/U6 syntenic sets and user-defined annotated-snRNA complement; no callable mask'))
        if r.analysis_group == sites_GROUPS[-1]:
            continue
        edges = np.rint(np.linspace(s, e, 21)).astype(int)
        assert (np.diff(edges) > 0).all()
        intervals = []
        for k in range(40):
            (a, b) = (s - 2000 + 50 * k, s - 1950 + 50 * k) if r.strand == '+' else (e + 1950 - 50 * k, e + 2000 - 50 * k)
            intervals.append((a, b, 'upstream', -2000 + 50 * k, -1950 + 50 * k))
        for k in range(20):
            j = k if r.strand == '+' else 19 - k
            intervals.append((int(edges[j]), int(edges[j + 1]), 'body', k * 50, (k + 1) * 50))
        for k in range(40):
            (a, b) = (e + 50 * k, e + 50 * (k + 1)) if r.strand == '+' else (s - 50 * (k + 1), s - 50 * k)
            intervals.append((a, b, 'downstream', 1000 + 50 * k, 1050 + 50 * k))
        window = fa.fetch(r.refseq_chrom, s - 2000, e + 2000).upper()
        for (k, (a, b, region, rs, re_)) in enumerate(intervals):
            values = sites_interval_counts(a, b, window[a - s + 2000:b - s + 2000], positions[r.chrom])
            bins.append(dict(species='Baboon', analysis_group=r.analysis_group, unit_id=r.unit_id, bin_index=k, region=region, relative_start=rs, relative_end=re_, chrom=r.chrom, genomic_start=a, genomic_end=b, **values))
    fa.close()
    d = pd.DataFrame(density)
    gb = pd.DataFrame(bins)
    body = gb[gb.region.eq('body')].groupby('unit_id')[['rare_snv_alt_count', 'effective_unmasked_bp']].sum()
    for r in d[d.analysis_group.ne(sites_GROUPS[-1])].itertuples():
        assert r.rare_snv_alt_count == body.loc[r.unit_id, 'rare_snv_alt_count'] and r.effective_unmasked_bp == body.loc[r.unit_id, 'effective_unmasked_bp']
    d.to_csv(sites_OUT / 'baboon_gene_locus_density.tsv', sep='\t', index=False)
    gb.to_csv(sites_OUT / 'baboon_profile_gene_bins.tsv.gz', sep='\t', index=False)
    prof = gb.groupby(['species', 'analysis_group', 'bin_index', 'region', 'relative_start', 'relative_end'], as_index=False).agg(gene_count=('unit_id', 'size'), contributing_genes=('effective_unmasked_bp', lambda v: (v > 0).sum()), total_rare_snv_alt_count=('rare_snv_alt_count', 'sum'), total_effective_unmasked_bp=('effective_unmasked_bp', 'sum'))
    prof['pooled_density_per_kb'] = 1000 * prof.total_rare_snv_alt_count / prof.total_effective_unmasked_bp
    prof['total_effective_callable_bp'] = np.nan
    control = d[d.analysis_group.eq(sites_GROUPS[-1])]
    mean = control.rare_snv_density_per_kb.mean()
    pooled = 1000 * control.rare_snv_alt_count.sum() / control.effective_unmasked_bp.sum()
    assert mean > 0 and pooled > 0
    prof['density_ratio_to_control'] = prof.pooled_density_per_kb / pooled
    bars = []
    stats = []
    for g in sites_GROUPS:
        x = d[d.analysis_group.eq(g)].rare_snv_density_per_kb
        bars.append(dict(species='Baboon', analysis_group=g, n_units=len(x), mean_density_per_kb=x.mean(), mean_density_ratio_to_control=x.mean() / mean))
    for (g1, g2) in combinations(sites_GROUPS, 2):
        x = d[d.analysis_group.eq(g1)].rare_snv_density_per_kb
        y = d[d.analysis_group.eq(g2)].rare_snv_density_per_kb
        t = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic', use_continuity=True)
        stats.append(dict(species='Baboon', group_1=g1, group_2=g2, n1=len(x), n2=len(y), U=t.statistic, p_value=t.pvalue, mean_density_fold=x.mean() / y.mean() if y.mean() else np.nan))
    newtables = {'panel_a_gene_locus_density.tsv': d, 'panel_a_pairwise_statistics.tsv': pd.DataFrame(stats), 'panel_b_control_mean_ratios.tsv': pd.DataFrame(bars), 'panel_c_profile_control_ratios.tsv': prof, 'control_denominators.tsv': pd.DataFrame([dict(species='Baboon', control_mean_density_per_kb=mean, control_pooled_density_per_kb=pooled, denominator_column='effective_unmasked_bp')])}
    for (name, new) in newtables.items():
        old = pd.read_csv(sites_BASE / name, sep='\t', float_precision='round_trip')
        joined = pd.concat([old, new], ignore_index=True)
        joined.to_csv(sites_OUT / name, sep='\t', index=False)
        pd.testing.assert_frame_equal(joined[joined.species.ne('Baboon')][old.columns].reset_index(drop=True), old, check_dtype=False, check_exact=True)
    summary = d.groupby('analysis_group').agg(n=('unit_id', 'size'), count=('rare_snv_alt_count', 'sum'), effective_bp=('effective_unmasked_bp', 'sum'), mean_density=('rare_snv_density_per_kb', 'mean'), zero_count=('rare_snv_alt_count', lambda x: (x == 0).sum()))
    summary.to_csv(sites_OUT / 'baboon_group_summary.tsv', sep='\t')
    (sites_OUT / 'baboon_variant_QC.json').write_text(json.dumps({c: p[-1] for (c, p) in positions.items()}, indent=2) + '\n')
    contract = json.loads((sites_OUT / 'analysis_contract.json').read_text())
    contract['AF_threshold'] = af_max
    (sites_OUT / 'analysis_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(summary.to_string(), flush=True)

def sites_main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'extract', 'count', 'all'])
    p.add_argument('--af', type=float, default=0.02)
    args = p.parse_args()
    sites_TMP.mkdir(parents=True, exist_ok=True)
    sites_OUT.mkdir(parents=True, exist_ok=True)
    actions = {'prepare': sites_prepare, 'extract': sites_extract,
               'count': lambda: sites_count(args.af)}
    for stage in ['prepare', 'extract', 'count'] if args.stage == 'all' else [args.stage]:
        actions[stage]()

def sites_entrypoint():
    sites_main()
callable_util = _component('update_rat_callable_fig1')
callable_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/update_baboon_callable_fig1.py').parents[1]
callable_BASE = callable_ROOT / 'data/processed/fig1_rat_control1000_20260927'
callable_OUT = callable_ROOT / 'data/processed/fig1_baboon_callable_20260927'
callable_RESULT = callable_ROOT / 'results/baboon_callable_Panubis1_20260927'
callable_TMP = callable_ROOT / 'tmp/baboon_callable_fig1_20260927'
callable_OLD = callable_ROOT / 'primate/data/processed/fig1_baboon_af0p02_complement_20260921'
callable_CORE = _component_file('analyze_fig1_baboon_af0p02')
callable_FA = callable_ROOT / 'primate/tmp/baboon_snrna_20260921/Panubis1.0.fa'
callable_GTF = callable_ROOT / 'primate/data/raw/baboon_Panubis1.0/Papio_anubis.Panubis1.0.115.gtf.gz'
callable_POOL = callable_ROOT / 'primate/data/processed/fig1_baboon_20260921/baboon_control_10000_candidates.tsv'
callable_COV = Path('/public5/home/xuejing/primates/baboon/coverage/bam/baboon.Panubis1.0.coverage.whitelist.bed')
callable_MAP = Path('/public5/home/xuejing/primates/baboon/mappability/mappablility/Panubis1.0.k150e1.mappability1.whitelist.bed')
callable_METRICS = Path(str(callable_COV) + '.metrics.json')
callable_CALL = callable_RESULT / 'baboon.Panubis1.0.coverage_intersect_mappability.callable.bed'
callable_GROUPS = callable_util.GROUPS
callable_NAMES = callable_util.NAMES
callable_SEED = None
callable_read = _partial(_shared.read_table, dtype={'chrom': str})

def callable_core_module():
    core = _component('analyze_fig1_baboon_af0p02')
    records = json.loads((callable_OLD / 'extraction_manifest.json').read_text())
    paths = {int(r['chrom']): (Path(r['signature']['source']), Path(r['index'])) for r in records}
    for r in records:
        (path, index) = paths[int(r['chrom'])]
        assert path.is_file() and index.is_file()
    core.source_for = lambda i: paths[i]
    (core.TMP, core.OUT) = (callable_TMP, callable_OUT)
    return core

def callable_setup():
    for d in [callable_OUT, callable_RESULT, callable_TMP]:
        assert not d.is_symlink()
        d.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(callable_TMP)
    os.environ['LC_ALL'] = 'C'
    callable_util.TMP = callable_TMP

def callable_prepare():
    global callable_SEED
    core = callable_core_module()
    cmap = core.chrommap()
    assert Path(str(callable_FA) + '.fai').exists()
    with pysam.FastaFile(str(callable_FA)) as fa:
        sizes = dict(zip(fa.references, fa.lengths))
    metrics = json.loads(callable_METRICS.read_text())
    assert sum(sizes.values()) == metrics['genome_bp']
    (cov, cov_stats) = callable_util.normalize_bed(callable_COV, 'coverage', sizes)
    (mapp, map_stats) = callable_util.normalize_bed(callable_MAP, 'mappability', sizes)
    overlap = callable_TMP / 'coverage_mappability.intersection.bed'
    callable_util.command(['bedtools', 'intersect', '-sorted', '-a', str(cov), '-b', str(mapp)], overlap)
    callable_util.command(['bedtools', 'merge', '-i', str(overlap)], callable_CALL)
    index = callable_util.index_bed(callable_CALL)
    assert set(cmap.values()) <= set(index)
    pd.DataFrame([dict(ensembl_chrom=c, refseq_chrom=r, length=sizes[r]) for (c, r) in cmap.items()]).to_csv(callable_RESULT / 'Panubis1.0.chromosome_aliases.tsv', sep='\t', index=False)
    forbidden = defaultdict(list)
    with gzip.open(callable_GTF, 'rt') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            f = line.split('\t')
            if len(f) == 9 and f[2] == 'gene' and (f[0] in cmap):
                forbidden[f[0]].append((max(0, int(f[3]) - 1 - 10000), min(sizes[cmap[f[0]]], int(f[4]) + 10000)))
    forbidden = {c: core.merge(v) for (c, v) in forbidden.items()}
    with (callable_TMP / 'gene_buffers_10kb.refseq.bed').open('w') as handle:
        for c in sorted(cmap, key=lambda c: cmap[c]):
            for (s, e) in forbidden[c]:
                handle.write(f'{cmap[c]}\t{s}\t{e}\n')
    pool = callable_read(callable_POOL)
    assert len(pool) == 10000 and pool.unit_id.is_unique
    rows = []
    with pysam.FastaFile(str(callable_FA)) as fa:
        for r in pool.itertuples(index=False):
            assert r.refseq_chrom == cmap[r.chrom] and r.end - r.start == 150
            seq = fa.fetch(r.refseq_chrom, r.start, r.end).upper()
            cbp = sum((e - s for (s, e) in callable_util.segments(index, r.refseq_chrom, r.start, r.end)))
            acgt = not set(seq) - set('ACGT')
            far = not core.overlaps(forbidden[r.chrom], r.start, r.end)
            rows.append(dict(**r._asdict(), raw_callable_bp=cbp, ACGT_only=acgt, outside_gene_buffer10kb=far, fully_callable=cbp == 150 and acgt and far))
    candidates = pd.DataFrame(rows)
    eligible = candidates[candidates.fully_callable]
    assert len(eligible) >= 1000
    draw_plan = _baboon_controls.seed_plan(callable_OUT / 'seed_plan.json')
    callable_SEED = int(draw_plan['seed'])
    candidates['selection_rank'] = np.nan
    candidates.to_csv(callable_OUT / 'baboon_control_candidate_prefilter.tsv', sep='\t', index=False)
    baseline = callable_read(callable_BASE / 'panel_a_gene_locus_density.tsv')
    baboon = baseline[baseline.species.eq('Baboon')]
    assert baboon.groupby('analysis_group').size().to_dict() == dict(zip(callable_GROUPS, [23, 88, 1570, 1000]))
    original = callable_read(callable_OLD / 'baboon_gene_locus_density.tsv')
    cols = ['unit_id', 'chrom', 'start', 'end', 'analysis_group', 'rare_snv_alt_count', 'effective_unmasked_bp', 'rare_snv_density_per_kb']
    pd.testing.assert_frame_equal(baboon[cols].sort_values('unit_id').reset_index(drop=True), original[cols].sort_values('unit_id').reset_index(drop=True), check_dtype=False, check_exact=True)
    targets = defaultdict(list)
    for r in baboon.itertuples():
        flank = 0 if r.analysis_group == callable_GROUPS[-1] else 2000
        targets[r.chrom].append((int(r.start) - flank, int(r.end) + flank))
    for r in eligible.itertuples():
        targets[r.chrom].append((r.start, r.end))
    for c in cmap:
        with (callable_TMP / f'chr{c}.targets.bed').open('w') as handle:
            for (s, e) in core.merge(targets[c]):
                assert 0 <= s < e <= sizes[cmap[c]]
                handle.write(f'{c}\t{s}\t{e}\n')
    bp = {c: int((v[:, 1] - v[:, 0]).sum()) for (c, v) in index.items()}
    plan = dict(assembly='Panubis1.0', callable_definition='coverage INTERSECTION mappability; never union', coordinates='0-based half-open BED', chromosome_mapping=cmap, coverage=callable_util.record(callable_COV), mappability=callable_util.record(callable_MAP), coverage_metrics=metrics, normalized_coverage=cov_stats, normalized_mappability=map_stats, callable_bed=callable_util.record(callable_CALL), callable_bp=sum(bp.values()), autosomal_callable_bp=sum((bp[c] for c in cmap.values())), callable_bp_by_contig=bp, candidate_pool=callable_util.record(callable_POOL), annotation=callable_util.record(callable_GTF), assembly_report=callable_util.record(core.RAW / 'assembly_report.txt'), fasta=callable_util.record(callable_FA, False), fasta_index=callable_util.record(str(callable_FA) + '.fai'), minimum_effective_callable_bp=50, control_length_bp=150, control_gene_buffer_bp=10000, control_seed=str(callable_SEED), fully_callable_candidates=len(eligible), control_policy='Single random draw from eligible candidates; seed from system entropy; no outcome-based selection', original_groups=dict(zip(callable_GROUPS, [23, 88, 1570, 1000])), baseline=str(callable_BASE), baseline_inputs={n: callable_util.record(callable_BASE / f'{n}.tsv') for n in callable_NAMES}, cohort_individuals=110, GQ_min=20, AF_max=0.02, allele_counting='Original ALT-only and position-level masks retained', biological_reclassification=False, U7_membership='Retained in the original operational pseudogene complement')
    (callable_OUT / 'analysis_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({k: plan[k] for k in ['callable_definition', 'callable_bp', 'autosomal_callable_bp', 'fully_callable_candidates']}, indent=2), flush=True)

def callable_extract(workers):
    core = callable_core_module()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(core.extract_chrom, range(1, 21)))
    assert len(records) == 20 and all((r['status'] == 'PASS' for r in records))
    (callable_OUT / 'extraction_manifest.json').write_text(json.dumps(records, indent=2) + '\n')

def callable_combine_exact(name, frame):
    old = callable_read(callable_BASE / f'{name}.tsv')
    frame = frame.reindex(columns=old.columns)
    buffer = io.StringIO()
    frame.to_csv(buffer, sep='\t', index=False, header=False)
    inserted = False
    with (callable_BASE / f'{name}.tsv').open() as source, (callable_OUT / f'{name}.tsv').open('w') as target:
        target.write(next(source))
        for line in source:
            if line.startswith('Baboon\t'):
                if not inserted:
                    target.write(buffer.getvalue())
                    inserted = True
            else:
                target.write(line)
    assert inserted
    new = callable_read(callable_OUT / f'{name}.tsv')
    pd.testing.assert_frame_equal(old[old.species.ne('Baboon')].reset_index(drop=True), new[new.species.ne('Baboon')].reset_index(drop=True), check_dtype=False, check_exact=True)

def callable_analyze():
    global callable_SEED
    core = callable_core_module()
    cmap = core.chrommap()
    reverse = {v: k for (k, v) in cmap.items()}
    plan = json.loads((callable_OUT / 'analysis_plan.json').read_text())
    draw_plan = _baboon_controls.seed_plan(callable_OUT / 'seed_plan.json')
    callable_SEED = int(draw_plan['seed'])
    plan.update(control_seed=str(callable_SEED), control_policy='Single random draw; no outcome-based selection')
    index = callable_util.index_bed(callable_CALL)
    manifest = json.loads((callable_OUT / 'extraction_manifest.json').read_text())
    assert all((r['status'] == 'PASS' for r in manifest))
    positions = {c: core.classify_positions(c, 0.02) for c in cmap}

    class Adapter:

        @staticmethod
        def interval_counts(chrom, s, e, seq, pos):
            return core.interval_counts(s, e, seq, pos[reverse[chrom]])

    def counts(chrom, s, e, seq):
        return callable_util.callable_counts(Adapter, index, positions, cmap[chrom], s, e, seq)
    old = callable_read(callable_BASE / 'panel_a_gene_locus_density.tsv')
    old = old[old.species.eq('Baboon')]
    genes = []
    audit = []
    windows = {}
    with pysam.FastaFile(str(callable_FA)) as fa:
        for r in old.itertuples(index=False):
            (s, e) = (int(r.start), int(r.end))
            seq = fa.fetch(cmap[r.chrom], s, e).upper()
            baseline = core.interval_counts(s, e, seq, positions[r.chrom])
            for key in ['rare_snv_alt_count', 'rare_snv_position_count', 'mask_bp', 'effective_unmasked_bp', 'rare_snv_density_per_kb']:
                assert baseline[key] == getattr(r, key), (r.unit_id, key)
            if r.analysis_group == callable_GROUPS[-1]:
                continue
            v = counts(r.chrom, s, e, seq)
            status = 'retained' if v['effective_callable_bp'] >= 50 else 'no_callable_overlap' if v['raw_callable_bp'] == 0 else 'no_effective_callable_bases' if v['effective_callable_bp'] == 0 else 'effective_callable_lt50'
            audit.append(dict(unit_id=r.unit_id, analysis_group=r.analysis_group, chrom=r.chrom, refseq_chrom=cmap[r.chrom], start=s, end=e, strand=r.strand, status=status, previous_rare_snv_alt_count=r.rare_snv_alt_count, previous_effective_bp=r.effective_unmasked_bp, previous_density_per_kb=r.rare_snv_density_per_kb, **v))
            if status == 'retained':
                record = r._asdict()
                record.update(v, refseq_chrom=cmap[r.chrom], denominator_basis='coverage_intersect_mappability_ACGT_minus_position_masks', source_note='110 baboons; GQ20; AF<=0.02; coverage intersect k150e1 mappability=1; effective callable bp>=50')
                genes.append(record)
                windows[r.unit_id] = (s - 2000, fa.fetch(cmap[r.chrom], s - 2000, e + 2000).upper())
        pre = callable_read(callable_OUT / 'baboon_control_candidate_prefilter.tsv')
        eligible = pre[pre.fully_callable]
        controls = []
        for r in eligible.itertuples(index=False):
            v = counts(r.chrom, r.start, r.end, fa.fetch(r.refseq_chrom, r.start, r.end).upper())
            assert v['raw_callable_bp'] == 150 and v['callable_bp'] == 150
            controls.append(dict(**{k: v for (k, v) in r._asdict().items() if k != 'raw_callable_bp'}, **v))
    genes = pd.DataFrame(genes)
    audit = pd.DataFrame(audit)
    controls = pd.DataFrame(controls)
    selected = _baboon_controls.select_controls(controls, callable_SEED)
    ranks = dict(zip(selected.unit_id, selected.selection_rank))
    controls['selection_rank'] = controls.unit_id.map(ranks)
    pre['selection_rank'] = pre.unit_id.map(ranks)
    selected[['unit_id', 'selection_rank']].rename(columns={'selection_rank': 'draw_rank'}).to_csv(callable_OUT / 'selected_ids.tsv', sep='\t', index=False)
    selected = selected.sort_values(['refseq_chrom', 'start']).reset_index(drop=True)
    selected['species'] = 'Baboon'
    selected['assembly'] = 'Panubis1.0'
    selected['af_threshold'] = 0.02
    selected['denominator_basis'] = 'coverage_intersect_mappability_ACGT_minus_position_masks'
    selected['source_note'] = f'110 baboons; GQ20; AF<=0.02; single random draw of 1000 callable controls; seed={callable_SEED}; no outcome-based selection'
    selected.to_csv(callable_OUT / 'baboon_control_1000.tsv', sep='\t', index=False)
    selected[['refseq_chrom', 'start', 'end', 'unit_id']].to_csv(callable_RESULT / 'baboon.Panubis1.0.fully_callable_control1000.bed', sep='\t', index=False, header=False)
    extra = [c for c in controls if c not in pre or c == 'unit_id']
    ca = pre.merge(controls[extra], on='unit_id', how='left', validate='one_to_one')
    ca['status'] = 'not_fully_callable_or_intergenic'
    ca.loc[ca.fully_callable, 'status'] = 'not_selected'
    ca.loc[ca.fully_callable & ca.effective_callable_bp.lt(50), 'status'] = 'effective_callable_lt50'
    ca.loc[ca.unit_id.isin(selected.unit_id), 'status'] = 'selected_by_single_random_draw'
    ca.to_csv(callable_OUT / 'baboon_control_candidate_audit.tsv', sep='\t', index=False)
    audit.to_csv(callable_OUT / 'baboon_callable_filter_audit.tsv', sep='\t', index=False)
    audit[audit.status.ne('retained')].to_csv(callable_OUT / 'baboon_callable_excluded.tsv', sep='\t', index=False)
    d = pd.concat([genes, selected.reindex(columns=genes.columns)], ignore_index=True)
    d.to_csv(callable_OUT / 'baboon_gene_locus_density.tsv', sep='\t', index=False)
    for g in callable_GROUPS:
        f = d[d.analysis_group.eq(g)].copy()
        f['score'] = 0
        f[['refseq_chrom', 'start', 'end', 'unit_id', 'score', 'strand']].astype({'start': int, 'end': int}).to_csv(callable_RESULT / f'{g}.Panubis1.0.callable.bed', sep='\t', index=False, header=False)
    oldbins = callable_read(callable_OLD / 'baboon_profile_gene_bins.tsv.gz')
    oldbins = oldbins[oldbins.unit_id.isin(genes.unit_id)]
    rows = []
    for r in oldbins.itertuples(index=False):
        (start, seq) = windows[r.unit_id]
        v = counts(r.chrom, r.genomic_start, r.genomic_end, seq[r.genomic_start - start:r.genomic_end - start])
        row = r._asdict()
        row.update(v)
        rows.append(row)
    gb = pd.DataFrame(rows)
    gb.to_csv(callable_OUT / 'baboon_profile_gene_bins.tsv.gz', sep='\t', index=False)
    body = gb[gb.region.eq('body')].groupby('unit_id')[['rare_snv_alt_count', 'effective_callable_bp']].sum()
    for r in genes.itertuples():
        assert body.loc[r.unit_id, 'rare_snv_alt_count'] == r.rare_snv_alt_count
        assert body.loc[r.unit_id, 'effective_callable_bp'] == r.effective_callable_bp
    cm = selected.rare_snv_density_per_kb.mean()
    cp = 1000 * selected.rare_snv_alt_count.sum() / selected.effective_callable_bp.sum()
    profile = gb.groupby(['species', 'analysis_group', 'bin_index', 'region', 'relative_start', 'relative_end'], as_index=False).agg(gene_count=('unit_id', 'size'), contributing_genes=('effective_callable_bp', lambda x: (x > 0).sum()), total_rare_snv_alt_count=('rare_snv_alt_count', 'sum'), total_effective_callable_bp=('effective_callable_bp', 'sum'))
    profile['pooled_density_per_kb'] = 1000 * profile.total_rare_snv_alt_count / profile.total_effective_callable_bp.replace(0, np.nan)
    profile['density_ratio_to_control'] = profile.pooled_density_per_kb / cp if cp > 0 else np.nan
    profile.to_csv(callable_OUT / 'baboon_profile_summary.tsv', sep='\t', index=False)
    bars = []
    stats = []
    for g in callable_GROUPS:
        v = d[d.analysis_group.eq(g)].rare_snv_density_per_kb
        assert len(v) > 0
        bars.append(dict(species='Baboon', analysis_group=g, n_units=len(v), mean_density_per_kb=v.mean(), mean_density_ratio_to_control=_baboon_controls.divide(v.mean(), cm)))
    for (g1, g2) in combinations(callable_GROUPS, 2):
        (x, y) = [d[d.analysis_group.eq(g)].rare_snv_density_per_kb for g in [g1, g2]]
        test = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic', use_continuity=True)
        stats.append(dict(species='Baboon', group_1=g1, group_2=g2, n1=len(x), n2=len(y), U=test.statistic, p_value=test.pvalue, mean_density_fold=x.mean() / y.mean() if y.mean() else np.nan))
    denom = pd.DataFrame([dict(species='Baboon', control_mean_density_per_kb=cm, control_pooled_density_per_kb=cp, denominator_column='effective_callable_bp')])
    for (name, frame) in zip(callable_NAMES, [d, pd.DataFrame(stats), pd.DataFrame(bars), profile, denom]):
        callable_combine_exact(name, frame)
    for c in cmap:
        path = callable_OUT / f'chr{c}.classified_target_alleles.tsv.gz'
        a = callable_read(path).rename(columns={'counted': 'passes_position_filter'})
        a['in_callable_genome'] = callable_util.within(index, cmap[c], a.pos1.to_numpy() - 1)
        a['counted'] = a.passes_position_filter & a.in_callable_genome
        a.to_csv(path, sep='\t', index=False)
    summary = d.groupby('analysis_group').agg(n_after=('unit_id', 'size'), rare_ALT_SNVs=('rare_snv_alt_count', 'sum'), effective_callable_bp=('effective_callable_bp', 'sum'), zero_SNV_loci=('rare_snv_alt_count', lambda x: x.eq(0).sum()))
    summary['n_before'] = pd.Series(dict(zip(callable_GROUPS, [23, 88, 1570, 10000])))
    summary['mean_ratio_to_control'] = pd.DataFrame(bars).set_index('analysis_group').mean_density_ratio_to_control
    summary.to_csv(callable_OUT / 'baboon_group_summary.tsv', sep='\t')
    plan.update(current_groups=d.groupby('analysis_group').size().to_dict(), old_2681_loci_recount_exact=True, selected_controls=1000, control_raw_callable_bp=150000, control_effective_callable_bp=int(selected.effective_callable_bp.sum()), control_min_effective_bp=int(selected.effective_callable_bp.min()), control_rare_ALT_SNVs=int(selected.rare_snv_alt_count.sum()), control_mean_density_per_kb=float(cm), control_pooled_density_per_kb=float(cp), candidates_failing_effective_ge50=int(controls.effective_callable_bp.lt(50).sum()), genotype_extractions=[dict(**r, derived_vcf=callable_util.record(callable_TMP / f"chr{r['chrom']}.GQ20.snvs.vcf.gz")) for r in manifest])
    (callable_OUT / 'baboon_callable_provenance.json').write_text(json.dumps(plan, indent=2) + '\n')
    overall = json.loads((callable_BASE / 'analysis_summary.json').read_text())
    overall.update(updated_species=['Baboon'], current_baboon_groups=plan['current_groups'], unchanged_non_baboon_species_exact=True, unchanged_non_rat_species_exact=False, baboon_callable_source=str(callable_OUT / 'baboon_callable_provenance.json'))
    (callable_OUT / 'analysis_summary.json').write_text(json.dumps(overall, indent=2) + '\n')
    carry = ['chicken_all_candidate_density_before_callable_filter.tsv', 'rat_callable_filter_audit.tsv', 'rat_callable_excluded.tsv', 'rat_callable_group_summary.tsv', 'rat_control_1000.tsv', 'rat_control_candidate_audit.tsv', 'control_sampling_provenance.json']
    for name in carry:
        shutil.copy2(callable_BASE / name, callable_OUT / name)
    print(summary.to_string(), flush=True)

def callable_entrypoint():
    global callable_args, callable_p, callable_stage
    callable_p = argparse.ArgumentParser(description=__doc__)
    callable_p.add_argument('stage', choices=['prepare', 'extract', 'analyze', 'all'])
    callable_p.add_argument('--workers', type=int, default=4)
    callable_args = callable_p.parse_args()
    callable_setup()
    actions = {'prepare': callable_prepare, 'extract': lambda: callable_extract(callable_args.workers), 'analyze': callable_analyze}
    for callable_stage in ['prepare', 'extract', 'analyze'] if callable_args.stage == 'all' else [callable_args.stage]:
        actions[callable_stage]()

def random_controls_entrypoint():
    _baboon_controls.main()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    _run_stages({'sites': 'sites_entrypoint', 'callable': 'callable_entrypoint', 'random_controls': 'random_controls_entrypoint'}, globals(), setup=_prepare_runtime)
