"""Consolidated final processing: analyze_rat_updated.

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
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import bisect
import pysam
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import os
import importlib.util
import io
import shutil
from scipy.stats import mannwhitneyu
from collections import defaultdict
mask_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/update_rat_callable_fig1.py').parents[1]
mask_RESULT = mask_ROOT / 'results/rat_callable_rn7_20260926'
mask_TMP = mask_ROOT / 'tmp/rat_callable_fig1_20260926'
mask_FASTA = Path('/public5/home/xuejing/rat/rn7/rn7.fa')
mask_CORE_PATH = _component_file('build_fig1_rat_no_callable_20260921')
mask_GROUPS = ('pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes', 'genome_wide_neutral')
mask_NAMES = ('panel_a_gene_locus_density', 'panel_a_pairwise_statistics', 'panel_b_control_mean_ratios', 'panel_c_profile_control_ratios', 'control_denominators')
mask_read = _shared.read_table
mask_sha = _shared.sha256_file

def mask_record(path, content_hash=True):
    path = Path(path)
    result = dict(path=str(path), resolved_path=str(path.resolve()), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    if content_hash:
        result['sha256'] = mask_sha(path)
    return result

def mask_command(args, destination):
    with (mask_TMP / 'commands.jsonl').open('a') as log:
        log.write(json.dumps(args) + '\n')
    with Path(destination).open('wb') as output:
        subprocess.run(args, stdout=output, check=True)

def mask_normalize_bed(path, label, sizes):
    raw = mask_TMP / f'{label}.bed3'
    (counts, bases) = (Counter(), Counter())
    with Path(path).open() as source, raw.open('w') as target:
        for (line_number, line) in enumerate(source, 1):
            if not line.strip() or line.startswith(('#', 'track', 'browser')):
                continue
            fields = line.split()
            (chrom, start, end) = (fields[0], int(fields[1]), int(fields[2]))
            assert chrom in sizes and 0 <= start < end <= sizes[chrom], (path, line_number, fields[:3])
            if label == 'mappability' and len(fields) > 3:
                assert float(fields[3]) == 1
            target.write(f'{chrom}\t{start}\t{end}\n')
            counts[chrom] += 1
            bases[chrom] += end - start
    sorted_path = mask_TMP / f'{label}.sorted.bed'
    merged = mask_TMP / f'{label}.merged.bed'
    mask_command(['sort', '-k1,1', '-k2,2n', '-k3,3n', '-T', str(mask_TMP), str(raw)], sorted_path)
    mask_command(['bedtools', 'merge', '-i', str(sorted_path)], merged)
    return (merged, dict(intervals_by_chrom=dict(counts), raw_bp_by_chrom=dict(bases)))

def mask_intersect_masks(coverage, mappability):
    overlap = mask_TMP / 'coverage_mappability.overlap.bed'
    output = mask_RESULT / 'rat.rn7.coverage_intersect_mappability.callable.bed'
    mask_command(['bedtools', 'intersect', '-sorted', '-a', str(coverage), '-b', str(mappability)], overlap)
    mask_command(['bedtools', 'merge', '-i', str(overlap)], output)
    return output

def mask_index_bed(path):
    data = pd.read_csv(path, sep='\t', header=None, usecols=[0, 1, 2], names=['chrom', 'start', 'end'], dtype={'chrom': str, 'start': np.int64, 'end': np.int64})
    result = {}
    for (chrom, sub) in data.groupby('chrom', sort=False):
        a = sub[['start', 'end']].to_numpy()
        assert (a[:, 1] > a[:, 0]).all()
        assert (a[1:, 0] > a[:-1, 1]).all(), (path, chrom, 'not merged/sorted')
        result[chrom] = a
    return result

def mask_segments(index, chrom, start, end):
    a = index.get(chrom)
    if a is None:
        return []
    left = np.searchsorted(a[:, 1], start, side='right')
    right = np.searchsorted(a[:, 0], end, side='left')
    return [(max(start, int(s)), min(end, int(e))) for (s, e) in a[left:right]]

def mask_within(index, chrom, positions):
    a = index[chrom]
    i = np.searchsorted(a[:, 0], positions, side='right') - 1
    return (i >= 0) & (positions < a[np.maximum(i, 0), 1])

def mask_callable_counts(core, index, positions, chrom, start, end, seq):
    sums = dict(non_acgt_bp=0, mask_bp=0, effective_unmasked_bp=0, rare_snv_alt_count=0, rare_snv_position_count=0)
    raw_callable = 0
    for (s, e) in mask_segments(index, chrom, start, end):
        values = core.interval_counts(chrom, s, e, seq[s - start:e - start], positions)
        raw_callable += e - s
        for key in sums:
            sums[key] += values[key]
    effective = sums['effective_unmasked_bp']
    return dict(region_bp=end - start, raw_callable_bp=raw_callable, non_acgt_bp=sum((base not in 'ACGT' for base in seq)), non_acgt_callable_bp=sums['non_acgt_bp'], callable_bp=raw_callable - sums['non_acgt_bp'], mask_bp=sums['mask_bp'], effective_callable_bp=effective, effective_unmasked_bp=np.nan, rare_snv_alt_count=sums['rare_snv_alt_count'], rare_snv_position_count=sums['rare_snv_position_count'], rare_snv_density_per_kb=1000 * sums['rare_snv_alt_count'] / effective if effective else np.nan)
sites_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/primate/transpecies_compare/scripts/build_fig1_rat_no_callable_20260921.py').parents[1]
sites_TMP = sites_ROOT / 'tmp/fig1_rat_no_callable_20260921'
sites_OUT = sites_ROOT / 'data/processed/fig1_rat_no_callable_20260921'
sites_AF = 0.02

def sites_merge(intervals):
    out = []
    for (s, e) in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]))
        else:
            out.append((s, e))
    return out

def sites_overlaps(xs, s, e):
    i = bisect.bisect_left(xs, (e,))
    return i > 0 and xs[i - 1][1] > s

def sites_classify_positions(chrom):
    positions = defaultdict(dict)
    qc = Counter()
    with pysam.VariantFile(str(sites_TMP / f'{chrom}.GQ20.snvs.vcf.gz')) as vf:
        for rec in vf:
            qc['records'] += 1
            for (i, alt) in enumerate(rec.alts or []):
                ac = rec.info['AC'][i]
                an = rec.info.get('AN', 0)
                valid = len(rec.ref) == len(alt) == 1 and rec.ref in 'ACGT' and (alt in 'ACGT') and (rec.ref != alt)
                af = ac / an if an and 0 <= ac <= an else None
                reasons = []
                if tuple(rec.filter.keys()) != ('PASS',):
                    reasons.append('non_PASS')
                if not valid:
                    reasons.append('noncanonical_SNV')
                if af is None:
                    reasons.append('invalid_AC_AN')
                elif af > sites_AF:
                    reasons.append('AF_above_threshold')
                positions[rec.start][rec.ref, alt] = (ac, an, af, reasons)
    rare = []
    counts = []
    masked = []
    audit = []
    for (pos, alts) in sorted(positions.items()):
        reasons = set((x for a in alts.values() for x in a[3]))
        if reasons:
            masked.append(pos)
        else:
            rare.append(pos)
            counts.append(len(alts))
        for (refalt, a) in alts.items():
            audit.append((chrom, pos + 1, *refalt, *a[:3], ';'.join(sorted(reasons)), not bool(reasons)))
        qc.update(reasons)
    pd.DataFrame(audit, columns=['chrom', 'pos1', 'ref', 'alt', 'AC', 'AN', 'AF', 'position_mask_reason', 'counted']).to_csv(sites_OUT / f'{chrom}.classified_target_alleles.tsv.gz', sep='\t', index=False)
    return (np.array(rare, dtype=int), np.r_[0, np.cumsum(counts)], np.array(masked, dtype=int), dict(qc))

def sites_interval_counts(chrom, s, e, seq, positions):
    (rare, prefix, mask, _) = positions[chrom]
    (left, right) = np.searchsorted(rare, [s, e])
    (ml, mr) = np.searchsorted(mask, [s, e])
    n_non_acgt = sum((x not in 'ACGT' for x in seq))
    eligible_mask = sum((seq[int(pos) - s] in 'ACGT' for pos in mask[ml:mr]))
    alt = int(prefix[right] - prefix[left])
    npos = int(right - left)
    assert all((seq[int(pos) - s] in 'ACGT' for pos in rare[left:right]))
    effective = e - s - n_non_acgt - eligible_mask
    return dict(region_bp=e - s, non_acgt_bp=n_non_acgt, mask_bp=eligible_mask, effective_unmasked_bp=effective, rare_snv_alt_count=alt, rare_snv_position_count=npos, rare_snv_density_per_kb=1000 * alt / effective if effective else np.nan)
variants_util = _component('update_rat_callable_fig1')
variants_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/update_rat_new_vcf_20260928.py').parents[1]
variants_BASE = variants_ROOT / 'data/processed/fig1_baboon_random_control'
variants_OUT = variants_ROOT / 'data/processed/fig1_rat_new_vcf_20260928'
variants_TMP = variants_ROOT / 'tmp/rat_new_vcf_20260928'
variants_RESULT = variants_ROOT / 'results/rat_new_vcf_20260928'
variants_RAT = variants_ROOT / 'data/processed/fig1_rat_control1000_20260927'
variants_CALL = variants_ROOT / 'results/rat_callable_rn7_20260926/rat.rn7.coverage_intersect_mappability.callable.bed'
variants_GROUPS = variants_util.GROUPS
variants_read = variants_util.read

def variants_core_load():
    core = _component('build_fig1_rat_no_callable_20260921')
    core.TMP = variants_TMP
    core.OUT = variants_OUT
    return core

def variants_extract(c):
    source = variants_ROOT / f'rat_vcf/rat_rn7_89inds_autosome_highdepth_chr{c}.GQ20.GTonly.SNPs.vcf.gz'
    before = variants_util.record(source, False)
    idx = variants_TMP / f'chr{c}.source.csi'
    target = variants_TMP / f'chr{c}.targets.bed'
    sig = dict(source=before, target_sha256=variants_util.sha(target))
    marker = variants_TMP / f'chr{c}.extract.json'
    dest = variants_TMP / f'chr{c}.GQ20.snvs.vcf.gz'
    if marker.exists() and json.loads(marker.read_text())['signature'] == sig and dest.exists():
        return json.loads(marker.read_text())
    with (variants_TMP / f'chr{c}.index.log').open('w') as log:
        subprocess.run(['bcftools', 'index', '--csi', '--force', '--output', str(idx), str(source)], stdout=log, stderr=log, check=True)
    with pysam.VariantFile(str(source), index_filename=str(idx)) as vf:
        header = str(vf.header)
        assert len(vf.header.samples) == 0
        for needle in ['FMT/GQ<20', 'fill-tags', 'AC,AF,AN', 'FILTER="PASS"']:
            assert needle in header, (c, needle)
        assert f'chr{c}' in vf.header.contigs
        (variants_OUT / f'chr{c}.source_header.txt').write_text(header)
        n = 0
        with pysam.VariantFile(str(dest), 'wz', header=vf.header) as output:
            for line in target.read_text().splitlines():
                (chrom, s, e) = line.split()
                (s, e) = (int(s), int(e))
                for r in vf.fetch(chrom, s, e):
                    if s <= r.start < e:
                        assert tuple(r.filter) == ('PASS',) and r.info['AN'] > 0
                        output.write(r)
                        n += 1
    subprocess.run(['bcftools', 'index', '--csi', '--force', str(dest)], check=True)
    assert variants_util.record(source, False) == before, 'Source changed during extraction'
    rec = dict(chrom=f'chr{c}', signature=sig, records=n, source_index=variants_util.record(idx), derived_vcf=variants_util.record(dest), status='PASS')
    marker.write_text(json.dumps(rec, indent=2) + '\n')
    print(f'Extracted chr{c}: {n} records', flush=True)
    return rec

def variants_combine(name, frame):
    old = variants_read(variants_BASE / f'{name}.tsv')
    buffer = io.StringIO()
    frame.reindex(columns=old.columns).to_csv(buffer, sep='\t', header=False, index=False)
    inserted = False
    with (variants_OUT / f'{name}.tsv').open('w') as f:
        for line in (variants_BASE / f'{name}.tsv').read_text().splitlines(keepends=True):
            if line.startswith('Rat\t'):
                if not inserted:
                    f.write(buffer.getvalue())
                    inserted = True
            else:
                f.write(line)
    assert inserted
    keep = lambda p: [s for s in p.read_bytes().splitlines(keepends=True) if not s.startswith(b'Rat\t')]
    assert keep(variants_BASE / f'{name}.tsv') == keep(variants_OUT / f'{name}.tsv')

def variants_main():
    for p in [variants_OUT, variants_TMP, variants_RESULT]:
        assert not p.is_symlink()
        p.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(variants_TMP)
    core = variants_core_load()
    prov = json.loads((variants_ROOT / 'data/processed/fig1_rat_callable_20260926/rat_callable_provenance.json').read_text())
    frozen = variants_read(variants_BASE / 'panel_a_gene_locus_density.tsv')
    frozen = frozen[frozen.species.eq('Rat')].copy()
    frozen = frozen.astype({'start': int, 'end': int})
    assert frozen.groupby('analysis_group').size().to_dict() == dict(zip(variants_GROUPS, [46, 37, 847, 1000]))
    regions = {}
    for r in frozen.itertuples():
        flank = 0 if r.analysis_group == variants_GROUPS[-1] else 2000
        regions.setdefault(r.chrom, []).append((r.start - flank, r.end + flank))
    for c in range(1, 21):
        (variants_TMP / f'chr{c}.targets.bed').write_text(''.join((f'chr{c}\t{s}\t{e}\n' for (s, e) in core.merge(regions.get(f'chr{c}', [])))))
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(variants_extract, range(1, 21)))
    (variants_OUT / 'extraction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    positions = {f'chr{c}': core.classify_positions(f'chr{c}') for c in range(1, 21)}
    index = variants_util.index_bed(variants_CALL)
    rows = []
    bins = []
    oldbins = variants_read(variants_ROOT / 'data/processed/fig1_rat_callable_20260926/rat_profile_gene_bins.tsv.gz')
    assert set(oldbins.unit_id) == set(frozen.loc[frozen.analysis_group.ne(variants_GROUPS[-1]), 'unit_id'])
    with pysam.FastaFile(str(variants_util.FASTA)) as fa:
        for r in frozen.itertuples(index=False):
            seq = fa.fetch(r.chrom, r.start, r.end).upper()
            val = variants_util.callable_counts(core, index, positions, r.chrom, r.start, r.end, seq)
            assert val['effective_callable_bp'] >= 50, ('Fixed locus no longer satisfies unchanged >=50-bp rule', r.unit_id, val)
            if r.analysis_group == variants_GROUPS[-1]:
                assert val['callable_bp'] == 150
            row = r._asdict()
            row.update(val, source_note='89 rats; NEW GTonly.SNPs input 20260928; upstream GQ20 and updated hard filters; AF<=0.02 ALT-only; fixed loci and controls')
            rows.append(row)
        for (uid, group) in oldbins.groupby('unit_id', sort=False):
            c = group.chrom.iloc[0]
            start = int(group.genomic_start.min())
            end = int(group.genomic_end.max())
            seq = fa.fetch(c, start, end).upper()
            for r in group.itertuples(index=False):
                row = r._asdict()
                row.update(variants_util.callable_counts(core, index, positions, r.chrom, r.genomic_start, r.genomic_end, seq[r.genomic_start - start:r.genomic_end - start]))
                bins.append(row)
    d = pd.DataFrame(rows)
    gb = pd.DataFrame(bins)
    gb = oldbins[['unit_id', 'bin_index']].merge(gb, on=['unit_id', 'bin_index'], validate='one_to_one')
    d.to_csv(variants_OUT / 'rat_gene_locus_density.tsv', sep='\t', index=False)
    gb.to_csv(variants_OUT / 'rat_profile_gene_bins.tsv.gz', sep='\t', index=False)
    controls = d[d.analysis_group.eq(variants_GROUPS[-1])]
    controls.to_csv(variants_OUT / 'rat_control_1000.tsv', sep='\t', index=False)
    controls[['chrom', 'start', 'end', 'unit_id']].to_csv(variants_RESULT / 'rat.rn7.fixed_control1000.bed', sep='\t', header=False, index=False)
    cm = controls.rare_snv_density_per_kb.mean()
    cp = 1000 * controls.rare_snv_alt_count.sum() / controls.effective_callable_bp.sum()
    bars = []
    stats = []
    for g in variants_GROUPS:
        v = d.loc[d.analysis_group.eq(g), 'rare_snv_density_per_kb']
        bars.append(dict(species='Rat', analysis_group=g, n_units=len(v), mean_density_per_kb=v.mean(), mean_density_ratio_to_control=v.mean() / cm))
    for (g1, g2) in combinations(variants_GROUPS, 2):
        (x, y) = [d.loc[d.analysis_group.eq(g), 'rare_snv_density_per_kb'] for g in [g1, g2]]
        u = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic', use_continuity=True)
        stats.append(dict(species='Rat', group_1=g1, group_2=g2, n1=len(x), n2=len(y), U=u.statistic, p_value=u.pvalue, mean_density_fold=x.mean() / y.mean()))
    profile = gb.groupby(['species', 'analysis_group', 'bin_index', 'region', 'relative_start', 'relative_end'], as_index=False).agg(gene_count=('unit_id', 'size'), contributing_genes=('effective_callable_bp', lambda x: (x > 0).sum()), total_rare_snv_alt_count=('rare_snv_alt_count', 'sum'), total_effective_callable_bp=('effective_callable_bp', 'sum'))
    profile['pooled_density_per_kb'] = 1000 * profile.total_rare_snv_alt_count / profile.total_effective_callable_bp.replace(0, np.nan)
    profile['density_ratio_to_control'] = profile.pooled_density_per_kb / cp
    profile.to_csv(variants_OUT / 'rat_profile_summary.tsv', sep='\t', index=False)
    denom = pd.DataFrame([dict(species='Rat', control_mean_density_per_kb=cm, control_pooled_density_per_kb=cp, denominator_column='effective_callable_bp')])
    for (name, frame) in zip(variants_util.NAMES, [d, pd.DataFrame(stats), pd.DataFrame(bars), profile, denom]):
        variants_combine(name, frame)
    for c in range(1, 21):
        path = variants_OUT / f'chr{c}.classified_target_alleles.tsv.gz'
        a = variants_read(path).rename(columns={'counted': 'passes_position_filter'})
        a['in_callable_genome'] = variants_util.within(index, f'chr{c}', a.pos1.to_numpy() - 1)
        a['counted'] = a.passes_position_filter & a.in_callable_genome
        a.to_csv(path, sep='\t', index=False)
    comparison = frozen[['unit_id', 'analysis_group', 'rare_snv_alt_count', 'effective_callable_bp', 'rare_snv_density_per_kb']].merge(d[['unit_id', 'rare_snv_alt_count', 'effective_callable_bp', 'rare_snv_density_per_kb']], on='unit_id', suffixes=('_old', '_new'), validate='one_to_one')
    comparison.to_csv(variants_OUT / 'rat_old_vs_new_locus_comparison.tsv', sep='\t', index=False)
    pd.DataFrame(bars).to_csv(variants_OUT / 'rat_group_summary.tsv', sep='\t', index=False)
    for name in ['baboon_control_1000.tsv', 'baboon_control_candidate_audit.tsv', 'baboon_group_summary.tsv', 'seed_plan.json', 'selected_ids.tsv']:
        shutil.copy2(variants_BASE / name, variants_OUT / name)
    for name in ['baboon_callable_filter_audit.tsv', 'baboon_callable_excluded.tsv', 'chicken_all_candidate_density_before_callable_filter.tsv']:
        if (variants_BASE / name).is_file():
            shutil.copy2(variants_BASE / name, variants_OUT / name)
    summary = json.loads((variants_BASE / 'analysis_summary.json').read_text())
    summary.update(updated_species=['Rat'], rat_variant_update='GTonly.SNPs 20260928; all other parameters and loci fixed')
    (variants_OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    details = dict(updated_species='Rat', source_pattern='rat_vcf/*GTonly.SNPs.vcf.gz', nominal_individuals=89, source_sites_only=True, source_upstream_GQ20=True, AC_AN_reused_from_new_source=True, AF_max=0.02, counting='ALT only', fixed_locus_counts=d.groupby('analysis_group').size().to_dict(), control_resampled=False, callable=prov['callable_output'], coverage=prov['coverage'], mappability=prov['mappability'], control_mean=float(cm), control_pooled=float(cp), base=str(variants_BASE), other_six_species_rows_byte_identical=True, limitation='New input has already removed failing/fixed ALT records; absent records cannot be used as position masks. No parent VCF or new filter/AN threshold introduced.')
    (variants_OUT / 'provenance.json').write_text(json.dumps(details, indent=2) + '\n')
    print(pd.DataFrame(bars).to_string(index=False), flush=True)

def variants_entrypoint():
    variants_main()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    # The frozen Fig1 set contains 847 rat pseudogenes; do not expand it.
    _run_stages({'variants': 'variants_entrypoint'}, globals(), setup=_prepare_runtime)
