"""Consolidated final processing: count_mutations_updated.

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
import argparse
import hashlib
import itertools
import json
import subprocess
from collections import Counter
import pandas as pd
import pysam
from cyvcf2 import VCF
import sys
import shutil
import os
counter_ROOT = (_PROCESSING_ROOT / 'fig2/scripts/recount_human_pseudo_20261005.py').parents[1]
counter_OUT = counter_ROOT / 'data/processed/figure2_pseudo1362_legacy_intervals_20261005'
counter_TMP = counter_ROOT / 'tmp/figure2_human_pseudo_20261005'
counter_SRC = counter_ROOT / 'fig1_gene_rules_20261004'
counter_FASTA = Path('/public/home/jiangzj/tools/annovar/humandb/hg38_seq/hg38.fa')
counter_MASK = counter_ROOT / 'transpecies_compare/hg38.genmap.k150e1_unique.cov_16-48.bed'
counter_SUMMARY = counter_ROOT / 'data/processed/figure2_fig1_human_populations_20260825/human_population_result_summary.tsv'
counter_COMP = str.maketrans('ACGT', 'TGCA')
counter_sha = _shared.sha256_file

def counter_rc(s):
    return s.translate(counter_COMP)[::-1]

def counter_prepare():
    counter_OUT.mkdir(parents=True, exist_ok=True)
    counter_TMP.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(counter_SRC / 'gene_sets/Human/gene_sets.tsv', sep='\t')
    genes = genes[genes.group == 'RNU_pseudogenes'].copy()
    assert len(genes) == 1362 and genes.gene_id.is_unique
    old = pd.read_csv(counter_ROOT / 'data/processed/figure2_fig1_gene_sets_20260825/fig1_gene_membership_callable_status.tsv', sep='\t')
    old = old[(old.species == 'Human') & (old.group == 'Pseudo-RNUs')]
    changes = [dict(gene_id=r.gene_id, gene_name=r.gene_name, change='added') for r in genes.itertuples() if r.gene_id not in set(old.stable_gene_id)]
    changes += [dict(gene_id=r.stable_gene_id, gene_name=r.display_name, change='removed') for r in old.itertuples() if r.stable_gene_id not in set(genes.gene_id)]
    pd.DataFrame(changes).to_csv(counter_OUT / 'membership_changes.tsv', sep='\t', index=False)
    genes.to_csv(counter_OUT / 'human_pseudogene_membership.tsv', sep='\t', index=False)
    command = ['bedtools', 'intersect', '-a', str(counter_SRC / 'gene_sets/Human/RNU_pseudogenes.bed'), '-b', str(counter_MASK)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = [dict(chrom=f[0], start=int(f[1]), end=int(f[2]), gene_id=f[3], strand=f[5]) for line in result.stdout.splitlines() if (f := line.split('\t'))]
    regions = pd.DataFrame(rows).drop_duplicates().sort_values(['chrom', 'start', 'end', 'gene_id', 'strand'])
    assert set(regions.gene_id) == set(genes.gene_id)
    assert regions.groupby('gene_id').strand.nunique().max() == 1
    regions['interval_bp'] = regions.end - regions.start
    check = genes.set_index('gene_id').callable_bp.sort_index()
    observed = regions.groupby('gene_id').interval_bp.sum().sort_index()
    assert (check == observed).all(), (check - observed)[check != observed]
    regions.to_csv(counter_OUT / 'fig1_regions.tsv', sep='\t', index=False)
    regions[['chrom', 'start', 'end', 'strand']].to_csv(counter_OUT / 'fig1_regions.bed', sep='\t', header=False, index=False)
    previous = pd.read_csv(counter_ROOT / 'data/processed/figure2_fig1_gene_sets_20260825/callable_strand_interval_membership.tsv', sep='\t')
    previous = previous[(previous.species == 'Human') & (previous.group == 'Pseudo-RNUs')].rename(columns={'stable_gene_id': 'gene_id'})
    retained = previous[previous.gene_id.isin(genes.gene_id)][['chrom', 'start', 'end', 'gene_id', 'strand', 'interval_bp']]
    missing = set(genes.gene_id) - set(retained.gene_id)
    compatible = pd.concat([retained, regions[regions.gene_id.isin(missing)]], ignore_index=True)
    compatible.to_csv(counter_OUT / 'legacy_regions.tsv', sep='\t', index=False)
    compatible[['chrom', 'start', 'end', 'strand']].to_csv(counter_OUT / 'legacy_regions.bed', sep='\t', header=False, index=False)
    info = dict(new_genes=len(genes), previous_genes=len(old), added=sum((r['change'] == 'added' for r in changes)), removed=sum((r['change'] == 'removed' for r in changes)), fig1_region_rows=len(regions), fig1_callable_bp=int(regions.interval_bp.sum()), legacy_region_rows=len(compatible), legacy_callable_bp=int(compatible.interval_bp.sum()), genome=str(counter_FASTA), mask=str(counter_MASK), mask_sha256=counter_sha(counter_MASK), gene_list_sha256=counter_sha(counter_SRC / 'gene_sets/Human/gene_sets.tsv'), intended_scope='Fixed October4 human gene list across cohorts; no new cohort-specific gene filtering.')
    (counter_OUT / 'region_comparison.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)

def counter_count(bed, vcfpath, *, fasta_path=None, max_af=0.001):
    bases = 'ATCG'
    contexts = [''.join(v) for v in itertools.product(bases, repeat=3)]
    result = {'1mer': {'context': {x: 0 for x in [*bases, 'CpG']}, 'variant': {}}, '3mer': {'context': dict.fromkeys(contexts, 0), 'variant': {}}}
    for ref in bases:
        for alt in bases:
            if ref == alt:
                continue
            result['1mer']['variant'][f'{ref}>{alt}'] = {'mask': 0, 'count': 0}
            if ref in 'CG':
                result['1mer']['variant'][f'{ref}(CpG)>{alt}'] = {'mask': 0, 'count': 0}
    for k in contexts:
        for b in bases:
            if b != k[1]:
                result['3mer']['variant'][f'{k}>{k[0]}{b}{k[2]}'] = {'mask': 0, 'count': 0}
    vcf = VCF(str(vcfpath))
    fasta = pysam.FastaFile(str(counter_FASTA if fasta_path is None else fasta_path))
    audit = Counter()
    for line in Path(bed).read_text().splitlines():
        (chrom, start, end, strand) = line.split('\t')[:4]
        start = int(start)
        end = int(end)
        seq = fasta.fetch(chrom, start - 1, end + 1).upper()
        if strand == '-':
            seq = counter_rc(seq)
        for i in range(len(seq) - 2):
            k = seq[i:i + 3]
            if k in result['3mer']['context']:
                result['3mer']['context'][k] += 1
        one = result['1mer']['context']
        one['CpG'] += seq.count('CG') * 2
        if seq.startswith('CG'):
            one['CpG'] -= 1
            one[seq[1]] -= 1
        if seq.endswith('CG'):
            one['CpG'] -= 1
            one[seq[-2]] -= 1
        inner = seq[1:-1].replace('CG', '')
        for b in bases:
            one[b] += inner.count(b)
        for rec in vcf(f'{chrom}:{start + 1}-{end}'):
            (ref, alt) = (rec.REF, rec.ALT[0])
            if ref not in bases or alt not in bases or len(ref) != 1 or (len(alt) != 1):
                continue
            (ac, an) = (rec.INFO.get('AC'), rec.INFO.get('AN'))
            if ac is None or an in (None, 0):
                raise ValueError(f'Missing AC/AN at {chrom}:{rec.POS}; no silent AF imputation')
            assert not isinstance(ac, tuple), 'Legacy counter expects split ALT records'
            state = 'mask' if rec.FILTER is not None or ac / an > max_af else 'count'
            k = fasta.fetch(chrom, rec.POS - 2, rec.POS + 1).upper()
            assert k[1] == ref, (chrom, rec.POS, ref, k)
            if strand == '-':
                ref = ref.translate(counter_COMP)
                alt = alt.translate(counter_COMP)
                k = counter_rc(k)
            if k not in result['3mer']['context']:
                audit['noncanonical_context_skipped'] += 1
                continue
            result['3mer']['variant'][f'{k}>{k[0]}{alt}{k[-1]}'][state] += 1
            key = f'{ref}(CpG)>{alt}' if 'CG' in k else f'{ref}>{alt}'
            result['1mer']['variant'][key][state] += 1
            audit[state] += 1
    vcf.close()
    fasta.close()
    return (result, dict(audit))

def counter_main():
    global counter_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--population', choices=['ASJ', 'EAS', 'SAS', 'FIN', 'SHUF2000'])
    parser.add_argument('--regions', choices=['fig1', 'legacy'], default='legacy',
                        help='Interval policy; legacy reproduces the frozen Fig2 and cohort results.')
    parser.add_argument('--data-dir', type=Path,
                        help='Override the policy-specific processed-data directory.')
    args = parser.parse_args()
    dirname = ('figure2_pseudo1362_legacy_intervals_20261005' if args.regions == 'legacy'
               else 'figure2_human_pseudo_20261005')
    counter_OUT = (args.data_dir or counter_ROOT / 'data/processed' / dirname).resolve()
    if args.prepare:
        counter_prepare()
        return
    counter_OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(counter_SUMMARY, sep='\t')
    row = summary[(summary.population == args.population) & (summary.group == 'Pseudo-RNUs')].iloc[0]
    (oldresult, audit) = counter_count(counter_ROOT / row.bed_path, counter_ROOT / row.vcf_path)
    assert oldresult == json.loads((counter_ROOT / row.result_path).read_text()), 'Indexed counter does not reproduce legacy input'
    output = counter_OUT / args.population
    output.mkdir(exist_ok=True)
    (result, newaudit) = counter_count(counter_OUT / f'{args.regions}_regions.bed', counter_ROOT / row.vcf_path)
    (output / 'pseudo.result').write_text(json.dumps(result) + '\n')
    info = dict(population=args.population, max_af=0.001, nominal_individuals=int(row.nominal_individuals), region_policy=args.regions, bed=str(counter_OUT / f'{args.regions}_regions.bed'), old_counter_reproduction=True, old_records=audit, new_records=newaudit, genes=1362, new_result_sha256=counter_sha(output / 'pseudo.result'), vcf=str(counter_ROOT / row.vcf_path), vcf_size=(counter_ROOT / row.vcf_path).stat().st_size, vcf_mtime=(counter_ROOT / row.vcf_path).stat().st_mtime, old_result_sha256=row.result_sha256)
    (output / 'recount_manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)

def counter_entrypoint():
    counter_main()
mouse_counter = _component('recount_human_pseudo_20261005')
mouse_ROOT = mouse_counter.ROOT
mouse_OUT = mouse_ROOT / 'data/processed/figure2_mouse_pseudo1032_20261005'
mouse_SOURCE = mouse_ROOT / 'fig1_gene_rules_20261004/gene_sets/Mouse'
mouse_MASK = mouse_ROOT / 'transpecies_compare/mm10.genmap.k150e1_unique.cov_81-243.bed'
mouse_FASTA = Path('/public5/home/xuejing/wildmouse/index/mm10.fa')
mouse_VCF = Path('/public4/home/jiangzj/analysis/mtr/histone_mutation_explore/all_species_summary/data/variant_data/mouse/mouse_148ind_autosome_allchr_10Xhighdepth_variants.snps.hardfilter.GQ20_missing.vcf.gz')
mouse_OLD = mouse_ROOT / 'data/processed/figure2_fig1_gene_sets_20260825'

def mouse_main():
    mouse_OUT.mkdir(exist_ok=True, parents=True)
    genes = pd.read_csv(mouse_SOURCE / 'gene_sets.tsv', sep='\t')
    genes = genes[genes.group == 'RNU_pseudogenes'].copy()
    prior = pd.read_csv(mouse_OLD / 'fig1_gene_membership_callable_status.tsv', sep='\t')
    prior = prior[(prior.species == 'Mouse') & (prior.group == 'Pseudo-RNUs')]
    assert len(genes) == 1032 and genes.gene_id.is_unique and (len(prior) == 1007)
    added = set(genes.gene_id) - set(prior.stable_gene_id)
    assert len(added) == 25
    assert not set(prior.stable_gene_id) - set(genes.gene_id)
    old = pd.read_csv(mouse_OLD / 'callable_strand_interval_membership.tsv', sep='\t')
    old = old[(old.species == 'Mouse') & (old.group == 'Pseudo-RNUs')].rename(columns={'stable_gene_id': 'gene_id'})
    keys = ['chrom', 'start', 'end', 'gene_id', 'strand', 'interval_bp']
    old = old[keys]
    absent = set(genes.gene_id) - set(old.gene_id)
    result = subprocess.run(['bedtools', 'intersect', '-a', str(mouse_SOURCE / 'RNU_pseudogenes.bed'), '-b', str(mouse_MASK)], capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        (chrom, start, end, gene_id, score, strand) = line.split('\t')[:6]
        if gene_id in absent:
            rows.append(dict(chrom=chrom, start=int(start), end=int(end), gene_id=gene_id, strand=strand, interval_bp=int(end) - int(start)))
    supplement = pd.DataFrame(rows).drop_duplicates()
    assert set(supplement.gene_id) == absent
    new = pd.concat([old, supplement], ignore_index=True)
    assert set(new.gene_id) == set(genes.gene_id)
    assert Counter(map(tuple, new[new.gene_id.isin(old.gene_id)][keys].values)) == Counter(map(tuple, old[keys].values))
    new.to_csv(mouse_OUT / 'mouse_pseudo_regions.tsv', sep='\t', index=False)
    new[['chrom', 'start', 'end', 'strand']].to_csv(mouse_OUT / 'mouse_pseudo_regions.bed', sep='\t', index=False, header=False)
    old[['chrom', 'start', 'end', 'strand']].to_csv(mouse_OUT / 'old_mouse_pseudo_regions.bed', sep='\t', index=False, header=False)
    genes.to_csv(mouse_OUT / 'mouse_pseudogene_membership.tsv', sep='\t', index=False)
    added_genes = genes[genes.gene_id.isin(added)].copy()
    added_genes['change'] = 'added'
    added_genes.to_csv(mouse_OUT / 'mouse_membership_changes.tsv', sep='\t', index=False)
    old_result = mouse_OLD / 'results/mouse/RNU_pseudogenes_fig1_af1e-2.result'
    (legacy, oldaudit) = mouse_counter.count(mouse_OUT / 'old_mouse_pseudo_regions.bed', mouse_VCF, fasta_path=mouse_FASTA, max_af=0.01)
    assert legacy == json.loads(old_result.read_text()), 'Indexed count does not reproduce every legacy 1mer/3mer value'
    (payload, audit) = mouse_counter.count(mouse_OUT / 'mouse_pseudo_regions.bed', mouse_VCF, fasta_path=mouse_FASTA, max_af=0.01)
    path = mouse_OUT / 'mouse_pseudo1032_af1e-2.result'
    path.write_text(json.dumps(payload) + '\n')
    info = dict(species='Mouse', assembly='mm10', nominal_individuals=148, max_af=0.01, old_genes=1007, new_genes=1032, added_genes=25, removed_genes=0, genes_missing_old_intervals=len(absent), new_interval_rows=len(new), new_interval_bp=int(new.interval_bp.sum()), old_interval_rows=len(old), old_interval_bp=int(old.interval_bp.sum()), old_count_reproduced_exactly=True, retained_legacy_rows_unchanged=True, old_records=oldaudit, new_records=audit, region_policy='Preserve all old interval rows and strands; supplement genes without old intervals using Fig1 coordinates intersected with original callable whitelist', result_path=str(path.relative_to(mouse_ROOT)), result_sha256=mouse_counter.sha(path), bed_sha256=mouse_counter.sha(mouse_OUT / 'mouse_pseudo_regions.bed'), gene_list_path=str(mouse_SOURCE / 'gene_sets.tsv'), gene_list_sha256=mouse_counter.sha(mouse_SOURCE / 'gene_sets.tsv'), callable_mask=str(mouse_MASK), callable_mask_sha256=mouse_counter.sha(mouse_MASK), vcf=str(mouse_VCF), vcf_size=mouse_VCF.stat().st_size, vcf_mtime=mouse_VCF.stat().st_mtime, fasta=str(mouse_FASTA))
    (mouse_OUT / 'recount_manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)

def mouse_entrypoint():
    mouse_main()
asj_ROOT = (_PROCESSING_ROOT / 'fig2/scripts/plot_figure2_pseudo_legacy_intervals_20261005.py').parents[1]
asj_DATA = asj_ROOT / 'data/processed/figure2_pseudo1362_legacy_intervals_20261005'
asj_recount = _component('recount_human_pseudo_20261005')

def asj_analyze():
    previous = asj_recount.OUT
    asj_DATA.mkdir(parents=True, exist_ok=True)
    for name in ['legacy_regions.bed', 'legacy_regions.tsv', 'human_pseudogene_membership.tsv', 'membership_changes.tsv']:
        if (previous / name).resolve() != (asj_DATA / name).resolve():
            shutil.copy2(previous / name, asj_DATA / name)
    regions = pd.read_csv(asj_DATA / 'legacy_regions.tsv', sep='\t')
    assert regions.gene_id.nunique() == 1362 and len(regions) == 1745
    assert regions.interval_bp.sum() == 185648
    summary = pd.read_csv(asj_recount.SUMMARY, sep='\t')
    row = summary[(summary.population == 'ASJ') & (summary.group == 'Pseudo-RNUs')].iloc[0]
    (old, audit) = asj_recount.count(asj_ROOT / row.bed_path, asj_ROOT / row.vcf_path)
    assert old == json.loads((asj_ROOT / row.result_path).read_text())
    (result, newaudit) = asj_recount.count(asj_DATA / 'legacy_regions.bed', asj_ROOT / row.vcf_path)
    target = asj_DATA / 'ASJ'
    target.mkdir(exist_ok=True)
    (target / 'pseudo.result').write_text(json.dumps(result) + '\n')
    manifest = dict(population='ASJ', genes=1362, interval_rows=len(regions), interval_bp=int(regions.interval_bp.sum()), region_policy='Preserve old interval multiplicity and strand rows for retained genes; add Fig1 callable intervals only for genes absent from old input', bed=str(asj_DATA / 'legacy_regions.bed'), bed_sha256=asj_recount.sha(asj_DATA / 'legacy_regions.bed'), max_af=0.001, nominal_individuals=1736, old_counter_reproduction=True, old_records=audit, new_records=newaudit, new_result_sha256=asj_recount.sha(target / 'pseudo.result'), vcf=str(asj_ROOT / row.vcf_path), vcf_size=(asj_ROOT / row.vcf_path).stat().st_size, vcf_mtime=(asj_ROOT / row.vcf_path).stat().st_mtime)
    (target / 'recount_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
    subprocess.run([sys.executable, str(_component_file('analyze_human_pseudo_20261005')), 'human', '--population', 'ASJ', '--data-dir', str(asj_DATA), '--region-policy', 'legacy'], check=True)

def asj_entrypoint():
    asj_analyze()
cohorts_ROOT = (_PROCESSING_ROOT / 'fig2/scripts/plot_figS11_S14_pseudo_legacy_20261005.py').parents[1]
cohorts_DATA = cohorts_ROOT / 'data/processed/figure2_pseudo1362_legacy_intervals_20261005'
cohorts_POPS = {'EAS': (11, 'EAS'), 'SAS': (12, 'SAS'), 'FIN': (13, 'FIN'), 'SHUF2000': (14, 'random2000')}

def cohorts_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--population', choices=cohorts_POPS, required=True)
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    pop = args.population
    tmp = cohorts_ROOT / f'tmp/figS11_S14_pseudo_legacy_20261005/{pop}'
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(tmp)
    os.environ['MPLCONFIGDIR'] = str(tmp / 'mplconfig')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    import pandas as pd
    recount = _component('recount_human_pseudo_20261005')
    (number, label) = cohorts_POPS[pop]
    out = cohorts_ROOT / f'final_result/figS{number}/pseudo1362_legacy_intervals_20261005'
    out.mkdir(parents=True, exist_ok=True)
    bed = cohorts_DATA / 'legacy_regions.bed'
    reference_manifest = json.loads((cohorts_DATA / 'ASJ/recount_manifest.json').read_text())
    if not args.plot_only:
        rows = pd.read_csv(recount.SUMMARY, sep='\t')
        row = rows[(rows.population == pop) & (rows.group == 'Pseudo-RNUs')].iloc[0]
        (legacy, oldcounts) = recount.count(cohorts_ROOT / row.bed_path, cohorts_ROOT / row.vcf_path)
        assert legacy == json.loads((cohorts_ROOT / row.result_path).read_text()), pop
        (result, newcounts) = recount.count(bed, cohorts_ROOT / row.vcf_path)
        destination = cohorts_DATA / pop
        destination.mkdir(exist_ok=True)
        (destination / 'pseudo.result').write_text(json.dumps(result) + '\n')
        manifest = dict(population=pop, nominal_individuals=int(row.nominal_individuals), max_af=0.001, genes=1362, interval_rows=1745, interval_bp=185648, region_policy=reference_manifest['region_policy'], bed=str(bed), bed_sha256=recount.sha(bed), same_bed_as_current_ASJ_Fig2=True, old_counter_reproduction=True, old_records=oldcounts, new_records=newcounts, new_result_sha256=recount.sha(destination / 'pseudo.result'), vcf=str(cohorts_ROOT / row.vcf_path), vcf_size=(cohorts_ROOT / row.vcf_path).stat().st_size, vcf_mtime=(cohorts_ROOT / row.vcf_path).stat().st_mtime)
        (destination / 'recount_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest, indent=2), flush=True)
        subprocess.run([sys.executable, str(_component_file('analyze_human_pseudo_20261005')), 'human', '--population', pop, '--data-dir', str(cohorts_DATA), '--region-policy', 'legacy'], check=True)

def cohorts_entrypoint():
    cohorts_main()

def _prepare_runtime():
    sys.dont_write_bytecode = True
    sys.dont_write_bytecode = True

if __name__ == '__main__':
    _run_stages({'counter': 'counter_entrypoint', 'mouse': 'mouse_entrypoint', 'asj': 'asj_entrypoint', 'cohorts': 'cohorts_entrypoint'}, globals(), setup=_prepare_runtime)
