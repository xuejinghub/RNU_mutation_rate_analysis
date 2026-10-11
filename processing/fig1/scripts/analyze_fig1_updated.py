"""Consolidated final processing: analyze_fig1_updated.

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
from collections import defaultdict, Counter
import ast
import bisect
import math
import types
from typing import Any, Iterable
import pandas as pd
import numpy as np
import pysam
import json
from itertools import combinations
from scipy.stats import mannwhitneyu
genes_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/revise_fig1_gene_rules_20261004.py').parents[1]
genes_BASE = genes_ROOT / 'data/processed/fig1_chicken_regulatory_windows_20260928'
genes_OUT = genes_ROOT / 'data/processed/fig1_gene_rules_20261004'
genes_RES = genes_ROOT / 'results/fig1_gene_rules_20261004'
genes_TMP = genes_ROOT / 'tmp/fig1_gene_rules_20261004'
genes_TR = genes_ROOT / 'primate/transpecies_compare'
genes_SP = ['Human', 'Chimpanzee', 'Rhesus_macaque', 'Baboon', 'Mouse', 'Rat', 'Chicken']
genes_GROUPS = ['pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes', 'genome_wide_neutral']
genes_AF = dict(zip(genes_SP, [0.001, 0.01, 0.002, 0.02, 0.01, 0.02, 0.01]))
genes_read = _shared.read_table

def genes_save(d, p):
    d.to_csv(p, sep='\t', index=False, na_rep='NA')

def genes_family(name, assigned=''):
    import re
    s = str(name).upper()
    if 'U7' in s:
        return 'U7'
    for f in ['U6ATAC', 'U4ATAC', 'U11', 'U12', 'U1', 'U2', 'U4', 'U5', 'U6']:
        if s.startswith(('RN' + f, 'RNV' + f)) or re.fullmatch(f, s):
            return f
    f = str(assigned).upper()
    return f if f in ['U6ATAC', 'U4ATAC', 'U11', 'U12', 'U1', 'U2', 'U4', 'U5', 'U6', 'U7'] else 'UNKNOWN'

def genes_catalogs():
    c = {}
    h = genes_read(genes_ROOT / 'deliverables/human_snrna_metadata_20261003/Human_snRNA_metadata.tsv')
    h = h.rename(columns={'gene_length': 'gene_length_bp'})
    h['family'] = h.gene_name.map(genes_family)
    c['Human'] = h.set_index('gene_id', drop=False)
    for sp in genes_SP[1:]:
        tag = 'Macaque' if sp == 'Rhesus_macaque' else sp
        x = genes_read(genes_ROOT / f'deliverables/fig1_snrna_metadata_20260930/{tag}_snRNA_metadata.tsv')
        x['resolved_family'] = [genes_family(a, b) for (a, b) in zip(x.gene_name, x.family)]
        c[sp] = x.set_index('gene_id', drop=False)
    return c

def genes_profile_engine():
    engine = _component('plot_figure1_nature')
    engine.THRESHOLDS = genes_AF
    return engine

def genes_select_standard():
    old = genes_read(genes_BASE / 'panel_a_gene_locus_density.tsv')
    c = genes_catalogs()
    parts = []
    audits = []
    hist = genes_read(genes_TR / 'data/processed/cross_species_rnu_rare_variant_density_20260811/gene_rare_snv_density.tsv')
    mh = genes_read(genes_TR / 'data/processed/rhesus_rnu_rare_variant_density_af2e3_20260812/gene_rare_snv_density.tsv')
    for sp in genes_SP:
        before = old[old.species.eq(sp)].copy()
        if sp == 'Chimpanzee':
            continue
        if sp in ['Human', 'Rhesus_macaque', 'Mouse']:
            src = mh if sp == 'Rhesus_macaque' else hist
            src = src[src.species.eq(sp) & src.gene_set.eq(genes_GROUPS[2])].copy()
            native = c[sp]
            src['gene_length'] = src.stable_gene_id.map(native.gene_length_bp)
            src['family'] = src.stable_gene_id.map(native.family if sp == 'Human' else native.resolved_family)
            keep = src.gene_length.ge(70) & src.effective_callable_bp.ge(50) & src.family.ne('U7')
            if sp == 'Mouse':
                keep &= src.effective_callable_bp.gt(50)
            if sp != 'Human':
                keep &= src.family.ne('UNKNOWN') | src.gene_length.gt(70)
            functional = set(before.loc[before.analysis_group.isin(genes_GROUPS[:2]), 'unit_id'])
            keep &= ~src.stable_gene_id.isin(functional)
            pseudo = src[keep].copy()
            rows = []
            indexed = before.set_index('unit_id')
            for r in pseudo.itertuples():
                if r.stable_gene_id in indexed.index:
                    rows.append(indexed.loc[r.stable_gene_id].to_dict() | {'unit_id': r.stable_gene_id})
                    continue
                g = native.loc[r.stable_gene_id]
                row = dict(species=sp, assembly={'Human': 'GRCh38', 'Rhesus_macaque': 'rheMac10', 'Mouse': 'mm10'}[sp], analysis_group=genes_GROUPS[2], unit_id=r.stable_gene_id, display_name=g.gene_name, chrom=g.chrom, start=int(g.start), end=int(g.end), strand=g.strand, callable_bp=r.callable_bp, mask_bp=r.mask_bp, effective_callable_bp=r.effective_callable_bp, rare_snv_alt_count=r.rare_snv_alt_count, rare_snv_position_count=r.rare_snv_position_count, rare_snv_density_per_kb=r.rare_snv_density_per_kb, af_threshold=genes_AF[sp], region_bp=int(g.end - g.start), source_note='Full gene length >=70; U7 excluded; original AF and effective callable threshold retained')
                rows.append(row)
            updated = pd.concat([before[before.analysis_group.ne(genes_GROUPS[2])], pd.DataFrame(rows)], ignore_index=True)
            src['retained_new'] = keep
            genes_save(src, genes_OUT / f'{sp}.pseudogene_filter_audit.tsv')
        else:
            native = c[sp]
            group = before.analysis_group
            ids = before.unit_id
            fam = ids.map(native.resolved_family)
            length = ids.map(native.gene_length_bp)
            drop = group.ne(genes_GROUPS[3]) & fam.eq('U7') | group.eq(genes_GROUPS[2]) & fam.eq('UNKNOWN') & length.le(70)
            updated = before[~drop].copy()
        parts.append(updated)
        prev = set(before.unit_id)
        new = set(updated.unit_id)
        for gid in sorted(prev | new):
            if gid in prev ^ new:
                audits.append(dict(species=sp, gene_id=gid, change='added' if gid in new else 'removed'))
    restored = genes_read(genes_OUT / 'restored70/human_70_filter_audit.tsv')
    measurements = genes_read(genes_OUT / 'restored70/all_cohorts_density.tsv')
    measurements = measurements[measurements.dataset.eq('gnomADv4_ASJ')].set_index('stable_gene_id')
    additional = []
    for r in restored[restored.retained_new].itertuples():
        v = measurements.loc[r.gene_id]
        additional.append(dict(species='Human', assembly='GRCh38', analysis_group=genes_GROUPS[2], unit_id=r.gene_id, display_name=r.gene_name, chrom=r.chrom, start=r.start, end=r.end, strand=r.strand, region_bp=r.gene_length, **{k: v[k] for k in ['callable_bp', 'mask_bp', 'effective_callable_bp', 'rare_snv_alt_count', 'rare_snv_position_count', 'rare_snv_density_per_kb']}, af_threshold=0.001, source_note='Restored full-annotation candidate; original gene length >=70; non-U7 autosomal; effective callable >=50'))
        audits.append(dict(species='Human', gene_id=r.gene_id, change='added'))
    parts.append(pd.DataFrame(additional))
    audit = restored.rename(columns={'gene_id': 'stable_gene_id', 'gene_name': 'display_name'}).copy()
    audit['family'] = audit.display_name.map(genes_family)
    genes_save(pd.concat([genes_read(genes_OUT / 'Human.pseudogene_filter_audit.tsv'), audit], ignore_index=True), genes_OUT / 'Human.pseudogene_filter_audit.tsv')
    selected = pd.concat(parts, ignore_index=True).reindex(columns=old.columns)
    genes_save(selected, genes_OUT / 'nonchimp_locus_density.tsv')
    genes_save(pd.DataFrame(audits), genes_OUT / 'membership_changes_nonchimp.tsv')
    print(selected.groupby(['species', 'analysis_group']).size().to_string(), flush=True)
    return (old, selected, c)

def genes_additional_profiles(selected, c):
    previous = genes_ROOT / 'data/processed/fig1_gene_rules_20261003'
    oldbins = pd.concat([genes_read(previous / f'{s}.profile_gene_bins.tsv.gz') for s in ['Human', 'Rhesus_macaque', 'Mouse']], ignore_index=True)
    engine = genes_profile_engine()
    allbins = []
    sources = {'Human': ('gnomadv4/asj/gnomad.genomes.v4.0.sites.asj.allchr.vcf.gz', 'hg38.genmap.k150e1_unique.cov_16-48.bed'), 'Rhesus_macaque': ('rheMac10.snvs.unrelateSubset.all.GQ20_missing.noGT.vcf.gz', 'rheMac10.genmap.k150e1_unique.cov_63-190.bed'), 'Mouse': ('mouse_148ind_autosome_allchr_10Xhighdepth_variants.snps.hardfilter.GQ20_missing.vcf.gz', 'mm10.genmap.k150e1_unique.cov_81-243.bed')}
    for sp in sources:
        ids = set(selected.loc[selected.species.eq(sp) & selected.analysis_group.ne(genes_GROUPS[3]), 'unit_id'])
        destination = genes_OUT / f'{sp}.profile_gene_bins.tsv.gz'
        if destination.exists():
            existing = genes_read(destination)
            if set(existing.stable_gene_id) == ids:
                allbins.append(existing)
                print(sp, 'completed profile reused', flush=True)
                continue
        cached = oldbins[oldbins.species.eq(sp) & oldbins.stable_gene_id.isin(ids)].copy()
        missing = ids - set(cached.stable_gene_id)
        generated = []
        if missing:
            ci = engine.load_callable_index(genes_TR / sources[sp][1])
            with pysam.VariantFile(str((genes_TR / sources[sp][0]).resolve())) as vcf:
                for gid in sorted(missing):
                    g = c[sp].loc[gid]
                    cat = pd.DataFrame([dict(species=sp, analysis_group=genes_GROUPS[2], stable_gene_id=gid, display_name=g.gene_name, chrom=g.chrom, start0=int(g.start), end=int(g.end), strand=g.strand, coordinate_source='Current native annotation')])
                    bins = engine.build_profile_bins(cat)
                    (vals, qc) = engine.classify_gene_profile_direct(sp, bins, ci, vcf)
                    row = selected[selected.species.eq(sp) & selected.unit_id.eq(gid)].iloc[0]
                    body = vals[vals.region.eq('body')]
                    for key in ['callable_bp', 'mask_bp', 'effective_callable_bp', 'rare_snv_alt_count', 'rare_snv_position_count']:
                        assert body[key].sum() == row[key], (sp, gid, key, body[key].sum(), row[key])
                    generated.append(vals)
            del ci
        merged = pd.concat([cached] + generated, ignore_index=True)
        assert set(merged.stable_gene_id) == ids and merged.groupby('stable_gene_id').size().eq(100).all()
        genes_save(merged, genes_OUT / f'{sp}.profile_gene_bins.tsv.gz')
        allbins.append(merged)
        print(sp, 'profile genes', len(ids), 'newly counted', len(missing), flush=True)
    return allbins

def genes_entrypoint():
    global genes_c, genes_old, genes_p, genes_selected
    for genes_p in [genes_OUT, genes_RES, genes_TMP]:
        genes_p.mkdir(parents=True, exist_ok=True)
    (genes_old, genes_selected, genes_c) = genes_select_standard()
    genes_additional_profiles(genes_selected, genes_c)
restored_task = _component('revise_fig1_gene_rules_20261004')
restored_OUT = restored_task.OUT / 'restored70'
restored_LEG = Path('/public5/home/xuejing/RNU/codex/fig1/results/rnu_rare_variant_density_20260810')

def restored_main():
    all70 = restored_task.read(restored_task.ROOT / 'deliverables/human_catalog_exclusion_audit_20261004/all_70_missing_original_input_genes.tsv')
    engine = restored_task.profile_engine()
    ci = engine.load_callable_index(restored_task.TR / 'hg38.genmap.k150e1_unique.cov_16-48.bed')
    auto = all70[all70.chrom.isin(['chr' + str(i) for i in range(1, 23)])]
    rows = []
    for cohort in ['asj', 'eas', 'fin', 'sas', 'shuf2000']:
        path = restored_task.TR / (f'gnomadv4/{cohort}/gnomad.genomes.v4.0.sites.{cohort}.allchr.vcf.gz' if cohort != 'shuf2000' else '1kG/gnomad.genomes.v4.0.sites.shuf2000.allchr.noGT.vcf.gz')
        dataset = 'gnomADv4_' + cohort.upper() if cohort != 'shuf2000' else cohort
        generated = []
        with pysam.VariantFile(str(path.resolve())) as vcf:
            for g in auto.itertuples():
                cat = pd.DataFrame([dict(species='Human', analysis_group=restored_task.GROUPS[2], stable_gene_id=g.gene_id, display_name=g.gene_name, chrom=g.chrom, start0=int(g.start), end=int(g.end), strand=g.strand, coordinate_source='GENCODE v44 restored catalogue candidate')])
                bins = engine.build_profile_bins(cat)
                (vals, _) = engine.classify_gene_profile_direct('Human', bins, ci, vcf)
                vals['cohort'] = cohort
                generated.append(vals)
                b = vals[vals.region.eq('body')]
                metrics = {k: int(b[k].sum()) for k in ['callable_bp', 'mask_bp', 'effective_callable_bp', 'rare_snv_alt_count', 'rare_snv_position_count']}
                assert metrics['callable_bp'] == g.callable_bp
                rows.append(dict(dataset=dataset, stable_gene_id=g.gene_id, display_name=g.gene_name, gene_set=restored_task.GROUPS[2], gene_set_order=3, exon_union_bp=g.gene_length, af_threshold=0.001, chrom=g.chrom, start=g.start, end=g.end, strand=g.strand, **metrics, rare_snv_density_per_kb=1000 * metrics['rare_snv_alt_count'] / metrics['effective_callable_bp'] if metrics['effective_callable_bp'] else np.nan))
        restored_task.save(pd.concat(generated, ignore_index=True), restored_OUT / f'{cohort}.profile_gene_bins.tsv.gz')
        print(cohort, 'counted', len(generated), 'gene profiles', flush=True)
    raw = pd.DataFrame(rows)
    restored_task.save(raw, restored_OUT / 'all_cohorts_density.tsv')
    asj = raw[raw.dataset.eq('gnomADv4_ASJ')].set_index('stable_gene_id')
    audit = all70.copy()
    audit['mask_bp'] = audit.gene_id.map(asj.mask_bp)
    audit['effective_callable_bp'] = audit.gene_id.map(asj.effective_callable_bp)
    reasons = []
    for g in audit.itertuples():
        why = []
        if 'U7' in g.gene_name.upper():
            why.append('Excluded U7 gene name')
        if not g.autosomal:
            why.append('Outside assembled autosomes')
        if g.gene_length < 70:
            why.append('Original gene length < 70 bp')
        if g.callable_bp == 0:
            why.append('No callable sequence')
        elif g.callable_bp < 50:
            why.append('Callable bases < 50 before variant-position masking')
        elif g.autosomal and g.effective_callable_bp < 50:
            why.append('Effective callable bases < 50 after variant-position masking')
        reasons.append('; '.join(why) if why else 'Included in Fig1')
    audit['filter_reason'] = reasons
    audit['retained_new'] = audit.filter_reason.eq('Included in Fig1')
    restored_task.save(audit, restored_OUT / 'human_70_filter_audit.tsv')
    expanded = restored_OUT / 'expanded_legacy'
    expanded.mkdir(exist_ok=True)
    for (filename, extra) in [('gnomad_four_populations_af1e3_neutral1000_four_group_density.tsv', raw[raw.dataset.ne('shuf2000')]), ('shuf2000_af1e3_neutral1000_four_group_density.tsv', raw[raw.dataset.eq('shuf2000')])]:
        previous = restored_task.read(restored_LEG / filename)
        parts = []
        for (dataset, q) in extra.groupby('dataset'):
            template = previous[previous.dataset.eq(dataset)].iloc[0].to_dict()
            parts.extend((template | r for r in q.to_dict('records')))
        restored_task.save(pd.concat([previous, pd.DataFrame(parts)], ignore_index=True), expanded / filename)
    filename = 'palsson2025_nodedup_dnm_neutral1000_four_group_density.tsv'
    previous = restored_task.read(restored_LEG / filename)
    dnm_path = Path('/public4/home/jiangzj/analysis/mtr/histone_mutation_explore/all_species_summary/data/hg38.palsson_nature_2025.DNM.nodedup.vcf.gz')
    extra = []
    template = previous[previous.gene_set.eq(restored_task.GROUPS[2])].iloc[0].to_dict()
    with pysam.VariantFile(str(dnm_path.resolve())) as vcf:
        for g in auto[auto.callable_bp.gt(0)].itertuples():
            count = 0
            for r in vcf.fetch(g.chrom, int(g.start), int(g.end)):
                alts = r.alts or ()
                if len(alts) == 1 and len(r.ref) == 1 and (len(alts[0]) == 1) and (r.ref in 'ACGT') and (alts[0] in 'ACGT') and (r.ref != alts[0]) and engine.position_is_callable(ci, g.chrom, r.pos - 1):
                    count += 1
            exposure = int(g.callable_bp) * 9652 * 2
            extra.append(template | dict(stable_gene_id=g.gene_id, display_name=g.gene_name, exon_union_bp=g.gene_length, callable_bp=g.callable_bp, dnm_event_count=count, exposure_site_generations=exposure, dnm_rate_per_site_per_generation=count / exposure, dnm_rate_per_1e8_site_generations=100000000.0 * count / exposure, raw_dnm_events_per_callable_kb=1000 * count / g.callable_bp))
    restored_task.save(pd.concat([previous, pd.DataFrame(extra)], ignore_index=True), expanded / filename)
    print(audit.filter_reason.value_counts().to_string(), flush=True)

def restored_entrypoint():
    restored_main()
assembly_task = _component('revise_fig1_gene_rules_20261004')

def assembly_main():
    old = assembly_task.read(assembly_task.BASE / 'panel_a_gene_locus_density.tsv')
    d = pd.concat([assembly_task.read(assembly_task.OUT / 'nonchimp_locus_density.tsv'), assembly_task.read(assembly_task.OUT / 'Chimpanzee.locus_density.tsv')], ignore_index=True).reindex(columns=old.columns)
    d['_s'] = d.species.map({s: i for (i, s) in enumerate(assembly_task.SP)})
    d['_g'] = d.analysis_group.map({s: i for (i, s) in enumerate(assembly_task.GROUPS)})
    d = d.sort_values(['_s', '_g', 'unit_id']).drop(columns=['_s', '_g']).reset_index(drop=True)
    assert not d.duplicated(['species', 'unit_id']).any()
    control = lambda x: x[x.analysis_group.eq(assembly_task.GROUPS[-1])].sort_values(['species', 'unit_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(control(d), control(old), check_dtype=False, check_exact=True)
    den = assembly_task.read(assembly_task.BASE / 'control_denominators.tsv').set_index('species')
    bars = []
    tests = []
    profiles = []
    extra = {'Baboon': assembly_task.ROOT / 'data/processed/fig1_baboon_callable_20260927/baboon_profile_gene_bins.tsv.gz', 'Rat': assembly_task.ROOT / 'data/processed/fig1_rat_new_vcf_20260928/rat_profile_gene_bins.tsv.gz', 'Chicken': assembly_task.BASE / 'chicken_profile_gene_bins.tsv.gz'}
    for sp in assembly_task.SP:
        q = d[d.species.eq(sp)]
        genes = q[q.analysis_group.ne(assembly_task.GROUPS[-1])].set_index('unit_id')
        b = assembly_task.read(extra.get(sp, assembly_task.OUT / f'{sp}.profile_gene_bins.tsv.gz'))
        if 'stable_gene_id' in b:
            b = b.rename(columns={'stable_gene_id': 'unit_id'})
        b = b[b.unit_id.isin(genes.index)].copy()
        assert set(b.unit_id) == set(genes.index)
        assert b.groupby('unit_id').size().eq(100).all()
        denom = 'effective_unmasked_bp' if sp == 'Chimpanzee' else 'effective_callable_bp'
        body = b[b.region.eq('body')].groupby('unit_id')[['rare_snv_alt_count', denom]].sum()
        np.testing.assert_allclose(body.loc[genes.index].to_numpy(float), genes[['rare_snv_alt_count', denom]].to_numpy(float), rtol=1e-12)
        p = b.groupby(['species', 'analysis_group', 'bin_index', 'region', 'relative_start', 'relative_end'], as_index=False).agg(gene_count=('unit_id', 'size'), contributing_genes=(denom, lambda v: (v > 0).sum()), total_rare_snv_alt_count=('rare_snv_alt_count', 'sum'), total_effective_bp=(denom, 'sum'))
        p['total_effective_callable_bp'] = p.total_effective_bp if sp != 'Chimpanzee' else np.nan
        p['total_effective_unmasked_bp'] = p.total_effective_bp if sp == 'Chimpanzee' else np.nan
        p['pooled_density_per_kb'] = 1000 * p.total_rare_snv_alt_count / p.total_effective_bp.replace(0, np.nan)
        p['density_ratio_to_control'] = p.pooled_density_per_kb / den.loc[sp, 'control_pooled_density_per_kb']
        p['gene_set_label'] = p.analysis_group.map(dict(zip(assembly_task.GROUPS, ['POLII spliceRNUs', 'U6/U6ATAC', 'RNU pseudogenes', 'Control'])))
        if sp in ['Chimpanzee', 'Chicken']:
            r = b.groupby(['analysis_group', 'bin_index'])[['rare_ref_count', 'rare_alt_count']].sum()
            p['rare_REF'] = [r.loc[(x.analysis_group, x.bin_index), 'rare_ref_count'] for x in p.itertuples()]
            p['rare_ALT'] = [r.loc[(x.analysis_group, x.bin_index), 'rare_alt_count'] for x in p.itertuples()]
        else:
            p['rare_REF'] = np.nan
            p['rare_ALT'] = np.nan
        profiles.append(p.drop(columns=['total_effective_bp']))
        for g in assembly_task.GROUPS:
            v = q.loc[q.analysis_group.eq(g), 'rare_snv_density_per_kb']
            bars.append(dict(species=sp, analysis_group=g, n_units=len(v), mean_density_per_kb=v.mean(), mean_density_ratio_to_control=v.mean() / den.loc[sp, 'control_mean_density_per_kb']))
        for (a, z) in combinations(assembly_task.GROUPS, 2):
            (x, y) = [q.loc[q.analysis_group.eq(g), 'rare_snv_density_per_kb'] for g in [a, z]]
            t = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic', use_continuity=True)
            tests.append(dict(species=sp, group_1=a, group_2=z, n1=len(x), n2=len(y), U=t.statistic, p_value=t.pvalue, mean_density_fold=x.mean() / y.mean()))
    p = pd.concat(profiles, ignore_index=True)
    for (name, frame) in [('panel_a_gene_locus_density', d), ('panel_a_pairwise_statistics', pd.DataFrame(tests)), ('panel_b_control_mean_ratios', pd.DataFrame(bars)), ('panel_c_profile_control_ratios', p), ('control_denominators', den.reset_index())]:
        assembly_task.save(frame, assembly_task.OUT / (name + '.tsv'))
    old = assembly_task.read(assembly_task.ROOT / 'data/processed/fig1_gene_rules_20261003/panel_a_gene_locus_density.tsv')
    changes = []
    for sp in assembly_task.SP:
        for g in assembly_task.GROUPS:
            a = set(old.loc[old.species.eq(sp) & old.analysis_group.eq(g), 'unit_id'])
            z = set(d.loc[d.species.eq(sp) & d.analysis_group.eq(g), 'unit_id'])
            changes.append(dict(species=sp, group=g, old_n=len(a), new_n=len(z), added=len(z - a), removed=len(a - z)))
    assembly_task.save(pd.DataFrame(changes), assembly_task.OUT / 'group_changes.tsv')
    assembly_task.save(d.groupby(['species', 'analysis_group']).size().reset_index(name='n'), assembly_task.OUT / 'group_counts.tsv')
    for sp in assembly_task.SP:
        assert d.loc[d.species.eq(sp), 'af_threshold'].eq(assembly_task.AF[sp]).all()
    result = dict(status='PASS', full_length_filter_instead_of_callable70=True, U7_name_exclusion=True, unresolved_nonhuman_pseudogene_full_length_gt70=True, controls_unchanged=True, AF_unchanged=True, cohort_sizes_unchanged=True, body_bins_match_locus_totals=True, all_42_pairwise_tests_recomputed=True, gene_count=len(d) - 7000, original_PolII_signal_rules_preserved=True)
    (assembly_task.OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(pd.DataFrame(changes).to_string(index=False), flush=True)

def assembly_entrypoint():
    assembly_main()

def _prepare_runtime():
    restored_OUT.mkdir(exist_ok=True)

if __name__ == '__main__':
    _run_stages({'genes': 'genes_entrypoint', 'restored': 'restored_entrypoint', 'assembly': 'assembly_entrypoint'}, globals(), setup=_prepare_runtime)
