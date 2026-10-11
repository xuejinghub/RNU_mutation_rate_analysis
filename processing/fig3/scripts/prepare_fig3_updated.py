"""Consolidated final processing: prepare_fig3_updated.

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
import json
import hashlib
import itertools
import shutil
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, fisher_exact
from scipy.stats import wilcoxon, norm, fisher_exact
import os
import sys
import subprocess
import pyBigWig
from scipy.stats import mannwhitneyu, wilcoxon, fisher_exact, norm
from itertools import combinations
import io
from scipy.stats import norm, fisher_exact
groups_ROOT = (_PROCESSING_ROOT / 'fig3/scripts/plot_fig3_three_RNU_groups_20260922.py').parents[1]
groups_SOURCE = groups_ROOT / 'final_result/fig3/fig1_pseudogenes_20260920/vector'
groups_OUT = groups_ROOT / 'final_result/fig3/three_RNU_groups_20260922'
groups_DATA = groups_OUT / 'data'
groups_GROUPS = ['POL2_spliceRNUs_strict41', 'functional_U6_U6ATAC_116', 'RNU_pseudogene_operational_complement']
groups_NAMES = ['POLII spliceRNUs', 'U6/U6ATAC', 'PseudoRNUs']
groups_OVERLAP_GROUPS = ['pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes']
groups_read = _partial(_shared.read_table, low_memory=False)
groups_digest = _shared.sha256_file

def groups_prepare_data():
    groups_DATA.mkdir(parents=True, exist_ok=True)
    (tables, manifest, eligibility) = ({}, [], [])
    for name in ['a_gene_values', 'a_pairwise', 'b_profile_CDS', 'c_gene_values', 'c_pairwise', 'd_profile_CDS', 'e_gene_values', 'e_paired', 'f_gene_values', 'f_summary', 'f_comparisons', 'cd_normalization', 'fig1_pseudogene_whitelist']:
        path = groups_SOURCE / 'data' / f'{name}.tsv'
        original = groups_read(path)
        selected = original.copy()
        if 'gene_set' in selected:
            selected = selected[selected.gene_set.isin(groups_GROUPS)].copy()
        if 'left_gene_set' in selected:
            selected = selected[selected.left_gene_set.isin(groups_GROUPS) & selected.right_gene_set.isin(groups_GROUPS)].copy()
        selected = selected.drop(columns=['CDS_bp'], errors='ignore')
        key = name.replace('_CDS', '')
        tables[key] = selected
        selected.to_csv(groups_DATA / f'{key}.tsv', sep='\t', index=False)
        manifest.append(dict(path=str(path), sha256=groups_digest(path), rows_before=len(original), rows_after=len(selected)))
    for panel in ['a', 'c']:
        values = tables[f'{panel}_gene_values']
        rows = []
        for (g1, g2) in itertools.combinations(groups_GROUPS, 2):
            x = values.loc[values.gene_set.eq(g1), 'FE'].dropna()
            y = values.loc[values.gene_set.eq(g2), 'FE'].dropna()
            test = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic')
            rows.append(dict(left_gene_set=g1, right_gene_set=g2, left_n=len(x), right_n=len(y), left_median=x.median(), right_median=y.median(), median_fold_left_over_right=x.median() / y.median(), mannwhitney_u=test.statistic, p_two_sided_asymptotic=test.pvalue))
        checked = pd.DataFrame(rows)
        pd.testing.assert_frame_equal(checked.reset_index(drop=True), tables[f'{panel}_pairwise'].reset_index(drop=True), check_exact=False, rtol=1e-12, atol=0)
        for (g, label) in zip(groups_GROUPS, groups_NAMES):
            part = values[values.gene_set.eq(g)]
            eligibility.append(dict(panel=panel, gene_set=label, input_loci=len(part), valid_loci=int(part.FE.count()), excluded_loci=int(part.FE.isna().sum()), exclusion_reasons=json.dumps(part.loc[part.FE.isna(), 'status'].value_counts().to_dict())))
    e = tables['e_gene_values']
    for (g, label) in zip(groups_GROUPS, groups_NAMES):
        pair = e[e.gene_set.eq(g)].pivot(index='stable_gene_id', columns='condition', values='spi_seq_fold_enrichment').dropna()
        test = wilcoxon(pair.rnaseh1, pair.lipid_control, alternative='two-sided', method='approx', zero_method='wilcox')
        old = tables['e_paired'].set_index('gene_set').loc[g]
        np.testing.assert_allclose([test.pvalue, pair.rnaseh1.median() / pair.lipid_control.median()], [old.p_two_sided_asymptotic, old.median_fold_rnaseh1_over_control], rtol=1e-12)
        assert len(pair) == old.n_paired_loci
        eligibility.append(dict(panel='e', gene_set=label, input_loci=e[e.gene_set.eq(g)].stable_gene_id.nunique(), valid_loci=len(pair), excluded_loci=0, exclusion_reasons='{}'))
    for r in tables['f_comparisons'].itertuples():
        test = fisher_exact([[r.human_overlap, r.human_total - r.human_overlap], [r.mouse_overlap, r.mouse_total - r.mouse_overlap]])
        np.testing.assert_allclose(test.pvalue, r.fisher_exact_p_two_sided, rtol=1e-12)
    tables['panel_eligibility'] = pd.DataFrame(eligibility)
    tables['panel_eligibility'].to_csv(groups_DATA / 'panel_eligibility.tsv', sep='\t', index=False)
    tables['c_hidden_points'] = tables['c_gene_values'].loc[tables['c_gene_values'].FE > 200].copy()
    tables['c_hidden_points'].to_csv(groups_DATA / 'c_hidden_points.tsv', sep='\t', index=False)
    assert len(tables['c_hidden_points']) == 2
    assert tables['c_hidden_points'].display_name.tolist() == ['RNU6-7', 'RNU6-1']
    for name in ['mask_manifest.json']:
        shutil.copyfile(groups_SOURCE / name, groups_OUT / name)
    (groups_OUT / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (groups_OUT / 'numeric_QA.json').write_text(json.dumps(dict(status='PASS', source=str(groups_SOURCE), retained_per_gene_values_unchanged=True, retained_group_profiles_unchanged=True, all_retained_raw_P_values_reproduced=True, panel_f_unchanged=True, ac_pairwise_comparisons_per_panel=3, e_paired_comparisons=3, c_hidden_points=2), indent=2) + '\n')
    return tables
membership_previous = _component('prepare_callable_fig3_20260917')
membership_ROOT = (_PROCESSING_ROOT / 'fig3/final_result/fig3/prepare_fig3_fig1_pseudogenes_20260920.py').parents[2]
membership_SOURCE = membership_ROOT / 'final_result/fig3/callable_siNT_CDS_dual_20260918/vector'
membership_OUT = membership_ROOT / 'final_result/fig3/fig1_pseudogenes_20260920'
membership_DATA = membership_OUT / 'data'
membership_FIG1 = Path('/public5/home/xuejing/RNU/codex/transpecies_compare/final_result/fig1/revised_ab_alpha70_20260918/panel_a_gene_locus_density.tsv')
membership_AB_MATRIX = membership_ROOT / 'final_result/fig3/callable_siNT_20260917/data/ab_gene_matrices.npz'
membership_CD_MATRIX = membership_ROOT / 'final_result/fig3/callable_mask_20260917/data/cd_gene_matrices.npz'
membership_GROUPS = membership_previous.GROUPS
membership_PSEUDO = membership_GROUPS[2]
membership_read = _partial(_shared.read_table, low_memory=False)

def membership_save(frame, name):
    frame.to_csv(membership_DATA / f'{name}.tsv', sep='\t', index=False)
membership_digest = _shared.sha256_file

def membership_pair_e(values):
    rows = []
    for group in membership_GROUPS:
        v = values[values.gene_set.eq(group)].pivot(index='stable_gene_id', columns='condition', values='spi_seq_fold_enrichment').dropna()
        (x, y) = (v.lipid_control.to_numpy(), v.rnaseh1.to_numpy())
        test = wilcoxon(y, x, alternative='two-sided', zero_method='wilcox', correction=False, method='approx')
        lp = (np.log(2) + norm.logsf(abs(test.zstatistic))) / np.log(10)
        rows.append(dict(gene_set=group, n_paired_loci=len(v), control_median=np.median(x), rnaseh1_median=np.median(y), median_fold_rnaseh1_over_control=np.median(y) / np.median(x), p_two_sided_asymptotic=10 ** lp, log10_p_two_sided_asymptotic=lp))
    return pd.DataFrame(rows)

def membership_overlap_summary(values):
    rows = []
    tests = []
    for group in ['pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes']:
        counts = {}
        for species in ['Human', 'Mouse']:
            d = values[values.analysis_group.eq(group) & values.species.eq(species) & values.eligible]
            k = int(d.rloop_overlap.sum())
            n = len(d)
            p = k / n
            z = norm.ppf(0.975)
            den = 1 + z * z / n
            center = (p + z * z / 2 / n) / den
            half = z * np.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / den
            counts[species] = (k, n)
            rows.append(dict(analysis_group=group, species=species, overlap_genes=k, total_genes=n, percent=100 * p, ci95_low_percent=100 * (center - half), ci95_high_percent=100 * (center + half)))
        (h, hn) = counts['Human']
        (m, mn) = counts['Mouse']
        (od, p) = fisher_exact([[h, hn - h], [m, mn - m]], alternative='two-sided')
        tests.append(dict(analysis_group=group, human_overlap=h, human_total=hn, mouse_overlap=m, mouse_total=mn, odds_ratio_human_vs_mouse=od, fisher_exact_p_two_sided=p))
    return (pd.DataFrame(rows), pd.DataFrame(tests))

def membership_rebuild_profile(gene_values, name, matrix_path):
    old = membership_read(membership_SOURCE / 'data' / f'{name}.tsv')
    full = membership_read(membership_SOURCE / 'data' / ('a_gene_values.tsv' if name.startswith('b') else 'c_gene_values.tsv'))
    index = pd.Index(np.load(matrix_path)['gene_ids'])
    with np.load(matrix_path) as cache:
        for (frame, check_old) in [(full, True), (gene_values, False)]:
            sub = frame[frame.gene_set.eq(membership_PSEUDO) & frame.FE.notna()].copy()
            positions = index.get_indexer(sub.stable_gene_id)
            assert (positions >= 0).all() and (not sub.stable_gene_id.duplicated().any())
            matrix = cache['FE'][positions]
            prof = membership_previous.core.profile_summary(sub, matrix)
            prof = prof[prof.gene_set.eq(membership_PSEUDO)].reset_index(drop=True)
            if check_old:
                expected = old[old.gene_set.eq(membership_PSEUDO)].reset_index(drop=True)
                np.testing.assert_allclose(prof.mean_fold_enrichment, expected.mean_fold_enrichment, rtol=1e-12, equal_nan=True)
                np.testing.assert_array_equal(prof.n_genes, expected.n_genes)
            else:
                if name.startswith('b'):
                    den = cache['effective_bp'][positions, 40:60]
                    body = membership_previous.core.ratio(np.nansum(matrix[:, 40:60] * den, axis=1), den.sum(axis=1))
                else:
                    den = cache['callable_bp'][positions, 40:60]
                    body = membership_previous.core.ratio(np.nansum(cache['ip'][positions, 40:60] * den, axis=1), np.nansum(cache['input'][positions, 40:60] * den, axis=1))
                np.testing.assert_allclose(body, sub.FE, rtol=1e-07, atol=1e-07, equal_nan=True)
                np.savez_compressed(membership_DATA / f'{name}_pseudogene_matrices.npz', gene_ids=sub.stable_gene_id.to_numpy(str), FE=matrix)
    result = pd.concat([old[old.gene_set.ne(membership_PSEUDO)], prof], ignore_index=True)
    order = {g: i for (i, g) in enumerate(membership_GROUPS)}
    result = result.assign(_order=result.gene_set.map(order)).sort_values(['_order', 'bin_index']).drop(columns='_order')
    membership_save(result, name)
    return float(np.nanmax(np.abs(prof.mean_fold_enrichment.to_numpy() - old.loc[old.gene_set.eq(membership_PSEUDO), 'mean_fold_enrichment'].to_numpy())))

def membership_main():
    membership_DATA.mkdir(parents=True, exist_ok=True)
    f1 = membership_read(membership_FIG1)
    wl = f1[f1.analysis_group.eq('RNU_pseudogenes') & f1.species.isin(['Human', 'Mouse'])].copy()
    wl['stable_gene_id'] = wl.unit_id.astype(str).str.split('.').str[0]
    assert not wl.duplicated(['species', 'stable_gene_id']).any()
    keep = {species: set(wl.loc[wl.species.eq(species), 'stable_gene_id']) for species in ['Human', 'Mouse']}
    assert len(keep['Human']) == 1323 and len(keep['Mouse']) == 1007
    membership_save(wl, 'fig1_pseudogene_whitelist')
    for path in (membership_SOURCE / 'data').glob('*.tsv'):
        if path.name not in ['scatter_audit.tsv', 'c_hidden_points.tsv']:
            shutil.copyfile(path, membership_DATA / path.name)
    for name in ['mask_manifest.json', 'annotation_manifest.json']:
        shutil.copyfile(membership_SOURCE / name, membership_OUT / name)
    counts = []
    removed = []
    values = {}
    summary = []
    for panel in ['a', 'c', 'e', 'f']:
        d = membership_read(membership_SOURCE / 'data' / f'{panel}_gene_values.tsv')
        column = 'analysis_group' if panel == 'f' else 'gene_set'
        is_pseudo = d[column].eq('RNU_pseudogenes' if panel == 'f' else membership_PSEUDO)
        member = np.array([r.stable_gene_id in keep[r.species] for r in d.itertuples()]) if panel == 'f' else d.stable_gene_id.isin(keep['Human'])
        good = ~is_pseudo | member
        selected = d[good].copy()
        values[panel] = selected
        membership_save(selected, f'{panel}_gene_values')
        bad = d[~good].copy()
        bad.insert(0, 'panel', panel)
        bad['exclusion_reason'] = 'not_in_Fig1_pseudogene_set'
        removed.append(bad)
        pd.testing.assert_frame_equal(selected.reset_index(drop=True), d.loc[good].reset_index(drop=True))
        for species in ['Human', 'Mouse'] if panel == 'f' else ['Human']:
            before = d[is_pseudo & (d.species.eq(species) if panel == 'f' else True)]
            after = selected[selected[column].eq('RNU_pseudogenes' if panel == 'f' else membership_PSEUDO)]
            if panel == 'f':
                after = after[after.species.eq(species)]
            if panel in ['a', 'c']:
                nold = int(before.FE.count())
                nnew = int(after.FE.count())
            elif panel == 'e':
                nold = len(before.pivot(index='stable_gene_id', columns='condition', values='spi_seq_fold_enrichment').dropna())
                nnew = len(after.pivot(index='stable_gene_id', columns='condition', values='spi_seq_fold_enrichment').dropna())
            else:
                nold = int(before.eligible.sum())
                nnew = int(after.eligible.sum())
            counts.append(dict(panel={'a': 'a/b', 'c': 'c/d'}.get(panel, panel), species=species, fig1_n=len(keep[species]), old_input_n=before.stable_gene_id.nunique(), new_input_n=after.stable_gene_id.nunique(), removed_not_in_fig1=before.stable_gene_id.nunique() - after.stable_gene_id.nunique(), old_valid_n=nold, new_valid_n=nnew, unavailable_after_filter=after.stable_gene_id.nunique() - nnew))
        if panel in ['a', 'c']:
            membership_save(membership_previous.core.pairwise(selected, 'FE'), f'{panel}_pairwise')
            for group in membership_GROUPS:
                for (tag, table) in [('before', d), ('after', selected)]:
                    x = table.loc[table.gene_set.eq(group), 'FE'].dropna()
                    summary.append(dict(panel=panel, version=tag, gene_set=group, n=len(x), median=x.median(), mean=x.mean()))
    membership_save(membership_pair_e(values['e']), 'e_paired')
    (fs, ft) = membership_overlap_summary(values['f'])
    membership_save(fs, 'f_summary')
    membership_save(ft, 'f_comparisons')
    pd.testing.assert_frame_equal(values['f'].reset_index(drop=True), membership_read(membership_SOURCE / 'data/f_gene_values.tsv'))
    pd.testing.assert_frame_equal(fs, membership_read(membership_SOURCE / 'data/f_summary.tsv'), check_exact=False, rtol=1e-12)
    pd.testing.assert_frame_equal(ft, membership_read(membership_SOURCE / 'data/f_comparisons.tsv'), check_exact=False, rtol=1e-12)
    delta = {name: membership_rebuild_profile(values[panel], name, path) for (panel, name, path) in [('a', 'b_profile_CDS', membership_AB_MATRIX), ('c', 'd_profile_CDS', membership_CD_MATRIX)]}
    membership_save(pd.DataFrame(counts), 'pseudogene_filter_counts')
    membership_save(pd.concat(removed, ignore_index=True), 'pseudogenes_removed')
    membership_save(pd.DataFrame(summary), 'group_summary_before_after')
    changed = []
    for name in ['a_pairwise', 'c_pairwise', 'e_paired', 'f_comparisons']:
        for (version, path) in [('before', membership_SOURCE / 'data' / f'{name}.tsv'), ('after', membership_DATA / f'{name}.tsv')]:
            tab = membership_read(path)
            tab.insert(0, 'version', version)
            tab.insert(0, 'table', name)
            changed.append(tab)
    membership_save(pd.concat(changed, ignore_index=True), 'statistics_before_after')
    sources = [membership_FIG1, membership_AB_MATRIX, membership_CD_MATRIX] + list((membership_SOURCE / 'data').glob('*.tsv'))
    manifest = [dict(path=str(p), sha256=membership_digest(p), bytes=p.stat().st_size) for p in sources]
    (membership_OUT / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    qa = dict(status='PASS', fig1_whitelist_source=str(membership_FIG1), human_whitelist_n=1323, mouse_whitelist_n=1007, membership_rule='intersection by stable gene ID, species-specific for f', per_gene_values_unchanged=True, non_pseudogene_cohorts_unchanged=True, normalization_and_masks_unchanged=True, full_old_pseudogene_profiles_reproduced=True, new_profile_body_reconstruction=True, all_pairwise_and_paired_tests_recomputed=True, f_numerically_unchanged=True, profile_max_abs_change=delta, counts=counts)
    (membership_OUT / 'numeric_QA.json').write_text(json.dumps(qa, indent=2) + '\n')
    print(pd.DataFrame(counts).to_string(index=False), flush=True)
    for name in ['a_pairwise', 'c_pairwise', 'e_paired']:
        print(name, membership_read(membership_DATA / f'{name}.tsv').to_string(index=False), flush=True)

def membership_entrypoint():
    membership_main()
human_ROOT = (_PROCESSING_ROOT / 'fig3/scripts/update_fig3_human_pseudogenes_20261005.py').parents[1]
human_extraction = _component('prepare_callable_fig3_20260917')
human_plot = _component('plot_fig3_three_RNU_groups_20260922')
human_BASE = human_ROOT / 'final_result/fig3/three_RNU_groups_20260922'
human_OUT = human_ROOT / 'final_result/fig3/human_pseudogenes_20261005'
human_DATA = human_OUT / 'data'
human_TMP = human_ROOT / 'tmp/fig3_human_pseudogenes_20261005'
human_NEW = human_ROOT / 'fig1_gene_rules_20261004/gene_sets/Human/gene_sets.tsv'
human_GROUPS = human_plot.GROUPS
human_PSEUDO = human_GROUPS[2]
human_core = human_extraction.core
human_AB_OLD = human_ROOT / 'final_result/fig3/callable_siNT_20260917/data/ab_gene_matrices.npz'
human_CD_OLD = human_ROOT / 'final_result/fig3/callable_mask_20260917/data/cd_gene_matrices.npz'
human_read = _partial(_shared.read_table, low_memory=False)

def human_save(d, n):
    d.to_csv(human_DATA / f'{n}.tsv', sep='\t', index=False)
human_sha = _shared.sha256_file

def human_pairwise(d):
    rows = []
    for (a, b) in combinations(human_GROUPS, 2):
        x = d.loc[d.gene_set.eq(a), 'FE'].dropna()
        y = d.loc[d.gene_set.eq(b), 'FE'].dropna()
        test = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic')
        rows.append(dict(left_gene_set=a, right_gene_set=b, left_n=len(x), right_n=len(y), left_median=x.median(), right_median=y.median(), median_fold_left_over_right=x.median() / y.median(), mannwhitney_u=test.statistic, p_two_sided_asymptotic=test.pvalue))
    return pd.DataFrame(rows)

def human_map_genes(new, old):
    bed = human_TMP / 'pseudogenes_hg38.bed'
    mapped = human_TMP / 'pseudogenes_hg19.bed'
    unmapped = human_TMP / 'unmapped.bed'
    b = new[['chrom', 'start', 'end', 'gene_id']].copy()
    b['score'] = 0
    b['strand'] = new.strand
    b.to_csv(bed, sep='\t', header=False, index=False)
    cmd = ['/public/home/xuejing/mambaforge/bin/liftOver', '-minMatch=0.95', str(bed), str(human_extraction.CHAIN), str(mapped), str(unmapped)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    (human_OUT / 'liftover_command.txt').write_text(' '.join(cmd) + '\n' + result.stdout + result.stderr)
    m = pd.read_csv(mapped, sep='\t', header=None, names=['chrom_hg19', 'start_hg19', 'end_hg19', 'gene_id', 'score', 'strand_hg19'])
    assert not m.gene_id.duplicated().any()
    audit = new.merge(m, on='gene_id', how='left', validate='one_to_one')
    audit['mapped_hg19'] = audit.start_hg19.notna()
    audit['previous_fig3_member'] = audit.gene_id.isin(old.stable_gene_id)
    shared = audit.merge(old, left_on='gene_id', right_on='stable_gene_id', suffixes=('_new', '_old'))
    for c in ['chrom_hg19', 'start_hg19', 'end_hg19']:
        assert (shared[c + '_new'] == shared[c + '_old']).all(), c
    flip = set(shared.loc[shared.strand_hg19 != shared.strand_old, 'gene_id'])
    audit['hg19_strand_correction'] = audit.gene_id.isin(flip)
    human_save(audit, 'pseudogene_coordinate_audit')
    shutil.copyfile(unmapped, human_DATA / 'pseudogenes_unmapped_hg19.bed')
    return audit

def human_extract_additions(mapped):
    k19 = human_extraction.KeepMask.load('hg19')
    k38 = human_extraction.KeepMask.load('hg38')
    mask19 = human_extraction.ExcludeOutside(k19)
    mask38 = human_extraction.ExcludeOutside(k38)
    norms = human_read(human_BASE / 'data/cd_normalization.tsv').set_index('sample').normalization_factor
    ab = [pyBigWig.open(str(p)) for p in human_core.AB]
    ev = [pyBigWig.open(str(p)) for p in human_core.E]
    cd = {f'SRX267500{i}': pyBigWig.open(str(human_core.AUDIT / f'SRX267500{i}_hg38.bw')) for i in range(3, 9)}
    ar = []
    cr = []
    er = []
    ma = {}
    mc = {}
    checks = []
    for r in mapped.itertuples():
        if not r.mapped_hg19:
            continue
        (s, e) = (int(r.start_hg19), int(r.end_hg19))
        edges = human_core.edges_for(s, e)
        (sm, ct, total, n, geom) = human_core.integrate_tracks(ab, r.chrom_hg19, edges, mask19, positive=True)
        den = n[40:60].sum()
        v = total[40:60].sum() / den if den else np.nan
        row = dict(stable_gene_id=r.gene_id, display_name=r.gene_name, gene_set=human_PSEUDO, chrom_hg19=r.chrom_hg19, start_hg19=s, end_hg19=e, strand=r.strand_hg19, gene_length_hg19_bp=e - s, gene_set_label='RNU pseudogenes', callable_bp=geom[40:60].sum(), excluded_bp=e - s - geom[40:60].sum(), signal_sum=total[40:60].sum(), effective_signal_bp=den, FE=v, status='ok' if den else 'no_callable_bases' if not geom[40:60].sum() else 'no_positive_finite_signal', region_definition='RNU gene body')
        reps = human_core.ratio(sm, ct)
        for i in range(2):
            row[f'rep{i + 1}_signal_sum'] = sm[i, 40:60].sum()
            row[f'rep{i + 1}_effective_bp'] = ct[i, 40:60].sum()
            row[f'rep{i + 1}_FE'] = sm[i, 40:60].sum() / ct[i, 40:60].sum() if ct[i, 40:60].sum() else np.nan
        ar.append(row)
        vals = np.array([bw.values(r.chrom_hg19, s, e, numpy=True) for bw in ab], float)
        keep = np.zeros(e - s, bool)
        for (u, w) in k19.overlaps(r.chrom_hg19, s, e):
            keep[u - s:w - s] = True
        vals[:, ~keep] = np.nan
        vals[vals <= 0] = np.nan
        cnt = np.isfinite(vals).sum(axis=0)
        avg = human_core.ratio(np.nansum(vals, axis=0), cnt)
        direct = avg[cnt > 0].mean() if (cnt > 0).any() else np.nan
        np.testing.assert_allclose(v, direct, rtol=1e-07, atol=1e-07, equal_nan=True)
        order = slice(None, None, -1) if r.strand_hg19 == '-' else slice(None)
        ma[r.gene_id] = dict(FE=human_core.ratio(total, n)[order], effective_bp=n[order], replicate_FE=reps[:, order])
        for (cond, bw) in zip(['lipid_control', 'rnaseh1'], ev):
            (ss, cc, _, _, gg) = human_core.integrate_tracks([bw], r.chrom_hg19, np.array([s, e]), mask19)
            er.append(dict(stable_gene_id=r.gene_id, display_name=r.gene_name, gene_set=human_PSEUDO, condition=cond, chrom=r.chrom_hg19, start=s, end=e, callable_bp=gg[0], signal_sum=ss[0, 0], effective_signal_bp=cc[0, 0], spi_seq_fold_enrichment=ss[0, 0] / cc[0, 0] if cc[0, 0] else np.nan, region_definition='RNU gene body'))
        edges = human_core.edges_for(int(r.start), int(r.end), 'floor')
        ip = np.zeros(100)
        inp = np.zeros(100)
        ipbody = 0.0
        inbody = 0.0
        row = dict(stable_gene_id=r.gene_id, display_name=r.gene_name, gene_set=human_PSEUDO, chrom=r.chrom, start=r.start, end=r.end, strand=r.strand, length_bp=r.end - r.start, matched_panel_c=True, region_definition='RNU gene body')
        order = slice(None, None, -1) if r.strand == '-' else slice(None)
        for (sample, bw) in cd.items():
            (ss, cc, _, _, gg) = human_core.integrate_tracks([bw], r.chrom, edges, mask38)
            factor = norms[sample]
            bodyden = gg[40:60].sum()
            raw = ss[0, 40:60].sum()
            value = raw / bodyden / factor if bodyden else np.nan
            row[sample] = value
            row[sample + '_raw_sum'] = raw
            signal = human_core.ratio(ss[0], gg) / factor
            if int(sample[-1]) % 2:
                ip += signal[order] / 3
                ipbody += value / 3
            else:
                inp += signal[order] / 3
                inbody += value / 3
            a = bw.values(r.chrom, int(r.start), int(r.end), numpy=True)
            keep = np.zeros(len(a), bool)
            for (u, w) in k38.overlaps(r.chrom, int(r.start), int(r.end)):
                keep[u - r.start:w - r.start] = True
            np.testing.assert_allclose(np.nansum(a[keep]), raw, rtol=1e-07, atol=1e-07)
        row.update(callable_bp=bodyden, excluded_bp=r.end - r.start - bodyden, normalized_ip_mean=ipbody, normalized_input_mean=inbody, FE=ipbody / inbody if inbody > 0 else np.nan, status='ok' if inbody > 0 else 'no_callable_bases' if not bodyden else 'zero_input')
        cr.append(row)
        mc[r.gene_id] = dict(ip=ip, input=inp, FE=human_core.ratio(ip, inp), callable_bp=gg[order])
        checks.append(dict(gene_id=r.gene_id, a_per_base_check='PASS', c_six_track_integrals='PASS'))
    for bw in ab + ev + list(cd.values()):
        bw.close()
    human_save(pd.DataFrame(checks), 'new_locus_signal_validation')
    return (pd.DataFrame(ar), pd.DataFrame(cr), pd.DataFrame(er), ma, mc)

def human_rebuild_matrices(values, cachepath, added, panel):
    with np.load(cachepath) as archive:
        cache = {key: archive[key] for key in archive.files}
        old_ids = pd.Index(cache['gene_ids'])
        keys = [k for k in cache if k != 'gene_ids']
        parts = {k: [] for k in keys}
        for gene_id in values.stable_gene_id:
            for key in keys:
                value = added[gene_id][key] if gene_id in added else cache[key][:, old_ids.get_loc(gene_id)] if key == 'replicate_FE' else cache[key][old_ids.get_loc(gene_id)]
                parts[key].append(value)
        mats = {k: np.stack(v, axis=1 if k == 'replicate_FE' else 0) for (k, v) in parts.items()}
    np.savez_compressed(human_DATA / f'{panel}_gene_matrices.npz', gene_ids=values.stable_gene_id.to_numpy(str), **mats)
    rows = []
    for group in human_GROUPS:
        good = values.gene_set.eq(group) & values.FE.notna()
        a = mats['FE'][good]
        for i in range(100):
            x = -2000 + (i + 0.5) * 50 if i < 40 else (i - 40 + 0.5) * 50 if i < 60 else 1000 + (i - 60 + 0.5) * 50
            valid = a[:, i][np.isfinite(a[:, i])]
            rows.append(dict(gene_set=group, bin_index=i, profile_x_bp=x, display_x=x / 2 if x < 0 else x if x <= 1000 else 1000 + (x - 1000) / 2, mean_fold_enrichment=valid.mean() if len(valid) else np.nan, n_genes=len(valid), cohort_n=len(a)))
    profile = pd.DataFrame(rows)
    name = 'b_profile' if panel == 'ab' else 'd_profile'
    old = human_read(human_BASE / 'data' / f'{name}.tsv')
    pd.testing.assert_frame_equal(profile[profile.gene_set.ne(human_PSEUDO)].reset_index(drop=True), old[old.gene_set.ne(human_PSEUDO)].reset_index(drop=True), check_exact=False, rtol=1e-13, atol=0)
    human_save(profile, name)

def human_main():
    human_DATA.mkdir(parents=True, exist_ok=True)
    human_TMP.mkdir(parents=True, exist_ok=True)
    allnew = human_read(human_NEW)
    new = allnew[allnew.group.eq('RNU_pseudogenes')].copy()
    assert len(new) == 1362 and (not new.gene_id.duplicated().any())
    old = {p: human_read(human_BASE / 'data' / f'{p}_gene_values.tsv') for p in ['a', 'c', 'e', 'f']}
    oldpseudo = old['a'][old['a'].gene_set.eq(human_PSEUDO)]
    keep = set(new.gene_id)
    old_ids = set(oldpseudo.stable_gene_id)
    added_ids = keep - old_ids
    audit = human_map_genes(new, oldpseudo)
    shared = new.merge(old['c'], left_on='gene_id', right_on='stable_gene_id', suffixes=('_new', '_old'))
    for c in ['chrom', 'start', 'end', 'strand']:
        assert (shared[c + '_new'] == shared[c + '_old']).all()
    membership = pd.concat([new.assign(change=np.where(new.gene_id.isin(old_ids), 'retained', 'added')), pd.DataFrame({'gene_id': sorted(old_ids - keep), 'change': 'removed'})], ignore_index=True)
    human_save(membership, 'human_pseudogene_membership')
    human_save(new, 'fig1_human_pseudogene_whitelist')
    flip = set(audit.loc[audit.hg19_strand_correction, 'gene_id'])
    (ar, cr, er, ma, mc) = human_extract_additions(audit[audit.gene_id.isin(added_ids | flip)])
    for gene_id in flip:
        np.testing.assert_allclose(ar.loc[ar.stable_gene_id.eq(gene_id), 'FE'], old['a'].loc[old['a'].stable_gene_id.eq(gene_id), 'FE'], rtol=1e-12, equal_nan=True)
        mc.pop(gene_id)
    cr = cr[cr.stable_gene_id.isin(added_ids)]
    er = er[er.stable_gene_id.isin(added_ids)]
    result = {}
    for (panel, add) in [('a', ar), ('c', cr), ('e', er)]:
        prior = old[panel]
        selected = prior[(prior.gene_set.ne(human_PSEUDO) | prior.stable_gene_id.isin(keep)) & ~prior.stable_gene_id.isin(add.stable_gene_id)].copy()
        out = pd.concat([selected, add], ignore_index=True)
        order = {g: i for (i, g) in enumerate(human_GROUPS)}
        out = out.assign(_order=out.gene_set.map(order)).sort_values('_order', kind='stable').drop(columns='_order').reset_index(drop=True)
        pd.testing.assert_frame_equal(out.loc[out.gene_set.ne(human_PSEUDO), prior.columns].reset_index(drop=True), prior[prior.gene_set.ne(human_PSEUDO)].reset_index(drop=True), check_dtype=False)
        human_save(out, f'{panel}_gene_values')
        result[panel] = out
    f = old['f']
    template = f[f.species.eq('Human') & f.analysis_group.eq('RNU_pseudogenes')].iloc[0].to_dict()
    peaks = human_core.bed(template['rloop_input'])
    mask = human_extraction.KeepMask.load('hg38')
    frows = []
    for r in new.itertuples():
        allowed = mask.overlaps(r.chrom, r.start, r.end)
        bp = np.diff(allowed, axis=1).sum()
        ov = sum((np.diff(peaks.overlaps(r.chrom, s, e), axis=1).sum() for (s, e) in allowed))
        original = np.diff(peaks.overlaps(r.chrom, r.start, r.end), axis=1).sum()
        row = dict(template)
        row.update(stable_gene_id=r.gene_id, display_name=r.gene_name, chrom=r.chrom, start0=r.start, end=r.end, strand=r.strand, coordinate_source='fig1_gene_rules_20261004/gene_sets/Human/gene_sets.tsv', gene_interval_bp=r.end - r.start, overlap_bp=ov, overlap_fraction=ov / bp if bp else np.nan, rloop_overlap=ov > 0, original_overlap_bp=original, original_rloop_overlap=original > 0, callable_bp=bp, excluded_bp=r.end - r.start - bp, eligible=bp > 0)
        frows.append(row)
    fn = pd.concat([f[~(f.species.eq('Human') & f.analysis_group.eq('RNU_pseudogenes'))], pd.DataFrame(frows)], ignore_index=True)
    human_save(fn, 'f_gene_values')
    result['f'] = fn
    for panel in ['a', 'c']:
        human_save(human_pairwise(result[panel]), f'{panel}_pairwise')
    estats = []
    for g in human_GROUPS:
        pair = result['e'][result['e'].gene_set.eq(g)].pivot(index='stable_gene_id', columns='condition', values='spi_seq_fold_enrichment').dropna()
        (x, y) = (pair.lipid_control, pair.rnaseh1)
        test = wilcoxon(y, x, method='approx', alternative='two-sided', zero_method='wilcox')
        estats.append(dict(gene_set=g, n_paired_loci=len(pair), control_median=x.median(), rnaseh1_median=y.median(), median_fold_rnaseh1_over_control=y.median() / x.median(), p_two_sided_asymptotic=test.pvalue, log10_p_two_sided_asymptotic=np.log10(test.pvalue)))
    human_save(pd.DataFrame(estats), 'e_paired')
    fs = []
    ft = []
    for group in ['pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes']:
        sizes = {}
        for species in ['Human', 'Mouse']:
            part = fn[fn.analysis_group.eq(group) & fn.species.eq(species) & fn.eligible]
            n = len(part)
            k = int(part.rloop_overlap.sum())
            p = k / n
            z = norm.ppf(0.975)
            den = 1 + z * z / n
            center = (p + z * z / 2 / n) / den
            half = z * np.sqrt(p * (1 - p) / n + z * z / 4 / n ** 2) / den
            sizes[species] = (k, n)
            fs.append(dict(analysis_group=group, species=species, overlap_genes=k, total_genes=n, percent=100 * p, ci95_low_percent=100 * (center - half), ci95_high_percent=100 * (center + half)))
        (h, hn) = sizes['Human']
        (m, mn) = sizes['Mouse']
        test = fisher_exact([[h, hn - h], [m, mn - m]])
        ft.append(dict(analysis_group=group, human_overlap=h, human_total=hn, mouse_overlap=m, mouse_total=mn, odds_ratio_human_vs_mouse=test.statistic, fisher_exact_p_two_sided=test.pvalue))
    human_save(pd.DataFrame(fs), 'f_summary')
    human_save(pd.DataFrame(ft), 'f_comparisons')
    human_rebuild_matrices(result['a'], human_AB_OLD, ma, 'ab')
    human_rebuild_matrices(result['c'], human_CD_OLD, mc, 'cd')
    eligibility = []
    for panel in ['a', 'c', 'e']:
        for (g, label) in zip(human_GROUPS, human_plot.NAMES):
            d = result[panel][result[panel].gene_set.eq(g)]
            n = d.FE.count() if panel != 'e' else len(d.pivot(index='stable_gene_id', columns='condition', values='spi_seq_fold_enrichment').dropna())
            bad = d.loc[d.FE.isna(), 'status'].value_counts().to_dict() if panel != 'e' else {}
            eligibility.append(dict(panel=panel, gene_set=label, input_loci=d.stable_gene_id.nunique(), valid_loci=int(n), excluded_loci=d.stable_gene_id.nunique() - int(n), exclusion_reasons=json.dumps(bad)))
    human_save(pd.DataFrame(eligibility), 'panel_eligibility')
    human_save(result['c'][result['c'].FE > 200], 'c_hidden_points')
    shutil.copyfile(human_BASE / 'data/cd_normalization.tsv', human_DATA / 'cd_normalization.tsv')
    shutil.copyfile(human_BASE / 'mask_manifest.json', human_OUT / 'mask_manifest.json')
    qa = dict(status='PASS', human_pseudogene_catalogue_n=len(new), old_n=len(old_ids), added=len(added_ids), removed=len(old_ids - keep), mapped_hg19=int(audit.mapped_hg19.sum()), unmapped_hg19=int((~audit.mapped_hg19).sum()), added_loci_extracted=int(ar.stable_gene_id.isin(added_ids).sum()), corrected_hg19_strand_gene_ids=sorted(flip), non_pseudogene_observations_unchanged=True, mouse_observations_unchanged=True, unchanged_hg38_coordinates=True, unchanged_existing_hg19_coordinates=True, new_loci_independent_base_checks=True, normalization_and_callable_masks_unchanged=True, panel_counts=eligibility)
    (human_OUT / 'numeric_QA.json').write_text(json.dumps(qa, indent=2) + '\n')
    inputs = [human_NEW, human_AB_OLD, human_CD_OLD, human_extraction.CHAIN] + list((human_BASE / 'data').glob('*.tsv'))
    (human_OUT / 'input_manifest.json').write_text(json.dumps([dict(path=str(p), sha256=human_sha(p)) for p in inputs], indent=2) + '\n')
    print(json.dumps(qa, indent=2), flush=True)

def human_entrypoint():
    human_main()
mouse_ROOT = (_PROCESSING_ROOT / 'fig3/scripts/plot_fig3_human_mouse_pseudogenes_20261005.py').parents[1]
mouse_TMP = mouse_ROOT / 'tmp/fig3_human_mouse_pseudogenes_20261005'
mouse_plot = _component('plot_fig3_three_RNU_groups_20260922')
mouse_BASE = mouse_ROOT / 'final_result/fig3/human_pseudogenes_20261005'
mouse_OUT = mouse_ROOT / 'final_result/fig3/human_mouse_pseudogenes_20261005'
mouse_DATA = mouse_OUT / 'data'
mouse_MOUSE = mouse_ROOT / 'fig1_gene_rules_20261004/gene_sets/Mouse/gene_sets.tsv'
mouse_BEDTOOLS = '/public/home/xuejing/mambaforge/bin/bedtools'
mouse_sha = _shared.sha256_file
mouse_read = _partial(_shared.read_table, low_memory=False)

def mouse_save(d, name):
    d.to_csv(mouse_DATA / f'{name}.tsv', sep='\t', index=False)

def mouse_prepare():
    mouse_TMP.mkdir(parents=True, exist_ok=True)
    mouse_DATA.mkdir(parents=True, exist_ok=True)
    manifest = [dict(path=str(p), sha256=mouse_sha(p)) for p in sorted((mouse_BASE / 'data').iterdir()) if p.is_file()]
    for p in (mouse_BASE / 'data').iterdir():
        if p.is_file():
            shutil.copyfile(p, mouse_DATA / p.name)
    shutil.copyfile(mouse_BASE / 'mask_manifest.json', mouse_OUT / 'mask_manifest.json')
    old = mouse_read(mouse_BASE / 'data/f_gene_values.tsv')
    sel = old.species.eq('Mouse') & old.analysis_group.eq('RNU_pseudogenes')
    previous = old[sel].copy()
    new = mouse_read(mouse_MOUSE)
    new = new[new.group.eq('RNU_pseudogenes')].copy()
    assert len(new) == 1032 and (not new.gene_id.duplicated().any())
    assert new.assembly.eq('mm10').all() and (new.end > new.start).all()
    shared = new.merge(previous, left_on='gene_id', right_on='stable_gene_id', suffixes=('_new', '_old'))
    assert (shared.chrom_new == shared.chrom_old).all() and (shared.start == shared.start0).all()
    assert (shared.end_new == shared.end_old).all() and (shared.strand_new == shared.strand_old).all()
    added = set(new.gene_id) - set(previous.stable_gene_id)
    removed = set(previous.stable_gene_id) - set(new.gene_id)
    assert len(added) == 25 and (not removed)
    peak_path = Path(previous.rloop_input.iloc[0])
    assert previous.rloop_input.nunique() == 1
    manifest += [dict(path=str(p), sha256=mouse_sha(p)) for p in [mouse_MOUSE, peak_path, mouse_BASE / 'mask_manifest.json']]
    b = new[['chrom', 'start', 'end', 'gene_id']].sort_values(['chrom', 'start', 'end'])
    bed = mouse_TMP / 'mouse_pseudogenes.mm10.bed'
    b.to_csv(bed, sep='\t', index=False, header=False)
    commands = []
    command = [mouse_BEDTOOLS, 'sort', '-i', str(peak_path)]
    commands.append(command)
    peak_sort = mouse_TMP / 'mouse_nine_sets.sorted.bed'
    peak_sort.write_text(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    command = [mouse_BEDTOOLS, 'coverage', '-a', str(bed), '-b', str(peak_sort), '-sorted']
    commands.append(command)
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    (mouse_TMP / 'bedtools_coverage.tsv').write_text(result.stdout)
    cov = pd.read_csv(io.StringIO(result.stdout), sep='\t', header=None, names=['chrom', 'start', 'end', 'gene_id', 'overlapping_regions', 'overlap_bp', 'length_bp', 'overlap_fraction'])
    assert len(cov) == 1032
    peaks = pd.read_csv(peak_sort, sep='\t', header=None, usecols=[0, 1, 2], names=['chrom', 'start', 'end'])
    intervals = {chrom: d[['start', 'end']].to_numpy(np.int64) for (chrom, d) in peaks.groupby('chrom')}
    for arr in intervals.values():
        assert (arr[1:, 0] >= arr[:-1, 1]).all()
    independent = []
    for r in cov.itertuples():
        arr = intervals.get(r.chrom, np.empty((0, 2), int))
        bp = int(np.maximum(0, np.minimum(arr[:, 1], r.end) - np.maximum(arr[:, 0], r.start)).sum())
        assert bp == r.overlap_bp
        independent.append(bp)
    cov['independent_overlap_bp'] = independent
    cov['independent_check'] = 'PASS'
    mouse_save(cov, 'mouse_overlap_validation')
    template = previous.iloc[0].to_dict()
    lookup = cov.set_index('gene_id')
    rows = []
    for r in new.itertuples():
        length = int(r.end - r.start)
        overlap = int(lookup.loc[r.gene_id, 'overlap_bp'])
        row = dict(template)
        row.update(stable_gene_id=r.gene_id, display_name=r.gene_name, chrom=r.chrom, start0=r.start, end=r.end, strand=r.strand, coordinate_source='fig1_gene_rules_20261004/gene_sets/Mouse/gene_sets.tsv', gene_interval_bp=length, overlap_bp=overlap, overlap_fraction=overlap / length, rloop_overlap=overlap > 0, original_overlap_bp=overlap, original_rloop_overlap=overlap > 0, callable_bp=length, excluded_bp=0, eligible=True)
        rows.append(row)
    current = pd.DataFrame(rows)
    compare = previous.set_index('stable_gene_id').sort_index()
    retained = current[current.stable_gene_id.isin(compare.index)].set_index('stable_gene_id').sort_index()
    for col in ['overlap_bp', 'rloop_overlap', 'callable_bp', 'gene_interval_bp']:
        np.testing.assert_array_equal(compare[col], retained[col])
    final = pd.concat([old[~sel], current], ignore_index=True)
    pd.testing.assert_frame_equal(old[~sel].reset_index(drop=True), final.iloc[:len(old[~sel])].reset_index(drop=True), check_dtype=False)
    mouse_save(final, 'f_gene_values')
    membership = new.copy()
    membership['change'] = np.where(membership.gene_id.isin(added), 'added', 'retained')
    membership = membership.merge(cov[['gene_id', 'overlap_bp', 'overlapping_regions']], on='gene_id', validate='one_to_one')
    mouse_save(membership, 'mouse_pseudogene_membership')
    mouse_save(new, 'fig1_mouse_pseudogene_whitelist')
    summaries = []
    tests = []
    for group in mouse_plot.OVERLAP_GROUPS:
        counts = {}
        for species in ['Human', 'Mouse']:
            part = final[final.analysis_group.eq(group) & final.species.eq(species) & final.eligible]
            n = len(part)
            k = int(part.rloop_overlap.sum())
            p = k / n
            z = norm.ppf(0.975)
            den = 1 + z * z / n
            center = (p + z * z / (2 * n)) / den
            half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            counts[species] = (k, n)
            summaries.append(dict(analysis_group=group, species=species, overlap_genes=k, total_genes=n, percent=100 * p, ci95_low_percent=100 * (center - half), ci95_high_percent=100 * (center + half)))
        (h, hn) = counts['Human']
        (m, mn) = counts['Mouse']
        test = fisher_exact([[h, hn - h], [m, mn - m]], alternative='two-sided')
        tests.append(dict(analysis_group=group, human_overlap=h, human_total=hn, mouse_overlap=m, mouse_total=mn, odds_ratio_human_vs_mouse=test.statistic, fisher_exact_p_two_sided=test.pvalue))
    mouse_save(pd.DataFrame(summaries), 'f_summary')
    mouse_save(pd.DataFrame(tests), 'f_comparisons')
    qa = dict(status='PASS', old_mouse_n=len(previous), new_mouse_n=len(new), added=len(added), removed=len(removed), old_overlap=int(previous.rloop_overlap.sum()), new_overlap=int(current.rloop_overlap.sum()), added_loci_overlapping=int((membership.change.eq('added') & membership.overlap_bp.gt(0)).sum()), all_1032_loci_independently_verified=True, retained_overlap_values_unchanged=True, human_data_unchanged=True, other_mouse_gene_sets_unchanged=True, mouse_mask_applied=False, peak_source=str(peak_path), peak_sha256=mouse_sha(peak_path), bedtools_version=subprocess.run([mouse_BEDTOOLS, '--version'], capture_output=True, text=True, check=True).stdout.strip())
    (mouse_OUT / 'numeric_QA.json').write_text(json.dumps(qa, indent=2) + '\n')
    (mouse_OUT / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (mouse_OUT / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    print(json.dumps(qa, indent=2), flush=True)
    return qa

def mouse_entrypoint():
    mouse_prepare()

def _prepare_runtime():
    sys.dont_write_bytecode = True
    os.environ.setdefault('MPLCONFIGDIR', str(human_ROOT / 'tmp/fig3_human_pseudogenes_20261005/matplotlib'))
    sys.path.insert(0, str(human_ROOT / 'final_result/fig3'))

if __name__ == '__main__':
    _run_stages({'membership': 'membership_entrypoint', 'human': 'human_entrypoint', 'mouse': 'mouse_entrypoint'}, globals(), setup=_prepare_runtime)
