"""Consolidated final processing: analyze_mutations_updated.

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
import ast
import hashlib
import json
import os
import math
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import argparse
import sys
import warnings
import shutil
constants_GROUP_ORDER = ['Pol II splice-RNUs', 'U6', 'Pseudo-RNUs']
constants_MUTATION_ORDER = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
initialization_ROOT = (_PROCESSING_ROOT / 'fig2/scripts/plot_figS11_S14_population_reproduction.py').parents[1]
initialization_base = _component('plot_figure2_abcd_b_fullrow_cd2to3_min7pt')
initialization_ENGINE = _component_file('plot_figure2_fig1_gene_sets')
initialization_SPECS = {'EAS': (11, 'EAS'), 'SAS': (12, 'SAS'), 'FIN': (13, 'FIN'), 'SHUF2000': (14, 'random2000')}
initialization_sha = _shared.sha256_file

def initialization_namespace(population):
    """Load the pinned engine's initialization and pure functions only.

    Exclude all top-level analyses, exports and preview rendering. No frozen
    source tables or previous figures are overwritten.
    """
    os.environ['FIG2_HUMAN_POPULATION'] = population
    tree = ast.parse(initialization_ENGINE.read_text())
    nodes = tree.body
    ns = {'__name__': 'frozen_analysis_engine', '__file__': str(initialization_ENGINE)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(initialization_ENGINE), 'exec'), ns)
    ns.update(SPECIES_ORDER=['Human', 'Macaque', 'Mouse'], CONTROL_ORDER=['U6', 'Pseudo-RNUs'], SHUF_SITE_CONTEXT_CACHE={}, EXPECTED_TCW={})
    previous = json.loads((initialization_ROOT / f'final_result/figS{initialization_SPECS[population][0] - 10}' / f"FigS{initialization_SPECS[population][0] - 10}_{('shuf2000' if population == 'SHUF2000' else population)}_analysis_manifest.json").read_text())
    paths = {tuple(key.split('|')): initialization_ROOT / value['path'] for (key, value) in previous['inputs']['cross_species_result_files'].items()}
    names = ['pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes']
    for (group, name) in zip(ns['GROUP_ORDER'], names):
        paths['Macaque', group] = initialization_ROOT / 'data/processed/figure2_macaque_af002_20260916/results' / f'{name}_fig1_af2e-3.result'
    ns['CROSS_SPECIES_INPUTS'] = paths
    ns['CROSS_SPECIES_SHA256'] = {key: initialization_sha(path) for (key, path) in paths.items()}
    return ns
strand_ROOT = (_PROCESSING_ROOT / 'fig2/scripts/plot_figS11_S14_raw_fisher_20260920.py').parents[1]
strand_old = _component('plot_figS11_S14_population_reproduction')
strand_base = strand_old.base
strand_SCORE_ALIAS = 'notebook_asymmetry_score_minus_log2_nontrans_over_trans'

def strand_recalculate(population, meta, previous):
    channels = []
    for group in strand_base.GROUP_ORDER:
        source = meta['input_results']['Human|' + group]
        path = strand_ROOT / source['path']
        data = json.loads(path.read_text())['3mer']
        assert len(data['context']) == 64 and len(data['variant']) == 192
        for (change, value) in data['variant'].items():
            (ref, alt) = change.split('>')
            if 'CG' in ref:
                continue
            (n, mask, length) = (int(value['count']), int(value['mask']), int(data['context'][ref]))
            eligible = length - mask
            assert 0 <= n < eligible
            hazard = -math.log1p(-n / eligible)
            channels.append(dict(population=population, group=group, channel=change, mutation=ref[1] + '>' + alt[1], raw_count=n, mask_count=mask, reference_opportunities=length, eligible_opportunities=eligible, poisson_hazard=hazard, corrected_count=length * hazard))
    channels = pd.DataFrame(channels)
    sums = channels.groupby(['group', 'mutation']).sum(numeric_only=True)
    before = previous.set_index(['group', 'mutation'])
    rows = []
    for group in strand_base.GROUP_ORDER:
        for mutation in strand_base.MUTATION_ORDER:
            partner = mutation.translate(str.maketrans('ACGT', 'TGCA'))
            (nts, ts) = (sums.loc[group, mutation], sums.loc[group, partner])
            (n1, n0, m1, m0) = map(int, [nts.raw_count, ts.raw_count, nts.eligible_opportunities, ts.eligible_opportunities])
            assert n1 <= m1 and n0 <= m0
            dn = nts.corrected_count / nts.reference_opportunities
            dt = ts.corrected_count / ts.reference_opportunities
            score = math.log2(dn / dt) if dn > 0 and dt > 0 else math.inf if dn > 0 else -math.inf if dt > 0 else math.nan
            prior = before.loc[group, mutation]
            assert np.isclose(score, prior[strand_SCORE_ALIAS], rtol=1e-12, atol=1e-12, equal_nan=True)
            assert n1 == prior.transcribed_raw_count and n0 == prior.nontranscribed_raw_count
            result = fisher_exact([[n1, m1 - n1], [n0, m0 - n0]], alternative='two-sided')
            rows.append(dict(population=population, group=group, mutation=mutation, partner=partner, nts_count=n1, ts_count=n0, nts_eligible=m1, ts_eligible=m0, nts_reference_opportunities=int(nts.reference_opportunities), ts_reference_opportunities=int(ts.reference_opportunities), nts_corrected_count=nts.corrected_count, ts_corrected_count=ts.corrected_count, nts_corrected_density=dn, ts_corrected_density=dt, corrected_log2_asymmetry=score, displayed_score=score if np.isfinite(score) else 0.0, fisher_a=n1, fisher_b=m1 - n1, fisher_c=n0, fisher_d=m0 - n0, fisher_odds_ratio=result.statistic, fisher_p_two_sided=result.pvalue, significant_p_0_05=result.pvalue < 0.05, previous_corrected_fisher_p=prior.fisher_p_two_sided_raw, previous_significant_p_0_05=prior.fisher_p_two_sided_raw < 0.05))
    result = pd.DataFrame(rows)
    assert len(result) == 18
    return (result, channels)
human_ROOT = (_PROCESSING_ROOT / 'fig2/scripts/analyze_human_pseudo_20261005.py').parents[1]
human_DATA = human_ROOT / 'data/processed/figure2_pseudo1362_legacy_intervals_20261005'
human_REGION_BED = 'legacy_regions.bed'
human_REGION_POLICY = 'Retained legacy interval rows and strands; Fig1 callable intervals supplement added or previously absent genes'
human_TMP = human_ROOT / 'tmp/figure2_human_pseudo_20261005'
human_engine = _component('plot_figS11_S14_population_reproduction')
human_strand = _component('plot_figS11_S14_raw_fisher_20260920')
human_GROUPS = human_engine.base.GROUP_ORDER
human_PSEUDO = 'Pseudo-RNUs'
human_POPULATIONS = ['ASJ', 'EAS', 'SAS', 'FIN', 'SHUF2000']

def human_read(p):
    return pd.read_csv(p, sep='\t')

def human_old_tables(pop):
    if pop == 'ASJ':
        root = human_ROOT / 'data/processed/figure2_fig1_gene_sets_analysis_20260825'
        sig = human_ROOT / 'data/processed/figure2_sbs2ss_explicit_20260910'
        mapping = {'panel_c_fits': 'fit_statistics.tsv', 'panel_c_activities': 'signature_activities.tsv', 'panel_c_reconstruction': 'reconstruction.tsv', 'panel_c_permutations': 'permutation_summary.tsv', 'panel_c_null': 'permutation_null.tsv.gz'}
        tables = {key: human_read(sig / name) for (key, name) in mapping.items()}
        tables.update(panel_a_density=human_read(root / 'panel_a_density.tsv'), panel_b_profile=human_read(root / 'sbs96_profile.tsv'))
        return tables
    directory = human_ROOT / f'final_result/figS{human_engine.SPECS[pop][0]}/APOBEC3_labels_20260920/data'
    return {p.stem: human_read(p) for p in directory.glob('panel_*.tsv')}

def human_namespace(pop):
    ns = human_engine.namespace('EAS' if pop == 'ASJ' else pop)
    summary = human_read(human_ROOT / 'data/processed/figure2_fig1_human_populations_20260825/human_population_result_summary.tsv')
    rows = summary[summary.population == pop].set_index('group').loc[human_GROUPS]
    inputs = {g: human_ROOT / rows.at[g, 'result_path'] for g in human_GROUPS}
    inputs[human_PSEUDO] = human_DATA / pop / 'pseudo.result'
    payloads = {g: json.loads(p.read_text()) for (g, p) in inputs.items()}
    ns.update(HUMAN_POPULATION=pop, HUMAN_SHORT_LABEL='shuf2000' if pop == 'SHUF2000' else pop, HUMAN_POPULATION_LABEL='Human ' + ('shuf2000' if pop == 'SHUF2000' else pop), HUMAN_NOMINAL_INDIVIDUALS=int(rows.nominal_individuals.iloc[0]), HUMAN_MAX_AF=0.001, HUMAN_INPUTS=inputs, HUMAN_SHA256={g: human_engine.sha(p) for (g, p) in inputs.items()}, HUMAN_RETAINED={g: sum((v['count'] for v in d['3mer']['variant'].values())) for (g, d) in payloads.items()}, HUMAN_MASKED={g: sum((v['mask'] for v in d['3mer']['variant'].values())) for (g, d) in payloads.items()}, HUMAN_VCF_PATH=human_ROOT / rows.vcf_path.iloc[0], HUMAN_BED_PATHS={g: human_DATA / human_REGION_BED if g == human_PSEUDO else human_ROOT / rows.at[g, 'bed_path'] for g in human_GROUPS})
    ns['GENE_COUNTS']['Human', human_PSEUDO] = 1362
    ns['CROSS_SPECIES_INPUTS'].update({('Human', g): p for (g, p) in inputs.items()})
    ns['CROSS_SPECIES_SHA256'] = {k: human_engine.sha(p) for (k, p) in ns['CROSS_SPECIES_INPUTS'].items()}
    return (ns, rows)

def human_main():
    global human_DATA, human_REGION_BED, human_REGION_POLICY
    ap = argparse.ArgumentParser()
    ap.add_argument('--population', choices=human_POPULATIONS, required=True)
    ap.add_argument('--data-dir', type=Path,
                    help='Override the policy-specific processed-data directory.')
    ap.add_argument('--region-policy', choices=['fig1', 'legacy'], default='legacy',
                    help='Interval policy; legacy reproduces the frozen Fig2 and cohort results.')
    args = ap.parse_args()
    pop = args.population
    dirname = ('figure2_pseudo1362_legacy_intervals_20261005' if args.region_policy == 'legacy'
               else 'figure2_human_pseudo_20261005')
    human_DATA = (args.data_dir or human_ROOT / 'data/processed' / dirname).resolve()
    if args.region_policy == 'legacy':
        human_REGION_BED = 'legacy_regions.bed'
        human_REGION_POLICY = 'Retained legacy interval rows and strands; Fig1 callable intervals supplement added or previously absent genes'
    else:
        human_REGION_BED = 'fig1_regions.bed'
        human_REGION_POLICY = 'Fig1 coordinates intersected with original callable whitelist'
    out = human_DATA / pop / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    (ns, summary) = human_namespace(pop)
    old = human_old_tables(pop)
    tables = {}
    (three, one, legacy, density) = ns['analyze_panel_a']()
    meta = {'input_results': {'Human|' + g: {'path': str(p), 'sha256': human_engine.sha(p)} for (g, p) in ns['HUMAN_INPUTS'].items()}}
    (asym, rawchannels) = human_strand.recalculate(pop, meta, legacy)
    for prefix in ['nts', 'ts']:
        asym[prefix + '_mask'] = asym[prefix + '_reference_opportunities'] - asym[prefix + '_eligible']
        asym[prefix + '_raw_density'] = asym[prefix + '_count'] / asym[prefix + '_eligible']
    asym['raw_density_ratio'] = asym.nts_raw_density / asym.ts_raw_density
    with np.errstate(divide='ignore', invalid='ignore'):
        asym['raw_log2_asymmetry'] = np.log2(asym.raw_density_ratio)
    asym = asym.rename(columns={'nts_corrected_density': 'corrected_nts_density', 'ts_corrected_density': 'corrected_ts_density', 'nts_corrected_count': 'corrected_nts_count', 'ts_corrected_count': 'corrected_ts_count'})
    tables.update(panel_a_3mer=three, panel_a_1mer=one, panel_a_asymmetry=asym, panel_a_legacy_asymmetry=legacy, panel_a_density=density, raw_noncpg_channels=rawchannels)
    (matrix, profile, counts) = ns['build_sbs96_analysis']()
    tables.update(panel_b_matrix=matrix, panel_b_profile=profile, panel_b_summary=counts)
    ref = ns['load_cosmic_signature_matrix'](ns['COSMIC_SIGNATURES']).loc[matrix.index]
    explicit = pd.read_csv(human_ROOT / 'data/processed/figure2_sbs2ss_explicit_20260910/reference_signatures_explicit_SBS2ss.tsv', sep='\t', index_col=0)
    assert np.allclose(ref.SBS2, explicit.loc[ref.index, 'SBS2ss*'])
    fn = ns['fit_signature_set']
    calls = [0]

    def tracked(*a, **kw):
        calls[0] += 1
        if calls[0] % 200 == 0:
            print(f'{pop}: fit call {calls[0]}', flush=True)
        return fn(*a, **kw)
    ns['fit_signature_set'] = tracked
    with warnings.catch_warnings(record=True) as captured:
        (fits, acts, rec) = ns['run_signature_models'](matrix[[human_PSEUDO]], ref)
    (perm, null, diag) = ns['targeted_permutation_test'](matrix[[human_PSEUDO]], ref, permutations=1000, seed=42)
    perm['significant_raw_p_0.05'] = perm.empirical_p < 0.05
    acts['signature'] = acts.signature.replace({'SBS2': 'SBS2ss*'})
    for (key, new, idcol) in [('panel_c_fits', fits, 'sample'), ('panel_c_activities', acts, 'group'), ('panel_c_reconstruction', rec, 'group'), ('panel_c_permutations', perm, 'sample'), ('panel_c_null', null, 'sample')]:
        new['full_fig1_gene_count'] = 1362
        tables[key] = pd.concat([old[key][old[key][idcol] != human_PSEUDO], new], ignore_index=True)
    tables['panel_c_permutations']['significant_raw_p_0.05'] = tables['panel_c_permutations'].empirical_p < 0.05
    (rates, tests) = ns['analyze_panel_d']()
    tables.update(panel_d_rates=rates, panel_d_tests=tests)
    for (key, value) in [('panel_a_density', 'notebook_merged_density'), ('panel_b_profile', 'pseudo_count')]:
        now = tables[key][tables[key].group != human_PSEUDO]
        before = old[key][old[key].group != human_PSEUDO]
        assert np.allclose(now[value], before[value], rtol=1e-12, atol=1e-14), key
    for (key, frame) in tables.items():
        frame.to_csv(out / (key + '.tsv'), sep='\t', index=key == 'panel_b_matrix')
    ref.rename(columns={'SBS2': 'SBS2ss*'}).to_csv(out / 'reference_signatures.tsv', sep='\t')
    newsummary = summary.reset_index()
    idx = newsummary.group == human_PSEUDO
    newsummary.loc[idx, 'result_path'] = str(ns['HUMAN_INPUTS'][human_PSEUDO].relative_to(human_ROOT))
    newsummary.loc[idx, 'result_sha256'] = human_engine.sha(ns['HUMAN_INPUTS'][human_PSEUDO])
    newsummary.loc[idx, 'bed_path'] = str((human_DATA / human_REGION_BED).relative_to(human_ROOT))
    newsummary.loc[idx, 'bed_sha256'] = human_engine.sha(human_DATA / human_REGION_BED)
    newsummary.loc[idx, 'full_fig1_gene_count'] = 1362
    bed = pd.read_csv(human_DATA / human_REGION_BED, sep='\t', header=None)
    newsummary.loc[idx, 'callable_bp'] = int((bed[2] - bed[1]).sum())
    newsummary.loc[idx, 'retained_rare_snv_records'] = ns['HUMAN_RETAINED'][human_PSEUDO]
    newsummary.loc[idx, 'masked_variant_opportunities'] = ns['HUMAN_MASKED'][human_PSEUDO]
    newsummary.to_csv(out / 'population_summary.tsv', sep='\t', index=False)
    meta.update(population=pop, new_pseudogene_count=1362, region_policy=human_REGION_POLICY, other_gene_sets_unchanged=True, fit_method='likelihood_bidirectional', threshold=0.005, connected_signatures=True, permutations=1000, seed=42, p_method='exceedances/1000', normalization='legacy notebook Poisson-hazard and genomic 3mer opportunity rescaling', diagnosis=diag, model_fit_warning_count=len(captured), limitations=['Legacy ALT-record counts and mutation-specific masking retained.', 'SHUF2000 broad TCW retains within-interval unique-position special case.', 'Only human pseudogene inputs were updated; interval policy is specified above; other groups keep legacy multiplicity.', 'Common 1362 gene universe used in all cohorts; no extra FIN-only gene filtering.'])
    (out / 'analysis_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(pop, counts[['group', 'retained_variants']].to_dict('records'), flush=True)

def human_entrypoint():
    human_main()
mouse_ROOT = (_PROCESSING_ROOT / 'fig2/scripts/update_figure2_mouse1032_20261005.py').parents[1]
mouse_DATA = mouse_ROOT / 'data/processed/figure2_mouse_pseudo1032_20261005'
mouse_OLD = mouse_ROOT / 'data/processed/figure2_pseudo1362_legacy_intervals_20261005'
mouse_analysis = _component('analyze_human_pseudo_20261005')
mouse_POPS = ['ASJ', 'EAS', 'SAS', 'FIN', 'SHUF2000']
mouse_PSEUDO = 'Pseudo-RNUs'
mouse_RESULT = mouse_DATA / 'mouse_pseudo1032_af1e-2.result'

def mouse_prepare_data():
    for filename in ['legacy_regions.bed', 'legacy_regions.tsv', 'human_pseudogene_membership.tsv', 'membership_changes.tsv']:
        shutil.copy2(mouse_OLD / filename, mouse_DATA / filename)
    mouse_analysis.DATA = mouse_DATA
    mouse_analysis.REGION_BED = 'legacy_regions.bed'
    for pop in mouse_POPS:
        folder = mouse_DATA / pop / 'analysis'
        folder.mkdir(exist_ok=True, parents=True)
        for source in (mouse_OLD / pop / 'analysis').glob('*'):
            if source.is_file():
                shutil.copy2(source, folder / source.name)
        shutil.copy2(mouse_OLD / pop / 'pseudo.result', mouse_DATA / pop / 'pseudo.result')
        (ns, _) = mouse_analysis.namespace(pop)
        key = ('Mouse', mouse_PSEUDO)
        ns['CROSS_SPECIES_INPUTS'][key] = mouse_RESULT
        ns['CROSS_SPECIES_SHA256'][key] = mouse_analysis.engine.sha(mouse_RESULT)
        ns['GENE_COUNTS'][key] = 1032
        ns['EXPECTED_TCW'].pop(key, None)
        (rates, tests) = ns['analyze_panel_d']()
        before = pd.read_csv(mouse_OLD / pop / 'analysis/panel_d_rates.tsv', sep='\t')
        keep = ~((before.species == 'Mouse') & (before.group == mouse_PSEUDO))
        pd.testing.assert_frame_equal(before[keep].drop(columns='source_path').reset_index(drop=True), rates[keep].drop(columns='source_path').reset_index(drop=True), check_dtype=False)
        rates.loc[keep, 'source_path'] = before.loc[keep, 'source_path']
        rates.to_csv(folder / 'panel_d_rates.tsv', sep='\t', index=False)
        tests.to_csv(folder / 'panel_d_tests.tsv', sep='\t', index=False)
        meta = json.loads((folder / 'analysis_manifest.json').read_text())
        meta['mouse_update'] = json.loads((mouse_DATA / 'recount_manifest.json').read_text())
        meta['human_analysis_reused_unchanged'] = True
        (folder / 'analysis_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('Updated cross-species rate/test tables for all five human figure versions', flush=True)

def mouse_entrypoint():
    mouse_prepare_data()

def _prepare_runtime():
    sys.dont_write_bytecode = True
    human_TMP.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(human_TMP)

if __name__ == '__main__':
    _run_stages({'human': 'human_entrypoint', 'mouse': 'mouse_entrypoint'}, globals(), setup=_prepare_runtime)
