"""Consolidated final processing: build_supplement_data_updated.

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
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import importlib.util
import sys
import pysam
from scipy.stats import chi2, norm
statistics_GROUP_ORDER = ['pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes', 'genome_wide_neutral']
statistics_GROUP_LABELS = {'pol2_spliceRNUs': 'POLII spliceRNUs', 'functional_U6s': 'U6/U6ATAC', 'RNU_pseudogenes': 'PseudoRNUs', 'genome_wide_neutral': 'Intergenic neutral'}
statistics_COHORT_ORDER = ['EAS', 'SAS', 'FIN', 'shuf2000']
statistics_COHORT_METADATA = {'EAS': {'sample_count': 2604, 'source_label': 'gnomAD v4 EAS'}, 'SAS': {'sample_count': 2419, 'source_label': 'gnomAD v4 SAS'}, 'FIN': {'sample_count': 5316, 'source_label': 'gnomAD v4 FIN'}, 'shuf2000': {'sample_count': 2000, 'source_label': 'gnomAD v3.1.2 HGDP+TGP shuf2000'}}

def statistics_bh_adjust(values: list[float]) -> np.ndarray:
    """Apply the Benjamini-Hochberg procedure to finite p values."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = ranked * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result

def statistics_calculate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize plotted gene/locus rates and pooled count/exposure rates."""
    rows: list[dict[str, object]] = []
    for cohort in statistics_COHORT_ORDER:
        cohort_frame = frame[frame['cohort'].eq(cohort)]
        for group in statistics_GROUP_ORDER:
            subset = cohort_frame[cohort_frame['analysis_group'].eq(group)]
            rates = subset['rate'].to_numpy(float)
            event_sum = int(subset['event_count'].sum())
            exposure_sum = int(subset['exposure'].sum())
            rows.append({'cohort': cohort, 'analysis_group': group, 'gene_set_label': statistics_GROUP_LABELS[group], 'n_units': len(subset), 'sample_count': statistics_COHORT_METADATA[cohort]['sample_count'], 'mean_unit_rate': float(np.mean(rates)), 'median_unit_rate': float(np.median(rates)), 'q1_unit_rate': float(np.quantile(rates, 0.25)), 'q3_unit_rate': float(np.quantile(rates, 0.75)), 'total_events': event_sum, 'total_effective_callable_bp': exposure_sum, 'pooled_rate_per_kb': 1000 * event_sum / exposure_sum})
    return pd.DataFrame(rows)

def statistics_calculate_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate six two-sided gene-level comparisons per cohort."""
    rows: list[dict[str, object]] = []
    for cohort in statistics_COHORT_ORDER:
        cohort_frame = frame[frame['cohort'].eq(cohort)]
        cohort_rows: list[dict[str, object]] = []
        for (first, second) in combinations(statistics_GROUP_ORDER, 2):
            left = cohort_frame.loc[cohort_frame['analysis_group'].eq(first), 'rate'].to_numpy(float)
            right = cohort_frame.loc[cohort_frame['analysis_group'].eq(second), 'rate'].to_numpy(float)
            test = mannwhitneyu(left, right, alternative='two-sided', method='asymptotic')
            cohort_rows.append({'cohort': cohort, 'group_1': first, 'group_2': second, 'label_1': statistics_GROUP_LABELS[first], 'label_2': statistics_GROUP_LABELS[second], 'n_1': len(left), 'n_2': len(right), 'mean_1': float(np.mean(left)), 'mean_2': float(np.mean(right)), 'mean_rate_ratio_1_over_2': float(np.mean(left) / np.mean(right)), 'mann_whitney_u': float(test.statistic), 'mann_whitney_p': float(test.pvalue)})
        q_values = statistics_bh_adjust([row['mann_whitney_p'] for row in cohort_rows])
        for (row, q_value) in zip(cohort_rows, q_values):
            row['mann_whitney_q'] = float(q_value)
            rows.append(row)
    return pd.DataFrame(rows)
cohorts_task = _component('revise_fig1_gene_rules_20261004')
cohorts_ROOT = cohorts_task.ROOT
cohorts_TR = cohorts_task.TR
cohorts_OUT = cohorts_task.OUT / 'supplements'
cohorts_LEG = cohorts_task.OUT / 'restored70/expanded_legacy'
cohorts_read = cohorts_task.read
cohorts_save = cohorts_task.save
cohorts_G = cohorts_task.GROUPS
cohorts_LABEL = dict(zip(cohorts_G, ['POLII spliceRNUs', 'U6/U6ATAC', 'RNU pseudogenes', 'Control']))

def cohorts_module(path, name):
    return _component('build_figS2_data')

def cohorts_current_ids():
    d = cohorts_read(cohorts_task.OUT / 'panel_a_gene_locus_density.tsv')
    return (d, {s: set(f.loc[f.analysis_group.ne(cohorts_G[-1]), 'unit_id']) for (s, f) in d.groupby('species')})

def cohorts_popdata():
    dest = cohorts_OUT / 'FigS2'
    dest.mkdir(exist_ok=True)
    dest3 = cohorts_OUT / 'FigS3'
    dest3.mkdir(exist_ok=True)
    (current, ids) = cohorts_current_ids()
    old = cohorts_read(cohorts_TR / 'final_result/figS2/revised_autosomal_control_20260916/figS2_gene_density.tsv.gz')
    raw = cohorts_read(cohorts_LEG / 'gnomad_four_populations_af1e3_neutral1000_four_group_density.tsv')
    raw = pd.concat([raw, cohorts_read(cohorts_LEG / 'shuf2000_af1e3_neutral1000_four_group_density.tsv')], ignore_index=True)
    raw['cohort'] = raw.dataset.map({'gnomADv4_EAS': 'EAS', 'gnomADv4_SAS': 'SAS', 'gnomADv4_FIN': 'FIN', 'shuf2000': 'shuf2000'})
    rows = []
    for cohort in ['EAS', 'SAS', 'FIN', 'shuf2000']:
        oldc = old[old.cohort.eq(cohort)]
        genes = raw[raw.cohort.eq(cohort) & raw.stable_gene_id.isin(ids['Human']) & raw.effective_callable_bp.ge(50)].copy()
        template = oldc.iloc[0].to_dict()
        for r in genes.itertuples():
            rows.append(template | dict(analysis_group=r.gene_set, unit_id=r.stable_gene_id, display_name=r.display_name, input_interval_bp=r.exon_union_bp, callable_bp=r.callable_bp, mask_bp=r.mask_bp, effective_callable_bp=r.effective_callable_bp, event_count=r.rare_snv_alt_count, exposure=r.effective_callable_bp, rate=r.rare_snv_density_per_kb, gene_set_label=cohorts_LABEL[r.gene_set], group_order=cohorts_G.index(r.gene_set)))
        rows += oldc[oldc.analysis_group.eq(cohorts_G[-1])].to_dict('records')
    d = pd.DataFrame(rows).reindex(columns=old.columns)
    b = cohorts_module(_component_file('build_figS2_data'), 'supp2_stats')
    b.GROUP_LABELS = cohorts_LABEL
    summary = b.calculate_summary(d)
    tests = b.calculate_tests(d)
    cohorts_save(d, dest / 'figS2_gene_density.tsv.gz')
    cohorts_save(summary, dest / 'figS2_group_summary.tsv')
    cohorts_save(tests, dest / 'figS2_pairwise_statistics.tsv')
    cohorts_save(d, dest3 / 'FigS3_locus_density.tsv.gz')
    cohorts_save(tests, dest3 / 'FigS3_pairwise_statistics.tsv')
    prior = cohorts_read(cohorts_ROOT / 'data/processed/fig1_gene_rules_20261003/supplements/FigS3/FigS3_profile_gene_bins.tsv.gz')
    pe = cohorts_task.profile_engine()
    ci = pe.load_callable_index(cohorts_TR / 'hg38.genmap.k150e1_unique.cov_16-48.bed')
    native = cohorts_task.catalogs()['Human']
    paths = {'EAS': cohorts_TR / 'gnomadv4/eas/gnomad.genomes.v4.0.sites.eas.allchr.vcf.gz', 'SAS': cohorts_TR / 'gnomadv4/sas/gnomad.genomes.v4.0.sites.sas.allchr.vcf.gz', 'FIN': cohorts_TR / 'gnomadv4/fin/gnomad.genomes.v4.0.sites.fin.allchr.vcf.gz', 'shuf2000': cohorts_TR / '1kG/gnomad.genomes.v4.0.sites.shuf2000.allchr.noGT.vcf.gz'}
    profiles = []
    bars = []
    denoms = []
    summaries = []
    for cohort in paths:
        locus = d[d.cohort.eq(cohort)]
        genes = locus[locus.analysis_group.ne(cohorts_G[-1])].set_index('unit_id')
        p = prior[prior.cohort.eq(cohort) & prior.stable_gene_id.isin(genes.index)].copy()
        missing = set(genes.index) - set(p.stable_gene_id)
        extra = []
        with pysam.VariantFile(str(paths[cohort].resolve())) as vcf:
            for gid in sorted(missing):
                g = native.loc[gid]
                bins = pe.build_profile_bins(pd.DataFrame([dict(species='Human', analysis_group=cohorts_G[2], stable_gene_id=gid, display_name=g.gene_name, chrom=g.chrom, start0=int(g.start), end=int(g.end), strand=g.strand, coordinate_source='GENCODE v44')]))
                (vals, qc) = pe.classify_gene_profile_direct('Human', bins, ci, vcf)
                vals['cohort'] = cohort
                extra.append(vals)
        p = pd.concat([p] + extra, ignore_index=True)
        body = p[p.region.eq('body')].groupby('stable_gene_id')[['rare_snv_alt_count', 'effective_callable_bp']].sum()
        np.testing.assert_allclose(body.loc[genes.index].to_numpy(), genes[['event_count', 'effective_callable_bp']].to_numpy(), rtol=1e-12)
        ctl = locus[locus.analysis_group.eq(cohorts_G[-1])]
        cm = ctl.rate.mean()
        cp = 1000 * ctl.event_count.sum() / ctl.effective_callable_bp.sum()
        denoms.append(dict(cohort=cohort, control_mean_density_per_kb=cm, control_pooled_density_per_kb=cp))
        for g in cohorts_G:
            q = locus[locus.analysis_group.eq(g)]
            bars.append(dict(cohort=cohort, analysis_group=g, n_units=len(q), mean_density_per_kb=q.rate.mean(), mean_density_ratio_to_control=q.rate.mean() / cm))
        s = p.groupby(['analysis_group', 'bin_index', 'region', 'relative_start', 'relative_end'], as_index=False).agg(gene_count=('stable_gene_id', 'size'), contributing_genes=('effective_callable_bp', lambda v: (v > 0).sum()), total_rare_snv_alt_count=('rare_snv_alt_count', 'sum'), total_effective_callable_bp=('effective_callable_bp', 'sum'))
        s['cohort'] = cohort
        s['pooled_density_per_kb'] = 1000 * s.total_rare_snv_alt_count / s.total_effective_callable_bp.replace(0, np.nan)
        s['control_pooled_density_per_kb'] = cp
        s['density_ratio_to_control'] = s.pooled_density_per_kb / cp
        for side in ['start', 'end']:
            v = s['relative_' + side]
            s['display_' + side] = np.where(v < 0, v, np.where(v <= 1000, v * 2, v + 1000))
        profiles.append(p)
        summaries.append(s)
        print('Population', cohort, 'genes', len(genes), 'new profiles', len(missing), flush=True)
    cohorts_save(pd.concat(profiles, ignore_index=True), dest3 / 'FigS3_profile_gene_bins.tsv.gz')
    cohorts_save(pd.concat(summaries, ignore_index=True), dest3 / 'FigS3_profile_summary.tsv')
    cohorts_save(pd.DataFrame(bars), dest3 / 'FigS3_control_relative_means.tsv')
    cohorts_save(pd.DataFrame(denoms), dest3 / 'FigS3_control_denominators.tsv')
    cohorts_save(pd.DataFrame([dict(check='all profile body counts and denominators match locus data', status='PASS')]), dest3 / 'FigS3_validation_checks.tsv')

def cohorts_dnm():
    dest = cohorts_OUT / 'FigS5'
    dest.mkdir(exist_ok=True)
    (cur, ids) = cohorts_current_ids()
    old = cohorts_read(cohorts_TR / 'final_result/figS5/locus_DNM_data.tsv')
    src = cohorts_read(cohorts_LEG / 'palsson2025_nodedup_dnm_neutral1000_four_group_density.tsv')
    genes = src[src.stable_gene_id.isin(ids['Human'])]
    template = old.iloc[0].to_dict()
    rows = [template | dict(analysis_group=r.gene_set, unit_id=r.stable_gene_id, input_interval_bp=r.exon_union_bp, callable_bp=r.callable_bp, effective_callable_bp=r.callable_bp, event_count=r.dnm_event_count, exposure=r.exposure_site_generations, rate=r.dnm_rate_per_1e8_site_generations, display_label=cohorts_LABEL[r.gene_set]) for r in genes.itertuples()]
    d = pd.concat([pd.DataFrame(rows), old[old.analysis_group.eq(cohorts_G[-1])]], ignore_index=True)
    s = []
    ts = []
    for g in cohorts_G:
        q = d[d.analysis_group.eq(g)]
        n = q.event_count.sum()
        e = q.exposure.sum()
        s.append(dict(analysis_group=g, panel='b', n_units=len(q), zero_count_units=int(q.event_count.eq(0).sum()), total_events=n, total_exposure=e, median_unit_rate=q.rate.median(), mean_unit_rate=q.rate.mean(), pooled_rate=100000000.0 * n / e, pooled_rate_ci_low=100000000.0 * 0.5 * chi2.ppf(0.025, 2 * n) / e if n else 0, pooled_rate_ci_high=100000000.0 * 0.5 * chi2.ppf(0.975, 2 * (n + 1)) / e))
    ss = pd.DataFrame(s).set_index('analysis_group')
    for (a, b) in combinations(cohorts_G, 2):
        (x, y) = (ss.loc[a], ss.loc[b])
        pooled = (x.total_events + y.total_events) / (x.total_exposure + y.total_exposure)
        z = (x.total_events / x.total_exposure - y.total_events / y.total_exposure) / np.sqrt(pooled * (1 / x.total_exposure + 1 / y.total_exposure))
        ts.append(dict(panel='b', group_1=a, group_2=b, count_1=x.total_events, count_2=y.total_events, exposure_1=x.total_exposure, exposure_2=y.total_exposure, pooled_rate_ratio_1_over_2=x.total_events / x.total_exposure / (y.total_events / y.total_exposure), poisson_score_z=z, poisson_score_p=2 * norm.sf(abs(z))))
    cohorts_save(d, dest / 'locus_DNM_data.tsv')
    cohorts_save(ss.reset_index(), dest / 'aggregate_rates.tsv')
    cohorts_save(pd.DataFrame(ts), dest / 'unadjusted_score_tests.tsv')
    print('Updated S5 DNM rates', ss[['n_units', 'total_events']].to_dict(), flush=True)

def cohorts_entrypoint():
    cohorts_popdata()
    cohorts_dnm()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    _run_stages({'cohorts': 'cohorts_entrypoint'}, globals(), setup=_prepare_runtime)
