"""Consolidated final processing: prepare_signals_updated.

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
import sys
import json
import gzip
import argparse
import math
import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import wilcoxon, norm, fisher_exact
import shutil
from scipy.stats import spearmanr
callable_ROOT = (_PROCESSING_ROOT / 'fig3/final_result/fig3/prepare_callable_fig3_20260917.py').parents[2]
callable_OUT = callable_ROOT / 'final_result/fig3/callable_mask_20260917'
callable_DATA = callable_OUT / 'data'
callable_RES = callable_OUT / 'resources'
callable_core = _component('recompute')
callable_GROUPS = callable_core.GROUPS
callable_MASK = callable_ROOT / 'fig2/transpecies_compare/hg38.genmap.k150e1_unique.cov_16-48.bed'
callable_CHAIN = Path('/public5/home/xuejing/RNU/KAS-CUTandTag_seq/deeptools/hg38ToHg19.over.chain.gz')
callable_COHORT = callable_ROOT / 'final_result/fig3/T47D_linear_7pt_20260908/b_cohort.tsv'
callable_TRACK = callable_core.E[0]

def callable_save(frame, name):
    frame.to_csv(callable_DATA / f'{name}.tsv', sep='\t', index=False)
callable_read = _partial(_shared.read_table, low_memory=False)

def callable_merge(a):
    if not len(a):
        return np.empty((0, 2), np.int64)
    a = a[np.lexsort((a[:, 1], a[:, 0]))]
    mx = np.maximum.accumulate(a[:, 1])
    idx = np.r_[0, np.flatnonzero(a[1:, 0] > mx[:-1]) + 1]
    return np.c_[a[idx, 0], np.maximum.reduceat(a[:, 1], idx)]

class callable_KeepMask(callable_core.Intervals):
    """Indexed whitelist, stored as merged int64 arrays per chromosome."""

    def __init__(self, d):
        self.d = d

    @classmethod
    def load(cls, assembly):
        with np.load(callable_RES / f'callable_{assembly}.npz') as x:
            return cls({k: x[k] for k in x.files})

    def total(self):
        return sum((np.diff(a, axis=1).sum() for a in self.d.values()))

class callable_ExcludeOutside:
    """Adapt a keep mask to the existing exact integration exclusion interface."""

    def __init__(self, keep):
        self.keep = keep

    def overlaps(self, c, s, e):
        keep = self.keep.overlaps(c, s, e)
        if not len(keep):
            return np.array([[s, e]], dtype=np.int64)
        starts = np.r_[s, keep[:, 1]]
        ends = np.r_[keep[:, 0], e]
        return np.c_[starts, ends][ends > starts]

def callable_prepare_masks():
    frame = pd.read_csv(callable_MASK, sep='\t', header=None, usecols=[0, 1, 2], names=['chrom', 'start', 'end'], dtype={'chrom': 'category', 'start': 'int64', 'end': 'int64'})
    good = callable_KeepMask({str(c): callable_merge(g[['start', 'end']].to_numpy()) for (c, g) in frame.groupby('chrom', observed=True)})
    np.savez_compressed(callable_RES / 'callable_hg38.npz', **good.d)
    projected = {}
    mapped_source = {}
    with gzip.open(callable_CHAIN, 'rt') as f:
        for line in f:
            x = line.split()
            if not x:
                continue
            if x[0] == 'chain':
                (_, score, tc, tsize, tstrand, tstart, tend, qc, qsize, qstrand, qstart, qend, cid) = x
                assert tstrand == '+'
                t = int(tstart)
                q = int(qstart)
                qs = int(qsize)
            else:
                n = int(x[0])
                a = good.overlaps(tc, t, t + n)
                if len(a):
                    mapped_source.setdefault(tc, []).append(a)
                    b = a - t + q
                    if qstrand == '-':
                        b = np.c_[qs - b[:, 1], qs - b[:, 0]]
                    projected.setdefault(qc, []).append(b)
                t += n
                q += n
                if len(x) == 3:
                    t += int(x[1])
                    q += int(x[2])
    hg19 = callable_KeepMask({c: callable_merge(np.concatenate(parts)) for (c, parts) in projected.items()})
    src = callable_KeepMask({c: callable_merge(np.concatenate(parts)) for (c, parts) in mapped_source.items()})
    np.savez_compressed(callable_RES / 'callable_hg19.npz', **hg19.d)
    manifest = dict(mask_path=str(callable_MASK), resolved_mask_path=str(callable_MASK.resolve()), mask_sha256=callable_core.sha(callable_MASK), chain_path=str(callable_CHAIN), chain_sha256=callable_core.sha(callable_CHAIN), semantics='KEEP, not exclude', hg38_callable_bp=int(good.total()), hg38_callable_mapped_bp=int(src.total()), hg19_callable_union_bp=int(hg19.total()), mapping='Exact ungapped chain-block projection; union target positions', mouse_filter=False, ENCODE_blacklist=False)
    (callable_OUT / 'mask_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(manifest, flush=True)

def callable_abe():
    d = callable_read(callable_COHORT)
    keep = callable_KeepMask.load('hg19')
    mask = callable_ExcludeOutside(keep)
    mat = np.full((len(d), 100), np.nan)
    bps = np.zeros_like(mat)
    rows = []
    evals = []
    checks = []
    with pyBigWig.open(str(callable_core.E[0])) as bw, pyBigWig.open(str(callable_core.E[1])) as rh:
        for (i, r) in enumerate(d.itertuples()):
            edges = callable_core.edges_for(r.start_hg19, r.end_hg19)
            (sm, ct, _, _, geom) = callable_core.integrate_tracks([bw], r.chrom_hg19, edges, mask, positive=False)
            b = callable_core.ratio(sm[0], ct[0])
            mat[i] = b[::-1] if r.strand == '-' else b
            bps[i] = ct[0, ::-1] if r.strand == '-' else ct[0]
            total = sm[0, 40:60].sum()
            den = ct[0, 40:60].sum()
            allowed = geom[40:60].sum()
            row = r._asdict()
            row.pop('Index', None)
            row.update(callable_bp=allowed, excluded_bp=r.end_hg19 - r.start_hg19 - allowed, signal_sum=total, effective_signal_bp=den, FE=total / den if den else np.nan, status='ok' if den else 'no_callable_bases' if not allowed else 'no_finite_signal')
            rows.append(row)
            (rs, rc, _, _, rg) = callable_core.integrate_tracks([rh], r.chrom_hg19, np.array([r.start_hg19, r.end_hg19]), mask)
            for (condition, ss, cc) in [('lipid_control', total, den), ('rnaseh1', rs[0, 0], rc[0, 0])]:
                evals.append(dict(stable_gene_id=r.stable_gene_id, display_name=r.display_name, gene_set=r.gene_set, condition=condition, chrom=r.chrom_hg19, start=r.start_hg19, end=r.end_hg19, callable_bp=allowed, signal_sum=ss, effective_signal_bp=cc, spi_seq_fold_enrichment=ss / cc if cc else np.nan))
            if r.gene_set == callable_GROUPS[0]:
                arr = bw.values(r.chrom_hg19, r.start_hg19, r.end_hg19, numpy=True).astype(float)
                ok = np.zeros(len(arr), bool)
                for (s, e) in keep.overlaps(r.chrom_hg19, r.start_hg19, r.end_hg19):
                    ok[s - r.start_hg19:e - r.start_hg19] = True
                ok &= np.isfinite(arr)
                value = arr[ok].mean() if ok.any() else np.nan
                np.testing.assert_allclose(value, row['FE'], equal_nan=True, rtol=1e-07)
                checks.append(dict(gene_id=r.stable_gene_id, FE=value, effective_bp=int(ok.sum())))
            if (i + 1) % 4000 == 0:
                print('a/b/e', i + 1, '/', len(d), flush=True)
    a = pd.DataFrame(rows)
    valid = a.FE.notna().to_numpy()
    ev = pd.DataFrame(evals)
    callable_save(a, 'a_gene_values')
    callable_save(callable_core.pairwise(a, 'FE'), 'a_pairwise')
    callable_save(callable_core.profile_summary(a[valid], mat[valid]), 'b_profile')
    callable_save(ev, 'e_gene_values')
    callable_save(pd.DataFrame(checks), 'abe_independent_checks')
    np.testing.assert_allclose(callable_core.ratio(np.nansum(mat[:, 40:60] * bps[:, 40:60], axis=1), bps[:, 40:60].sum(axis=1)), a.FE, equal_nan=True, rtol=1e-07)
    np.savez_compressed(callable_DATA / 'ab_gene_matrices.npz', gene_ids=d.stable_gene_id.to_numpy(str), FE=mat, effective_bp=bps)
    tests = []
    for g in callable_GROUPS:
        p = ev[ev.gene_set.eq(g)].pivot(index='stable_gene_id', columns='condition', values='spi_seq_fold_enrichment').dropna()
        (x, y) = (p.lipid_control.to_numpy(), p.rnaseh1.to_numpy())
        test = wilcoxon(y, x, alternative='two-sided', zero_method='wilcox', correction=False, method='approx')
        lp = (math.log(2) + norm.logsf(abs(test.zstatistic))) / math.log(10)
        tests.append(dict(gene_set=g, n_paired_loci=len(p), control_median=np.median(x), rnaseh1_median=np.median(y), median_fold_rnaseh1_over_control=np.median(y) / np.median(x), paired_wilcoxon_statistic=test.statistic, p_two_sided_asymptotic=10 ** lp, log10_p_two_sided_asymptotic=lp))
    callable_save(pd.DataFrame(tests), 'e_paired')
    print(a.groupby('gene_set').FE.agg(['count', 'median']), flush=True)

def callable_cd():
    d = callable_read(callable_core.AUDIT / 'cohort_hg38.tsv')
    d = d[d.matched_panel_c].reset_index(drop=True)
    keep = callable_KeepMask.load('hg38')
    mask = callable_ExcludeOutside(keep)
    ip = np.zeros((len(d), 100))
    inp = np.zeros_like(ip)
    bodyip = np.zeros(len(d))
    bodyin = np.zeros(len(d))
    lengths = np.zeros_like(ip)
    gbp = np.zeros(len(d))
    norms = []
    for num in range(3, 9):
        sample = f'SRX267500{num}'
        bw = pyBigWig.open(str(callable_core.AUDIT / f'{sample}_hg38.bw'))
        integral = 0.0
        bp = 0
        for chrom in ['chr' + str(k) for k in range(1, 23)]:
            for start in range(0, bw.chroms()[chrom], 5000000):
                end = min(start + 5000000, bw.chroms()[chrom])
                (sm, ct, _, _, geom) = callable_core.integrate_tracks([bw], chrom, np.array([start, end]), mask)
                integral += sm[0, 0]
                bp += geom[0]
        factor = integral / bp
        norms.append(dict(sample=sample, callable_autosomal_signal_sum=integral, callable_autosomal_bp=bp, normalization_factor=factor))
        print('Callable normalization', sample, factor, flush=True)
        sv = []
        rawsum = []
        for (i, r) in enumerate(d.itertuples()):
            edges = callable_core.edges_for(r.start, r.end, 'floor')
            (sm, ct, _, _, geom) = callable_core.integrate_tracks([bw], r.chrom, edges, mask)
            m = callable_core.ratio(sm[0], geom) / factor
            den = geom[40:60].sum()
            total = sm[0, 40:60].sum()
            v = total / den / factor if den else np.nan
            if r.strand == '-':
                m = m[::-1]
            if num % 2:
                ip[i] += m / 3
                bodyip[i] += v / 3
            else:
                inp[i] += m / 3
                bodyin[i] += v / 3
            gbp[i] = den
            lengths[i] = geom[::-1] if r.strand == '-' else geom
            sv.append(v)
            rawsum.append(total)
            if (i + 1) % 7000 == 0:
                print('c/d', sample, i + 1, flush=True)
        d[sample] = sv
        d[sample + '_raw_sum'] = rawsum
        bw.close()
    d['callable_bp'] = gbp
    d['excluded_bp'] = d.end - d.start - gbp
    d['normalized_ip_mean'] = bodyip
    d['normalized_input_mean'] = bodyin
    d['FE'] = callable_core.ratio(bodyip, bodyin)
    d['status'] = np.where(gbp == 0, 'no_callable_bases', np.where(bodyin > 0, 'ok', 'zero_input'))
    valid = d.FE.notna().to_numpy()
    fe = callable_core.ratio(ip, inp)
    callable_save(d, 'c_gene_values')
    callable_save(pd.DataFrame(norms), 'cd_normalization')
    callable_save(callable_core.pairwise(d, 'FE'), 'c_pairwise')
    callable_save(callable_core.profile_summary(d[valid], fe[valid]), 'd_profile')
    np.savez_compressed(callable_DATA / 'cd_gene_matrices.npz', gene_ids=d.stable_gene_id.to_numpy(str), ip=ip, input=inp, FE=fe, callable_bp=lengths)
    print(d.groupby('gene_set').FE.agg(['count', 'median']), flush=True)

def callable_f():
    d = callable_read(callable_ROOT / 'data/processed/fig3_panel_d_nine_public_set_gene_level.tsv')
    keep = callable_KeepMask.load('hg38')
    rows = []
    for species in ['Human', 'Mouse']:
        part = d[d.species.eq(species)]
        peaks = callable_core.bed(part.rloop_input.iloc[0])
        for r in part.itertuples():
            allowed = keep.overlaps(r.chrom, r.start0, r.end) if species == 'Human' else np.array([[r.start0, r.end]])
            bp = int(np.diff(allowed, axis=1).sum())
            ov = 0
            for (s, e) in allowed:
                ov += int(np.diff(peaks.overlaps(r.chrom, s, e), axis=1).sum())
            row = r._asdict()
            row.pop('Index', None)
            row.pop('overlap_component_count', None)
            row.update(original_overlap_bp=r.overlap_bp, original_rloop_overlap=r.rloop_overlap, callable_bp=bp, excluded_bp=r.end - r.start0 - bp, overlap_bp=ov, overlap_fraction=ov / bp if bp else np.nan, rloop_overlap=ov > 0, eligible=bp > 0)
            rows.append(row)
    v = pd.DataFrame(rows)
    summary = []
    tests = []
    for g in ['pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes']:
        nums = {}
        for species in ['Human', 'Mouse']:
            x = v[v.analysis_group.eq(g) & v.species.eq(species) & v.eligible]
            k = int(x.rloop_overlap.sum())
            n = len(x)
            p = k / n
            z = norm.ppf(0.975)
            den = 1 + z * z / n
            cent = (p + z * z / 2 / n) / den
            half = z * np.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / den
            nums[species] = (k, n)
            summary.append(dict(analysis_group=g, species=species, overlap_genes=k, total_genes=n, percent=p * 100, ci95_low_percent=(cent - half) * 100, ci95_high_percent=(cent + half) * 100))
        (h, hn) = nums['Human']
        (m, mn) = nums['Mouse']
        (od, p) = fisher_exact([[h, hn - h], [m, mn - m]], alternative='two-sided')
        tests.append(dict(analysis_group=g, human_overlap=h, human_total=hn, mouse_overlap=m, mouse_total=mn, odds_ratio_human_vs_mouse=od, fisher_exact_p_two_sided=p))
    mouse = v[v.species.eq('Mouse')]
    assert (mouse.overlap_bp == mouse.original_overlap_bp).all()
    callable_save(v, 'f_gene_values')
    callable_save(pd.DataFrame(summary), 'f_summary')
    callable_save(pd.DataFrame(tests), 'f_comparisons')
    print(pd.DataFrame(summary).to_string(index=False), flush=True)

def callable_entrypoint():
    global callable_args, callable_p
    callable_p = argparse.ArgumentParser()
    callable_p.add_argument('stage', choices=['masks', 'abe', 'cd', 'f'])
    callable_args = callable_p.parse_args()
    callable_DATA.mkdir(parents=True, exist_ok=True)
    callable_RES.mkdir(parents=True, exist_ok=True)
    actions = {'masks': callable_prepare_masks, 'abe': callable_abe,
               'cd': callable_cd, 'f': callable_f}
    actions[callable_args.stage]()
replicates_previous = _component('prepare_callable_fig3_20260917')
replicates_ROOT = replicates_previous.ROOT
replicates_SOURCE = replicates_previous.OUT
replicates_OUT = replicates_ROOT / 'final_result/fig3/callable_siNT_20260917'
replicates_DATA = replicates_OUT / 'data'
replicates_core = replicates_previous.core
replicates_GROUPS = replicates_core.GROUPS

def replicates_save(d, n):
    d.to_csv(replicates_DATA / f'{n}.tsv', sep='\t', index=False)

def replicates_main():
    replicates_DATA.mkdir(parents=True, exist_ok=True)
    unchanged = []
    for name in ['c_gene_values', 'c_pairwise', 'd_profile', 'e_gene_values', 'e_paired', 'f_gene_values', 'f_summary', 'f_comparisons', 'cd_normalization']:
        src = replicates_SOURCE / 'data' / f'{name}.tsv'
        dest = replicates_DATA / src.name
        shutil.copyfile(src, dest)
        unchanged.append(name)
    shutil.copyfile(replicates_SOURCE / 'mask_manifest.json', replicates_OUT / 'mask_manifest.json')
    d = replicates_previous.read(replicates_previous.COHORT)
    keep = replicates_previous.KeepMask.load('hg19')
    mask = replicates_previous.ExcludeOutside(keep)
    matrices = np.full((len(d), 100), np.nan)
    bp = np.zeros_like(matrices)
    rep_mats = np.full((2, len(d), 100), np.nan)
    rows = []
    checks = []
    binchecks = 0
    bws = [pyBigWig.open(str(x)) for x in replicates_core.AB]
    for (i, r) in enumerate(d.itertuples()):
        edges = replicates_core.edges_for(r.start_hg19, r.end_hg19)
        (sm, ct, total, n, geom) = replicates_core.integrate_tracks(bws, r.chrom_hg19, edges, mask, positive=True)
        m = replicates_core.ratio(total, n)
        matrices[i] = m[::-1] if r.strand == '-' else m
        bp[i] = n[::-1] if r.strand == '-' else n
        value = total[40:60].sum() / n[40:60].sum() if n[40:60].sum() else np.nan
        row = r._asdict()
        row.pop('Index', None)
        row.update(callable_bp=geom[40:60].sum(), excluded_bp=r.end_hg19 - r.start_hg19 - geom[40:60].sum(), signal_sum=total[40:60].sum(), effective_signal_bp=n[40:60].sum(), FE=value, status='ok' if n[40:60].sum() else 'no_callable_bases' if not geom[40:60].sum() else 'no_positive_finite_signal')
        for k in range(2):
            rv = replicates_core.ratio(sm[k], ct[k])
            rep_mats[k, i] = rv[::-1] if r.strand == '-' else rv
            row[f'rep{k + 1}_signal_sum'] = sm[k, 40:60].sum()
            row[f'rep{k + 1}_effective_bp'] = ct[k, 40:60].sum()
            row[f'rep{k + 1}_FE'] = sm[k, 40:60].sum() / ct[k, 40:60].sum() if ct[k, 40:60].sum() else np.nan
        rows.append(row)
        if r.gene_set == replicates_GROUPS[0]:
            for (j, (s, e)) in enumerate([(r.start_hg19, r.end_hg19)] + [(int(edges[x]), int(edges[x + 1])) for x in [0, 39, 40, 49, 59, 60, 99]]):
                vals = np.array([bw.values(r.chrom_hg19, s, e, numpy=True) for bw in bws], float)
                allowed = np.zeros(e - s, bool)
                for (u, v) in keep.overlaps(r.chrom_hg19, s, e):
                    allowed[u - s:v - s] = True
                vals[:, ~allowed] = np.nan
                vals[vals <= 0] = np.nan
                count = np.isfinite(vals).sum(axis=0)
                good = count > 0
                av = replicates_core.ratio(np.nansum(vals, axis=0), count)
                avg = av[good].mean() if good.any() else np.nan
                expected = value if j == 0 else m[[0, 39, 40, 49, 59, 60, 99][j - 1]]
                np.testing.assert_allclose(avg, expected, equal_nan=True, rtol=1e-08, atol=1e-07)
                if j == 0:
                    checks.append(dict(gene_id=r.stable_gene_id, FE=avg, effective_bp=int(good.sum())))
                else:
                    binchecks += 1
        if (i + 1) % 4000 == 0:
            print('siNT a/b', i + 1, '/', len(d), flush=True)
    for bw in bws:
        bw.close()
    a = pd.DataFrame(rows)
    valid = a.FE.notna().to_numpy()
    old = replicates_previous.read(replicates_SOURCE / 'data/a_gene_values.tsv').set_index('stable_gene_id')
    np.testing.assert_array_equal(a.callable_bp, old.loc[a.stable_gene_id, 'callable_bp'])
    np.testing.assert_allclose(replicates_core.ratio(np.nansum(matrices[:, 40:60] * bp[:, 40:60], axis=1), bp[:, 40:60].sum(axis=1)), a.FE, equal_nan=True, rtol=1e-08, atol=1e-07)
    replicates_save(a, 'a_gene_values')
    replicates_save(replicates_core.pairwise(a, 'FE'), 'a_pairwise')
    replicates_save(replicates_core.profile_summary(a[valid], matrices[valid]), 'b_profile')
    replicates_save(pd.DataFrame(checks), 'abe_independent_checks')
    np.savez_compressed(replicates_DATA / 'ab_gene_matrices.npz', gene_ids=d.stable_gene_id.to_numpy(str), FE=matrices, effective_bp=bp, replicate_FE=rep_mats)
    concordance = []
    individual = []
    for g in replicates_GROUPS:
        good = a[a.gene_set.eq(g)].dropna(subset=['rep1_FE', 'rep2_FE'])
        (rho, p) = spearmanr(good.rep1_FE, good.rep2_FE)
        concordance.append(dict(gene_set=g, n_both_replicates=len(good), spearman_rho=rho, p=p))
        for k in [1, 2]:
            v = a.loc[a.gene_set.eq(g), f'rep{k}_FE'].dropna()
            individual.append(dict(gene_set=g, replicate=k, n=len(v), median=v.median(), mean=v.mean()))
    replicates_save(pd.DataFrame(concordance), 'ab_replicate_concordance')
    replicates_save(pd.DataFrame(individual), 'ab_replicate_summary')
    oldaudit = replicates_previous.read(replicates_SOURCE / 'data/cohort_audit.tsv')
    audit = []
    for g in replicates_GROUPS:
        x = a[a.gene_set.eq(g)]
        audit.append(dict(panel='a/b', gene_set=g, input_n=len(x), valid_n=int(x.FE.count()), no_callable_bases=int((x.callable_bp == 0).sum()), other_missing=int(((x.callable_bp > 0) & x.FE.isna()).sum()), partially_callable=int(((x.callable_bp > 0) & (x.callable_bp < x.gene_length_hg19_bp)).sum()), callable_bp_total=x.callable_bp.sum(), original_bp_total=x.gene_length_hg19_bp.sum()))
    replicates_save(pd.concat([pd.DataFrame(audit), oldaudit[oldaudit.panel.ne('a/b')]], ignore_index=True), 'cohort_audit')
    comp = []
    for g in replicates_GROUPS:
        x = a[a.gene_set.eq(g)].copy()
        x['lipid_FE'] = old.loc[x.stable_gene_id, 'FE'].to_numpy()
        common = x.dropna(subset=['FE', 'lipid_FE'])
        comp.append(dict(gene_set=g, siNT_n=int(x.FE.count()), lipid_n=int(x.lipid_FE.count()), common_n=len(common), siNT_median=x.FE.median(), lipid_median=x.lipid_FE.median(), common_siNT_median=common.FE.median(), common_lipid_median=common.lipid_FE.median()))
    replicates_save(pd.DataFrame(comp), 'ab_vs_lipid_comparison')
    manifest = dict(status='PASS', tracks=[dict(path=str(x), sha256=replicates_core.sha(x)) for x in replicates_core.AB], mask_source=str(replicates_SOURCE / 'mask_manifest.json'), mask_identical_to_lipid=True, positive_finite_support=True, independent_gene_checks=len(checks), independent_bin_checks=binchecks, ab_body_reconstruction=True, unchanged_cdef_tables=unchanged)
    (replicates_OUT / 'numeric_QA.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(a.groupby('gene_set').FE.agg(['count', 'median']), flush=True)
    print(pd.DataFrame(concordance).to_string(index=False), flush=True)

def replicates_entrypoint():
    replicates_main()

def _prepare_runtime():
    sys.path.insert(0, str(callable_ROOT / 'final_result/fig3/blacklist_filtered_20260917/scripts'))

if __name__ == '__main__':
    _run_stages({'callable': 'callable_entrypoint', 'replicates': 'replicates_entrypoint'}, globals(), setup=_prepare_runtime)
