"""Consolidated final processing: signal_utils.

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
import gzip
import hashlib
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
constants_GROUP_ORDER = ['POL2_spliceRNUs_strict41', 'functional_U6_U6ATAC_116', 'RNU_pseudogene_operational_complement', 'lecluze_sperm_expressed_genes']
intervals_OUT = (_PROCESSING_ROOT / 'fig3/final_result/fig3/blacklist_filtered_20260917/scripts/recompute.py').parents[1]
intervals_ROOT = intervals_OUT.parents[2]
intervals_DATA = intervals_OUT / 'data'
intervals_prep = _component('prepare_figure3_data')
intervals_GROUPS = intervals_prep.GROUP_ORDER
intervals_A3ROOT = Path('/public5/home/xuejing/RNU/APOBEC3B')
intervals_AB = [intervals_A3ROOT / 'GSM5777032_3_siNT_DMSO_rep1_FE.bw', intervals_A3ROOT / 'GSM5777033_3_siNT_DMSO_rep2_FE.bw']
intervals_E = [intervals_A3ROOT / 'GSM5795929_Flag-A3B_SPI-seq_lipid.bw', intervals_A3ROOT / 'GSM5795927_Flag-A3B_SPI-seq_RNaseH1.bw']
intervals_AUDIT = intervals_ROOT / 'final_result/fig3/rloop_ce_audit_20260911/data'
intervals_read = _partial(_shared.read_table, low_memory=False)

def intervals_save(d, name):
    d.to_csv(intervals_DATA / f'{name}.tsv', sep='\t', index=False)
intervals_sha = _shared.sha256_file

class intervals_Intervals:

    def __init__(self, rows):
        grouped = {}
        for (c, s, e, *_) in rows:
            if int(e) > int(s):
                grouped.setdefault(str(c), []).append((int(s), int(e)))
        self.d = {}
        for (c, a) in grouped.items():
            merged = []
            for (s, e) in sorted(a):
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(e, merged[-1][1])
                else:
                    merged.append([s, e])
            self.d[c] = np.array(merged, dtype=np.int64)

    def overlaps(self, c, s, e):
        a = self.d.get(c)
        if a is None:
            return np.empty((0, 2), dtype=np.int64)
        x = a[np.searchsorted(a[:, 1], s, side='right'):np.searchsorted(a[:, 0], e, side='left')].copy()
        if len(x):
            x[:, 0] = np.maximum(x[:, 0], s)
            x[:, 1] = np.minimum(x[:, 1], e)
        return x

    def rows(self):
        return [(c, int(s), int(e)) for c in sorted(self.d) for (s, e) in self.d[c]]

    def write(self, path):
        with open(path, 'w') as f:
            for row in self.rows():
                f.write('\t'.join(map(str, row)) + '\n')

    def bp(self):
        return sum((e - s for (c, s, e) in self.rows()))

def intervals_bed(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as f:
        return intervals_Intervals([x.split()[:3] for x in f if x.strip() and (not x.startswith(('#', 'chrom\t')))])

def intervals_edges_for(s, e, rounding='round'):
    body = np.linspace(s, e, 21)
    body = (np.rint(body) if rounding == 'round' else np.floor(body)).astype(int)
    return np.r_[np.arange(s - 2000, s, 50), body, np.arange(e + 50, e + 2001, 50)]

def intervals_integrate_tracks(bws, chrom, edges, mask, positive=False):
    """Integrate exact piecewise-constant segments on the same retained bases.

    Return per-track integrals and covered lengths, a per-base replicate mean
    integral/length, and analysable geometric lengths for every original bin.
    Missing raw coverage is handled as zero only by the R-ChIP caller.
    """
    (s, e) = (int(edges[0]), int(edges[-1]))
    ivs = []
    breaks = [edges]
    excluded = mask.overlaps(chrom, s, e)
    if len(excluded):
        breaks.append(excluded.ravel())
    for bw in bws:
        end = min(e, bw.chroms().get(chrom, 0))
        start = max(s, 0)
        raw = bw.intervals(chrom, start, end) if end > start else None
        a = np.asarray(raw or [], float).reshape(-1, 3)
        if len(a):
            a = a[np.isfinite(a[:, 2]) & (a[:, 2] > 0 if positive else np.ones(len(a), bool))]
            if len(a):
                a[:, 0] = np.maximum(a[:, 0], s)
                a[:, 1] = np.minimum(a[:, 1], e)
                breaks.append(a[:, :2].ravel())
        ivs.append(a)
    z = np.unique(np.concatenate(breaks))
    left = z[:-1]
    length = np.diff(z)
    keep = length > 0
    if len(excluded):
        j = np.searchsorted(excluded[:, 0], left, side='right') - 1
        valid = j >= 0
        keep[valid] &= left[valid] >= excluded[j[valid], 1]
    keep &= (left >= 0) & (z[1:] <= min((bw.chroms().get(chrom, 0) for bw in bws)))
    weights = length * keep
    binid = np.clip(np.searchsorted(edges, left, side='right') - 1, 0, len(edges) - 2)
    sums = []
    counts = []
    allvalues = []
    for a in ivs:
        val = np.full(len(left), np.nan)
        if len(a):
            j = np.searchsorted(a[:, 0], left, side='right') - 1
            v = j >= 0
            v[v] = left[v] < a[j[v], 1]
            val[v] = a[j[v], 2]
        ok = np.isfinite(val)
        sums.append(np.bincount(binid, weights=np.where(ok, val, 0) * weights, minlength=len(edges) - 1))
        counts.append(np.bincount(binid, weights=ok * weights, minlength=len(edges) - 1))
        allvalues.append(val)
    vals = np.asarray(allvalues)
    n = np.isfinite(vals).sum(axis=0)
    avg = np.divide(np.nansum(vals, axis=0), n, out=np.zeros(len(left)), where=n > 0)
    return (np.asarray(sums), np.asarray(counts), np.bincount(binid, weights=avg * weights, minlength=len(edges) - 1), np.bincount(binid, weights=(n > 0) * weights, minlength=len(edges) - 1), np.bincount(binid, weights=weights, minlength=len(edges) - 1))

def intervals_ratio(a, b):
    return np.divide(a, b, out=np.full(np.shape(a), np.nan, dtype=float), where=np.asarray(b) > 0)

def intervals_pairwise(d, signal):
    rows = []
    for (l, r) in combinations(intervals_GROUPS, 2):
        x = d.loc[d.gene_set.eq(l), signal].dropna().to_numpy()
        y = d.loc[d.gene_set.eq(r), signal].dropna().to_numpy()
        (u, p) = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic')
        rows.append(dict(left_gene_set=l, right_gene_set=r, left_n=len(x), right_n=len(y), left_median=np.median(x), right_median=np.median(y), median_fold_left_over_right=np.median(x) / np.median(y), mannwhitney_u=u, p_two_sided_asymptotic=p))
    return pd.DataFrame(rows)

def intervals_profile_summary(d, mat):
    rows = []
    for group in intervals_GROUPS:
        a = mat[d.gene_set.eq(group).to_numpy()]
        for j in range(100):
            x = -2000 + (j + 0.5) * 50 if j < 40 else (j - 40 + 0.5) * 50 if j < 60 else 1000 + (j - 60 + 0.5) * 50
            good = a[:, j][np.isfinite(a[:, j])]
            rows.append(dict(gene_set=group, bin_index=j, profile_x_bp=x, display_x=x / 2 if x < 0 else x if x <= 1000 else 1000 + (x - 1000) / 2, mean_fold_enrichment=good.mean() if len(good) else np.nan, n_genes=len(good), cohort_n=len(a)))
    return pd.DataFrame(rows)

def _prepare_runtime():
    sys.path.insert(0, str(intervals_ROOT / 'final_result/fig3'))
