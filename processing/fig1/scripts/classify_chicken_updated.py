"""Consolidated final processing: classify_chicken_updated.

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
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path
import csv
import gzip
import hashlib
import json
import re
import subprocess
import zipfile
from collections import defaultdict
import numpy as np
import pandas as pd
import pyBigWig
peaks_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/classify_chicken_snrna.py').parents[1]
peaks_RAW = peaks_ROOT / 'data/raw'
peaks_TMP = peaks_ROOT / 'tmp/chicken_fig1_20260924'
peaks_OUT = peaks_ROOT / 'results/chicken_snrna_GRCg7b_complement_20260924'
peaks_GTF = peaks_RAW / 'Gallus_gallus.GRCg7b.115.gtf.gz'
peaks_PEAKZIP = peaks_RAW / 'Chicken_FAANG_Peak_Calls.zip'
peaks_CHAIN = peaks_RAW / 'galGal6ToGRCg7b.over.chain.gz'
peaks_ALIAS = peaks_RAW / 'GCF_016699485.2.chromAlias.txt'
peaks_FAMILY_POL2 = {'U1', 'U2', 'U4', 'U5', 'U11', 'U12', 'U4ATAC'}
peaks_FAMILY_U6 = {'U6', 'U6ATAC'}
peaks_sha = _shared.sha256_file

def peaks_parse_gtf() -> list[dict]:
    rows = []
    with gzip.open(peaks_GTF, 'rt') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9 or fields[2] != 'gene':
                continue
            attr = dict(re.findall('(\\w+) "([^"]+)"', fields[8]))
            if attr.get('gene_biotype') != 'snRNA':
                continue
            chrom = fields[0]
            name = attr.get('gene_name', 'UNNAMED').upper()
            fam = re.sub('[^A-Z0-9]', '', name)
            fam = fam if fam in peaks_FAMILY_POL2 | peaks_FAMILY_U6 | {'U7'} else 'UNNAMED'
            rows.append(dict(gene_id=attr['gene_id'], gene_name=name, family=fam, chrom=chrom, start=int(fields[3]) - 1, end=int(fields[4]), strand=fields[6], autosomal=chrom.isdigit() and 1 <= int(chrom) <= 39))
    assert len(rows) == len({r['gene_id'] for r in rows})
    rows.sort(key=lambda r: (int(r['chrom']) if r['chrom'].isdigit() else 100, r['start'], r['gene_id']))
    return rows

def peaks_read_peak_member(zf: zipfile.ZipFile, name: str) -> dict[str, list[tuple[int, int]]]:
    rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with zf.open(name) as handle:
        for raw in handle:
            fields = raw.decode().rstrip('\n').split('\t')
            if len(fields) >= 3 and fields[0].startswith('chr'):
                rows[fields[0]].append((int(fields[1]), int(fields[2])))
    for v in rows.values():
        v.sort()
    return rows

def peaks_consensus(a: dict, b: dict) -> list[tuple[str, int, int]]:
    result = []
    for chrom in sorted(set(a) & set(b)):
        (left, right) = (a[chrom], b[chrom])
        i = j = 0
        spans = []
        while i < len(left) and j < len(right):
            (s, e) = (max(left[i][0], right[j][0]), min(left[i][1], right[j][1]))
            if s < e:
                if spans and s <= spans[-1][1]:
                    spans[-1] = (spans[-1][0], max(e, spans[-1][1]))
                else:
                    spans.append((s, e))
            if left[i][1] <= right[j][1]:
                i += 1
            else:
                j += 1
        result.extend(((chrom, s, e) for (s, e) in spans))
    return result

def peaks_prepare_peaks() -> dict[str, int]:
    peaks_TMP.mkdir(parents=True, exist_ok=True)
    peaks_OUT.mkdir(parents=True, exist_ok=True)
    records = []
    summary = {}
    with zipfile.ZipFile(peaks_PEAKZIP) as zf:
        names = set(zf.namelist())
        tissues = sorted(set(re.findall('ATAC_(.+?)_CA_Peaks.bed', '\n'.join(names))))
        assert 'Testis' in tissues and len(tissues) >= 15
        for (assay, selected) in (('ATAC', tissues), ('H3K4me3', ['Testis']), ('H3K27ac', ['Testis'])):
            for tissue in selected:
                stem = f'Peak_Calls/{assay}_{tissue}_'
                (a_name, b_name) = (stem + 'CA_Peaks.bed', stem + 'CB_Peaks.bed')
                assert a_name in names and b_name in names, (a_name, b_name)
                hits = peaks_consensus(peaks_read_peak_member(zf, a_name), peaks_read_peak_member(zf, b_name))
                summary[f'{assay}:{tissue}'] = len(hits)
                records.extend(((chrom, start, end, f'{assay}|{tissue}|{i}') for (i, (chrom, start, end)) in enumerate(hits, 1)))
    for (tissue, samples) in (('HD3', (5035, 5036)), ('eRBC', (5037, 5038)), ('aRBC', (5039, 5040))):
        found = []
        for n in samples:
            path = next(peaks_RAW.glob(f'GSM713{n}_*PolIISer5_peaks.narrowPeak.gz'))
            rows = defaultdict(list)
            with gzip.open(path, 'rt') as handle:
                for line in handle:
                    f = line.split('\t')
                    rows['chr' + f[0]].append((int(f[1]), int(f[2])))
            for spans in rows.values():
                spans.sort()
            found.append(rows)
        hits = peaks_consensus(*found)
        summary[f'PolII_Ser5P:{tissue}'] = len(hits)
        records.extend(((chrom, start, end, f'PolII_Ser5P|{tissue}|{i}') for (i, (chrom, start, end)) in enumerate(hits, 1)))
    bed = peaks_TMP / 'consensus_peaks.galGal6.bed'
    with bed.open('w') as handle:
        for (chrom, start, end, name) in records:
            handle.write(f'{chrom}\t{start}\t{end}\t{name}\n')
    with gzip.open(peaks_CHAIN, 'rb') as source, (peaks_TMP / 'galGal6ToGRCg7b.chain').open('wb') as target:
        import shutil
        shutil.copyfileobj(source, target)
    mapped = peaks_TMP / 'consensus_peaks.GRCg7b.refseq.bed'
    unmapped = peaks_TMP / 'consensus_peaks.unmapped.bed'
    subprocess.run(['liftOver', '-minMatch=0.95', str(bed), str(peaks_TMP / 'galGal6ToGRCg7b.chain'), str(mapped), str(unmapped)], check=True)
    summary['all_input_consensus_peaks'] = len(records)
    summary['lifted_peaks'] = sum((1 for _ in mapped.open()))
    summary['unmapped_peaks'] = sum((1 for line in unmapped.open() if not line.startswith('#')))
    return summary

def peaks_lifted_peak_index() -> dict[str, list[tuple[int, int, str, str]]]:
    alias = {}
    with peaks_ALIAS.open() as handle:
        next(handle)
        for line in handle:
            a = line.rstrip('\n').split('\t')
            if len(a) >= 4:
                alias[a[0]] = a[3]
    index = defaultdict(list)
    bad = Counter()
    with (peaks_TMP / 'consensus_peaks.GRCg7b.refseq.bed').open() as handle:
        for line in handle:
            f = line.rstrip('\n').split('\t')
            chrom = alias.get(f[0])
            if chrom is None:
                bad[f[0]] += 1
                continue
            (assay, tissue, _) = f[3].split('|')
            index[chrom].append((int(f[1]), int(f[2]), assay, tissue))
    for (chrom, spans) in index.items():
        spans.sort()
        starts = [p[0] for p in spans]
        max_ends = []
        current = -1
        for p in spans:
            current = max(current, p[1])
            max_ends.append(current)
        index[chrom] = (spans, starts, max_ends)
    (peaks_OUT / 'unaliased_lifted_chromosomes.json').write_text(json.dumps(bad, indent=2) + '\n')
    return index

def peaks_overlaps(index: dict, chrom: str, start: int, end: int) -> list[tuple]:
    (spans, starts, max_ends) = index.get(chrom, ([], [], []))
    i = bisect_left(starts, end) - 1
    result = []
    while i >= 0 and max_ends[i] > start:
        if spans[i][1] > start:
            result.append(spans[i])
        i -= 1
    return result

def peaks_write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
evidence_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/finalize_chicken_snrna.py').parents[1]
evidence_TMP = evidence_ROOT / 'tmp/chicken_fig1_20260924'
evidence_RAW = evidence_ROOT / 'data/raw'
evidence_OUT = evidence_ROOT / 'results/chicken_snrna_GRCg7b_complement_20260924'
evidence_POL2_FAMILIES = {'U1', 'U2', 'U4', 'U5', 'U11', 'U12', 'U4ATAC'}
evidence_U6_FAMILIES = {'U6', 'U6ATAC'}
evidence_CHROMHMM_STATES = ['TssA', 'TssAHet', 'TxFlnk', 'TxFlnkWk', 'TxFlnkHet', 'EnhA', 'EnhAMe', 'EnhAWk', 'EnhAHet', 'EnhPois', 'ATAC_Is', 'TssBiv', 'Repr', 'ReprWk', 'Qui']

def evidence_signal(bw, chrom: str, start: int, end: int) -> tuple[float, int]:
    if chrom not in bw.chroms() or end > bw.chroms(chrom):
        return (0.0, 0)
    values = np.nan_to_num(np.abs(bw.values(chrom, start, end, numpy=True)), nan=0.0, posinf=0.0)
    return (float(values.sum()), int((values > 0).sum()))

def evidence_proseq_table(rows: pd.DataFrame, old: dict) -> pd.DataFrame:
    tracks = {}
    for (rep, gsm) in ((1, 'GSM6716751'), (2, 'GSM6716752')):
        for strand in ('pos', 'neg'):
            path = evidence_RAW / f'{gsm}_PRO-BSA-Chicken-{rep}.{strand}.bigwig'
            assert path.is_file() and path.stat().st_size > 1000000, path
            tracks[rep, strand] = pyBigWig.open(str(path))
    result = []
    for r in rows.itertuples():
        loc = old.get(r.gene_id)
        rec = dict(gene_id=r.gene_id, proseq_one_way_mapped=bool(loc), proseq_old_chrom=loc[0] if loc else '', proseq_old_start=loc[1] if loc else None, proseq_old_end=loc[2] if loc else None, proseq_old_strand=loc[3] if loc else '')
        for rep in (1, 2):
            if loc:
                (chrom, s, e, strand) = loc
                sense = 'pos' if strand == '+' else 'neg'
                anti = 'neg' if strand == '+' else 'pos'
                (ss, bp) = evidence_signal(tracks[rep, sense], chrom, s, e)
                (aa, _) = evidence_signal(tracks[rep, anti], chrom, s, e)
            else:
                ss = aa = 0.0
                bp = 0
            rec[f'proseq_rep{rep}_sense_sum_RPKM'] = ss
            rec[f'proseq_rep{rep}_sense_nonzero_bp'] = bp
            rec[f'proseq_rep{rep}_antisense_sum_RPKM'] = aa
        rec['proseq_dual_rep_sense_support'] = bool(loc and all((rec[f'proseq_rep{rep}_sense_nonzero_bp'] >= 3 and rec[f'proseq_rep{rep}_sense_sum_RPKM'] > rec[f'proseq_rep{rep}_antisense_sum_RPKM'] for rep in (1, 2))))
        result.append(rec)
    for bw in tracks.values():
        bw.close()
    return pd.DataFrame(result)

def evidence_chromhmm_table(rows: pd.DataFrame, old: dict) -> pd.DataFrame:
    """Lift relevant galGal6 chromatin-state slices to GRCg7b."""
    pd.DataFrame([dict(state=f'E{i}', name=name) for (i, name) in enumerate(evidence_CHROMHMM_STATES, 1)]).to_csv(evidence_OUT / 'chromhmm_state_key.tsv', sep='\t', index=False)
    windows = defaultdict(list)
    by_id = rows.set_index('gene_id')
    for (gene_id, (chrom, s, e, strand)) in old.items():
        tss = s if strand == '+' else e - 1
        windows[chrom].append((max(0, tss - 100), tss + 101, gene_id))
    for v in windows.values():
        v.sort()
    bed = evidence_TMP / 'snrna_chromhmm_old_relevant.bed'
    kept = 0
    selected_states = {f'E{x}' for x in range(1, 13)}
    starts_by_chrom = {chrom: [x[0] for x in v] for (chrom, v) in windows.items()}
    with bed.open('w') as target:
        for num in (1, 2, 3):
            with zipfile.ZipFile(evidence_RAW / f'Chicken_FAANG_Chromatin_state_{num}.zip') as zf:
                for name in sorted((n for n in zf.namelist() if n.endswith('_16_segments.bed'))):
                    tissue = Path(name).name.split('_16_segments')[0]
                    with zf.open(name) as handle:
                        for raw in handle:
                            f = raw.decode().rstrip('\n').split('\t')
                            if len(f) < 4 or f[0] not in windows or f[3] not in selected_states:
                                continue
                            (s, e) = (int(f[1]), int(f[2]))
                            w = windows[f[0]]
                            starts = starts_by_chrom[f[0]]
                            for (a, b, gene_id) in w[:bisect_left(starts, e)]:
                                (lo, hi) = (max(s, a), min(e, b))
                                if lo < hi:
                                    target.write(f'{f[0]}\t{lo}\t{hi}\t{gene_id}|{tissue}|{f[3]}\n')
                                    kept += 1
    mapped = evidence_TMP / 'snrna_chromhmm_active.GRCg7b.refseq.bed'
    subprocess.run(['liftOver', '-minMatch=0.95', str(bed), str(evidence_TMP / 'galGal6ToGRCg7b.chain'), str(mapped), str(evidence_TMP / 'snrna_chromhmm_active.unmapped.bed')], check=True)
    alias = {}
    with (evidence_RAW / 'GCF_016699485.2.chromAlias.txt').open() as handle:
        next(handle)
        for line in handle:
            f = line.rstrip('\n').split('\t')
            alias[f[0]] = f[3]
    evidence = defaultdict(set)
    details = []
    for line in mapped.read_text().splitlines():
        f = line.split('\t')
        (gene_id, tissue, state) = f[3].split('|')
        if gene_id not in by_id.index:
            continue
        g = by_id.loc[gene_id]
        chrom = alias.get(f[0], '')
        tss = int(g.start) if g.strand == '+' else int(g.end) - 1
        if chrom != str(g.chrom) or not (int(f[1]) < tss + 101 and int(f[2]) > tss - 100):
            continue
        evidence[gene_id].add((tissue, state))
        details.append(dict(gene_id=gene_id, tissue=tissue, state=state, chrom=chrom, start=int(f[1]), end=int(f[2])))
    pd.DataFrame(details, columns=['gene_id', 'tissue', 'state', 'chrom', 'start', 'end']).to_csv(evidence_OUT / 'chromhmm_lifted_promoter_evidence.tsv', sep='\t', index=False)
    table = []
    for r in rows.itertuples():
        tuples = evidence[r.gene_id]
        active = sorted({t for (t, s) in tuples if s in {'E1', 'E3', 'E6', 'E7', 'E8', 'E11'}})
        promoter = sorted({t for (t, s) in tuples if s in {'E1', 'E3'}})
        bivalent = sorted({t for (t, s) in tuples if s == 'E12'})
        table.append(dict(gene_id=r.gene_id, chromhmm_active_tissues=','.join(active), chromhmm_active_n_tissues=len(active), chromhmm_promoter_tissues=','.join(promoter), chromhmm_promoter_n_tissues=len(promoter), chromhmm_bivalent_n_tissues=len(bivalent), chromhmm_all_states=';'.join((f'{t}:{s}' for (t, s) in sorted(tuples)))))
    print(f'ChromHMM relevant slices: {kept}; lifted valid: {len(details)}', flush=True)
    return pd.DataFrame(table)

def evidence_finalize(rows: pd.DataFrame) -> pd.DataFrame:
    for (i, r) in rows.iterrows():
        if not r.autosomal:
            (group, why) = ('excluded_non_autosomal', 'Z_W_or_noncanonical')
        elif r.family == 'U7':
            (group, why) = ('excluded_U7', 'nonsplice_U7_family')
        elif r.family in evidence_POL2_FAMILIES and (r.proseq_dual_rep_sense_support and (r.atac_n_tissues > 0 or r.chromhmm_active_n_tissues > 0 or r.testis_h3k4me3 or r.testis_h3k27ac) or (r.pol2_ser5p_n_tissues > 0 and r.atac_n_tissues > 0)):
            (group, why) = ('pol2_spliceRNUs', 'dual_rep_strand_specific_PROseq_or_Ser5P_plus_ATAC')
        elif r.family in evidence_U6_FAMILIES and (r.atac_n_tissues > 0 or r.chromhmm_active_n_tissues > 0 or r.testis_h3k4me3 or r.testis_h3k27ac):
            (group, why) = ('functional_U6s', 'U6_annotation_and_native_chromatin_support')
        else:
            (group, why) = ('RNU_pseudogenes', 'autosomal_nonU7_Ensembl_snRNA_without_functional_POL2_or_U6_support')
        rows.at[i, 'analysis_group'] = group
        rows.at[i, 'assignment_reason'] = why
    return rows
windows_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/reclassify_chicken_human_windows_20260928.py').parents[1]
windows_BASE = windows_ROOT / 'data/processed/fig1_species_AF_restored_20260928'
windows_OLD_STEM = 'figure1_seven_species_Arial180x160_species_AF_chimp_per_group_20260928'
windows_GROUPS = ('pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes', 'genome_wide_neutral')
windows_NAMES = ('panel_a_gene_locus_density', 'panel_a_pairwise_statistics', 'panel_b_control_mean_ratios', 'panel_c_profile_control_ratios', 'control_denominators')
windows_WINDOWS = {'regulatory': (-50, 50), 'PolII_Ser5P': (-50, 400), 'H3K27ac': (-400, 200), 'H3K4me3': (-1000, 200)}
windows_ACTIVE = {'E1', 'E3', 'E6', 'E7', 'E8', 'E11'}
windows_read = _shared.read_table

def windows_window(r, target):
    t = int(r['start']) if r['strand'] == '+' else int(r['end'])
    (a, b) = windows_WINDOWS[target]
    return (max(0, t + a), t + b) if r['strand'] == '+' else (max(0, t - b), t - a)
classification_initial = _component('classify_chicken_snrna')
classification_original = _component('finalize_chicken_snrna')
classification_spatial = _component('reclassify_chicken_human_windows_20260928')
classification_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/reclassify_chicken_regulatory_windows_20260928.py').parents[1]
classification_TMP = classification_ROOT / 'tmp/chicken_regulatory_windows_20260928'
classification_RESULT = classification_ROOT / 'results/chicken_regulatory_windows_20260928'
classification_OUT = classification_ROOT / 'data/processed/fig1_chicken_regulatory_windows_20260928'
classification_BASE = classification_spatial.BASE
classification_STEM = 'figure1_seven_species_Arial180x160_chicken_regulatory_windows_20260928'
classification_OLD_STEM = classification_spatial.OLD_STEM
(classification_GROUPS, classification_NAMES, classification_read) = (classification_spatial.GROUPS, classification_spatial.NAMES, classification_spatial.read)
classification_OLD_CAT = classification_ROOT / 'results/chicken_snrna_GRCg7b_complement_20260924/all_ensembl_snRNA_82.tsv'

def classification_retained_state_intervals(rows):
    """Change the association window on the original mapped evidence only."""
    alias = {}
    for line in classification_initial.ALIAS.read_text().splitlines()[1:]:
        f = line.split('\t')
        if len(f) >= 4:
            alias[f[0]] = f[3]
    byid = {r['gene_id']: r for r in rows}
    states = {g: set() for g in byid}
    details = []
    source = classification_initial.TMP / 'snrna_chromhmm_active.GRCg7b.refseq.bed'
    for line in source.read_text().splitlines():
        f = line.split('\t')
        (gene, tissue, state) = f[3].split('|')
        if state not in classification_spatial.ACTIVE or gene not in byid:
            continue
        r = byid[gene]
        (s, e) = classification_spatial.window(r, 'regulatory')
        (a, b) = (int(f[1]), int(f[2]))
        if alias.get(f[0]) == r['chrom'] and a < e and (b > s):
            states[gene].add((tissue, state))
            details.append(dict(gene_id=gene, tissue=tissue, state=state, chrom=r['chrom'], start=a, end=b, overlap_bp=min(e, b) - max(s, a)))
    pd.DataFrame(details).to_csv(classification_RESULT / 'chromatin_state_TSS50_evidence.tsv', sep='\t', index=False)
    return states

def classification_main():
    for p in (classification_TMP, classification_RESULT, classification_OUT):
        p.mkdir(parents=True, exist_ok=True)
    contract = dict(scope='Regulatory evidence windows only; original chicken PolII classifier retained', regulatory_windows_relative_to_TSS={'ATAC': [-50, 50], 'active_chromatin_states': [-50, 50], 'H3K27ac': [-400, 200], 'H3K4me3': [-1000, 200]}, original_PolII_Ser5P_window='gene body extended by 100 bp on both sides', original_PROseq_rule='Both replicates: >=3 nonzero sense bases and summed sense > antisense', original_PolII_rule='PolII-family AND ((PROseq AND any chromatin support) OR (Ser5P AND ATAC))', U6_rule='U6/U6ATAC annotation AND any chromatin support', pseudogene_rule='Complement among all autosomal non-U7 Ensembl snRNAs', mean_FC_gt2_applied=False, coordinate_system='GRCg7b; zero-based half-open; strand-oriented regulatory windows', human_window_sources=['User human cCRE TSS +/-50 criterion', '/public5/home/xuejing/RNU/codex/gene_family/scripts/extract_processed_signals.py'], liftOver_minMatch=0.95, chicken_AF=0.01, controls='Existing 1000, no resampling', chromatin_state_source='Original mapped state intervals; only the overlap window changes', user_correction='Do not change PolII conditions; update cCRE-related multiomics windows only')
    (classification_RESULT / 'parameter_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    old = pd.read_csv(classification_OLD_CAT, sep='\t', dtype={'chrom': str}, float_precision='round_trip')
    rows = classification_initial.parse_gtf()
    classification_initial.OUT = classification_RESULT
    peaks = classification_initial.lifted_peak_index()
    states = classification_retained_state_intervals(rows)
    new = old.copy()
    details = []
    for (i, r) in new.iterrows():
        for (assay, target, col) in [('ATAC', 'regulatory', 'atac'), ('H3K27ac', 'H3K27ac', 'testis_h3k27ac'), ('H3K4me3', 'H3K4me3', 'testis_h3k4me3')]:
            (s, e) = classification_spatial.window(r, target)
            hits = [p for p in classification_initial.overlaps(peaks, r.chrom, s, e) if p[2] == assay]
            tissues = sorted({p[3] for p in hits})
            if assay == 'ATAC':
                new.at[i, 'atac_tissues'] = ','.join(tissues)
                new.at[i, 'atac_n_tissues'] = len(tissues)
                new.at[i, 'testis_atac'] = 'Testis' in tissues
            else:
                new.at[i, col] = 'Testis' in tissues
            for (a, b, _, tissue) in hits:
                details.append(dict(gene_id=r.gene_id, assay=assay, tissue=tissue, chrom=r.chrom, window_start=s, window_end=e, peak_start=a, peak_end=b, overlap_bp=min(e, b) - max(s, a)))
        st = states[r.gene_id]
        tissues = sorted({t for (t, _) in st})
        promoters = sorted({t for (t, s) in st if s in {'E1', 'E3'}})
        new.at[i, 'chromhmm_active_tissues'] = ','.join(tissues)
        new.at[i, 'chromhmm_active_n_tissues'] = len(tissues)
        new.at[i, 'chromhmm_promoter_tissues'] = ','.join(promoters)
        new.at[i, 'chromhmm_promoter_n_tissues'] = len(promoters)
        new.at[i, 'chromhmm_active_states'] = ';'.join((f'{t}:{s}' for (t, s) in sorted(st)))
        new.at[i, 'regulatory_tss_boundary'] = int(r.start) if r.strand == '+' else int(r.end)
    new = new.drop(columns=['chromhmm_bivalent_n_tissues', 'chromhmm_all_states'])
    new['testis_ccre_like'] = new.testis_atac & new.testis_h3k4me3 & new.testis_h3k27ac
    frozen = [c for c in old.columns if c.startswith(('proseq_', 'pol2_ser5p_'))]
    pd.testing.assert_frame_equal(new[frozen], old[frozen], check_exact=True)
    new = classification_original.finalize(new)
    new.to_csv(classification_RESULT / 'all_ensembl_snRNA_82.tsv', sep='\t', index=False)
    pd.DataFrame(details).to_csv(classification_RESULT / 'assay_window_peak_evidence.tsv', sep='\t', index=False)
    diff = new.merge(old[['gene_id', 'analysis_group']], on='gene_id', suffixes=('_new', '_old'))
    diff['membership_changed'] = diff.analysis_group_new.ne(diff.analysis_group_old)
    diff.to_csv(classification_RESULT / 'membership_comparison.tsv', sep='\t', index=False)
    evidence_compare = []
    for key in ['atac_n_tissues', 'testis_h3k4me3', 'testis_h3k27ac', 'chromhmm_active_n_tissues']:
        for i in range(len(new)):
            if old.loc[i, key] != new.loc[i, key]:
                evidence_compare.append(dict(gene_id=new.loc[i, 'gene_id'], field=key, old=old.loc[i, key], new=new.loc[i, key]))
    pd.DataFrame(evidence_compare).to_csv(classification_RESULT / 'regulatory_evidence_changes.tsv', sep='\t', index=False)
    summary = dict(counts=new.groupby('analysis_group').size().to_dict(), original_PolII_signal_columns_exactly_preserved=True, original_classifier_function_reused='finalize_chicken_snrna.finalize', changed_gene_ids=diff.loc[diff.membership_changed, 'gene_id'].tolist())
    (classification_RESULT / 'classification_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

def classification_entrypoint():
    classification_main()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    _run_stages({'classification': 'classification_entrypoint'}, globals(), setup=_prepare_runtime)
