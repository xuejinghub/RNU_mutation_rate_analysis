"""Consolidated final processing: analyze_chimpanzee_updated.

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
import gzip
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import argparse
import hashlib
import numpy as np
import pysam
from scipy.stats import mannwhitneyu
from types import SimpleNamespace
mapping_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/map_chimpanzee_all_rnu_macaque_20260928.py').parents[1]
mapping_RESULT = mapping_ROOT / 'results/chimpanzee_all_rnu_macaque_synteny_20260928'
mapping_TMP = mapping_ROOT / 'tmp/chimpanzee_all_rnu_macaque_synteny_20260928'
mapping_SOURCE = mapping_ROOT / 'primate/data/processed/fig1_chimpanzee_af0p02_20260923'
mapping_RAW = mapping_ROOT / 'primate/data/raw/chimpanzee_PanTro5'
mapping_OLD_SYN = mapping_ROOT / 'primate/results/human_chimpanzee_pol2_synteny_20260923'
mapping_AN = mapping_ROOT.parent / 'AN_validation'
mapping_MACSCRIPT = _component_file('analyze_rnu_pol2_orthology')
mapping_MASTER = mapping_ROOT / 'primate/transpecies_compare/rnu_three_gene_sets_20260810/three_gene_set_master_membership_1901.tsv'
mapping_HGTF = mapping_ROOT / 'primate/transpecies_compare/gencode.v44.annotation.gtf.gz'
mapping_HFA = Path('/public5/home/xuejing/human_hg38/coverage/hg38.fa')
mapping_GROUPS = ['pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes']
mapping_CHAIN = mapping_RAW / 'hg38ToPanTro5.over.chain.gz'
mapping_mac = _component('analyze_rnu_pol2_orthology')
mapping_read = mapping_mac.read_tsv
mapping_write = mapping_mac.write_tsv

def mapping_sha(path):
    return mapping_mac.file_sha256(Path(path))

def mapping_family(name):
    value = str(name).upper()
    for f in ['U6ATAC', 'U4ATAC', 'U11', 'U12', 'U1', 'U2', 'U4', 'U5', 'U6', 'U7']:
        if value.startswith(('RN' + f, 'RNV' + f)) or re.search('(?:^|[^A-Z0-9])' + f + '(?![A-Z0-9])', value):
            return f
    return 'UNKNOWN'

def mapping_choose(targets, candidates, core, flanks, extended, blast):
    """Same core/flank/extended/sequence tiers and reuse rule as macaque."""
    byquery = defaultdict(list)
    bypair = {}
    for r in blast:
        if r['pident'] < 65 or r['query_coverage'] < 0.6:
            continue
        if r['family'] == r['human_family']:
            byquery[r['human_gene_id']].append(r)
        key = (r['human_gene_id'], r['monkey_gene_id'])
        if key not in bypair or r['bitscore'] > bypair[key]['bitscore']:
            bypair[key] = r
    out = []
    cby = {c['monkey_gene_id']: c for c in candidates}
    for target in targets:
        hid = target['stable_gene_id']
        chosen = []
        method = ''
        confidence = ''
        relationship = ''
        mapping = core.get(hid)
        if mapping:
            direct = [h for h in mapping_mac.candidate_overlaps(mapping, target['family'], candidates) if h['reciprocal_overlap'] >= 0.3]
            if direct:
                best = direct[0]['reciprocal_overlap']
                chosen = [h for h in direct if h['reciprocal_overlap'] >= best - 0.05]
                method = 'core liftOver minMatch=0.20 + same-family overlap'
                confidence = 'high' if best >= 0.7 else 'moderate'
                relationship = 'one-to-one candidate' if len(chosen) == 1 else 'array-level co-ortholog candidates'
        if not chosen:
            for (inner, outer) in mapping_mac.FLANK_SCHEMES:
                mapping = flanks[inner, outer].get(hid)
                if not mapping:
                    continue
                near = mapping_mac.nearest_candidates(mapping, target['family'], candidates, mapping_mac.MAX_NEAREST_BP)
                if near:
                    shortlist = [h for h in near if h['distance_bp'] <= near[0]['distance_bp'] + 2000]
                    supported = [h for h in shortlist if (hid, h['monkey_gene_id']) in bypair]
                    chosen = supported if supported else shortlist[:1]
                    method = f'independent flank-anchor liftOver ({inner:,}–{outer:,} bp) + nearest same-family annotation'
                    confidence = 'moderate' if supported and len(flanks[inner, outer]) else 'low'
                    relationship = 'one-to-one candidate' if len(chosen) == 1 else 'array-level co-ortholog candidates'
                    break
        if not chosen:
            for pad in mapping_mac.EXTENDED_PADS:
                mapping = extended[pad].get(hid)
                if not mapping:
                    continue
                near = mapping_mac.nearest_candidates(mapping, target['family'], candidates, mapping_mac.MAX_NEAREST_BP)
                if near:
                    shortlist = [h for h in near if h['distance_bp'] <= near[0]['distance_bp'] + 2000]
                    supported = [h for h in shortlist if (hid, h['monkey_gene_id']) in bypair]
                    chosen = supported if supported else shortlist[:1]
                    method = f'extended-locus liftOver ±{pad:,} bp + nearest same-family annotation'
                    confidence = 'moderate' if supported else 'low'
                    relationship = 'one-to-one candidate' if len(chosen) == 1 else 'array-level co-ortholog candidates'
                    break
        if not chosen:
            hits = byquery.get(hid, [])
            if hits:
                best = hits[0]['bitscore']
                ids = []
                for h in hits:
                    if h['bitscore'] >= best * 0.98 and h['monkey_gene_id'] not in ids:
                        ids.append(h['monkey_gene_id'])
                chosen = [cby[x] for x in ids[:10] if x in cby]
                method = 'relaxed same-family BLASTN only (no positional support)'
                confidence = 'low'
                relationship = 'family-level homolog candidates; orthology unresolved'
                mapping = None
        evidence = [bypair[hid, c['monkey_gene_id']] for c in chosen if (hid, c['monkey_gene_id']) in bypair]
        bestseq = max(evidence, key=lambda x: x['bitscore'], default=None)
        out.append(dict(human_gene_id=hid, human_display_name=target['display_name'], human_family=target['family'], analysis_group=target.get('analysis_group', ''), human_chrom=target['chrom'], human_start0=target['start'], human_end=target['end'], human_strand=target['strand'], relationship_class=relationship or 'unresolved', confidence=confidence or 'unresolved', selected_method=method or 'no method passed relaxed criteria', monkey_gene_ids=';'.join((c['monkey_gene_id'] for c in chosen)), candidate_count=len(chosen), mapped_chrom=mapping['chrom'] if chosen and mapping else '', mapped_start0=mapping['start0'] if chosen and mapping else '', mapped_end=mapping['end'] if chosen and mapping else '', mapped_strand=mapping['strand'] if chosen and mapping else '', best_sequence_identity_pct=bestseq['pident'] if bestseq else '', best_sequence_query_coverage=bestseq['query_coverage'] if bestseq else ''))
    target_sources = defaultdict(set)
    for r in out:
        for target in r['monkey_gene_ids'].split(';'):
            if target:
                target_sources[target].add(r['human_gene_id'])
    reused = {t for (t, ids) in target_sources.items() if len(ids) > 1}
    for r in out:
        targets = [t for t in r['monkey_gene_ids'].split(';') if t]
        r['target_max_source_count'] = max((len(target_sources[t]) for t in targets), default=0)
        r['relationship_before_reuse'] = r['relationship_class']
        if set(targets) & reused and 'orthology unresolved' not in r['relationship_class']:
            r['relationship_class'] = 'many-to-one / array-level co-ortholog candidate'
        r['retained'] = r['relationship_class'] == 'one-to-one candidate'
    return out

def mapping_parity_check():
    """Replay the new selector on the original saved macaque mapping evidence."""
    d = mapping_AN / 'results/rnu_pol2_orthology'
    old = mapping_read(d / 'orthology_calls.tsv')
    targets = [dict(stable_gene_id=r['human_gene_id'], display_name=r['human_display_name'], family=r['human_family'], chrom=r['human_chrom'], start=int(r['human_start0']), end=int(r['human_end']), strand=r['human_strand']) for r in old]
    candidates = mapping_read(d / 'monkey_rnu_candidate_catalog.tsv')
    for r in candidates:
        r['start0'] = int(r['start0'])
        r['end'] = int(r['end'])
        r['center'] = (r['start0'] + r['end']) / 2
    core = {r['human_gene_id']: dict(chrom=r['mapped_chrom'], start0=int(r['mapped_start0']), end=int(r['mapped_end']), strand=r['mapped_strand']) for r in mapping_read(d / 'core_liftover_all_thresholds.tsv') if float(r['min_match']) == 0.2 and r['lifted'] == 'True'}
    flanks = {s: {} for s in mapping_mac.FLANK_SCHEMES}
    extended = {p: {} for p in mapping_mac.EXTENDED_PADS}
    for r in mapping_read(d / 'independent_flank_liftover_all_schemes.tsv'):
        if r['predicted_chrom']:
            flanks[int(r['inner_gap_bp']), int(r['outer_gap_bp'])][r['human_gene_id']] = dict(chrom=r['predicted_chrom'], start0=int(r['predicted_start0']), end=int(r['predicted_end']), strand=r['chain_strand'])
    for r in mapping_read(d / 'extended_liftover_all_pads.tsv'):
        if r['lifted'] == 'True':
            extended[int(r['pad_bp'])][r['human_gene_id']] = dict(chrom=r['mapped_chrom'], start0=int(r['mapped_start0']), end=int(r['mapped_end']), strand=r['mapped_strand'])
    blast = mapping_read(d / 'relaxed_blastn_same_family_hits.tsv')
    for r in blast:
        for k in ['pident', 'query_coverage', 'bitscore']:
            r[k] = float(r[k])
    new = mapping_choose(targets, candidates, core, flanks, extended, blast)
    checks = []
    columns = ['monkey_gene_ids', 'relationship_class', 'selected_method', 'confidence']
    for (a, b) in zip(old, new):
        assert a['human_gene_id'] == b['human_gene_id']
        assert all((a[k] == b[k] for k in columns)), (a, b)
        checks.append(dict(human_gene_id=a['human_gene_id'], display_name=a['human_display_name'], same_target_ids=True, same_relationship=True, same_tier=True, same_confidence=True))
    mapping_write(mapping_RESULT / 'macaque_41_selection_parity.tsv', checks)
    assert sum((r['retained'] for r in new)) == 29
    print('Macaque selection parity: all 41 calls and 29 one-to-one calls exactly reproduced', flush=True)

def mapping_prepare():
    cat = mapping_read(mapping_SOURCE / 'chimpanzee_catalog.tsv')
    human = [r for r in cat if r['analysis_group'] in mapping_GROUPS]
    assert Counter((r['analysis_group'] for r in human)) == dict(zip(mapping_GROUPS, [41, 116, 1323]))
    metadata = {r['stable_gene_id']: r for r in mapping_read(mapping_MASTER)}
    genes = {g.gene_id: g for g in mapping_mac.read_gtf_genes(mapping_HGTF)}
    targets = []
    reconcile = []
    for r in human:
        g = genes[r['unit_id']]
        fam = mapping_family(g.name)
        assert fam != 'UNKNOWN', r
        assert (g.chrom, g.start0, g.end, g.strand) == (r['chrom'], int(r['start']), int(r['end']), r['strand'])
        assert r['display_name'] == g.name
        targets.append(dict(stable_gene_id=r['unit_id'], display_name=r['display_name'], family=fam, chrom=r['chrom'], start=int(r['start']), end=int(r['end']), strand=r['strand'], analysis_group=r['analysis_group']))
        reconcile.append(dict(human_gene_id=r['unit_id'], gene_name=g.name, source_group=r['analysis_group'], prior_table_family=metadata[r['unit_id']]['analysis_family'], annotation_name_family=fam, family_differs=metadata[r['unit_id']]['analysis_family'] != fam))
    for r in targets:
        r['core_interval'] = (r['chrom'], r['start'], r['end'], r['stable_gene_id'], r['strand'])
    mapping_write(mapping_RESULT / 'human_input_1480.tsv', [{k: v for (k, v) in r.items() if k != 'core_interval'} for r in targets])
    mapping_write(mapping_RESULT / 'human_family_reconciliation.tsv', reconcile)
    aliases = {}
    for line in (mapping_RAW / 'panTro5.chromAlias.txt').read_text().splitlines():
        if line.startswith('#'):
            continue
        f = line.split('\t')
        if len(f) > 1 and f[1]:
            aliases[f[1]] = f[0]
    candidates = []
    with gzip.open(mapping_RAW / 'Pan_troglodytes.Pan_tro_3.0.115.gtf.gz', 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            a = line.rstrip().split('\t')
            if a[2] != 'gene':
                continue
            at = mapping_mac.parse_attributes(a[8])
            if at.get('gene_biotype') != 'snRNA':
                continue
            (start, end) = (int(a[3]) - 1, int(a[4]))
            name = at.get('gene_name', '')
            candidates.append(dict(monkey_gene_id=at['gene_id'].split('.')[0], chrom=aliases[a[0]], start0=start, end=end, center=(start + end) / 2, symbol=name, strand=a[6], biotype='snRNA', family=mapping_family(name)))
    candidates.sort(key=lambda x: (x['chrom'], x['start0'], x['end'], x['monkey_gene_id']))
    assert len(candidates) == len({c['monkey_gene_id'] for c in candidates}) == 1903
    mapping_write(mapping_RESULT / 'chimp_Ensembl115_candidate_catalog.tsv', [{'chimp_gene_id': r['monkey_gene_id'], **{k: v for (k, v) in r.items() if k != 'monkey_gene_id'}} for r in candidates])
    return (targets, candidates, genes)

def mapping_lift(rows, key, label, mm):
    inp = mapping_TMP / f'{label}.hg38.bed'
    mapping_mac.write_bed(inp, rows, key)
    mapped = mapping_TMP / f'{label}.mapped.bed'
    unmapped = mapping_TMP / f'{label}.unmapped.bed'
    cmd = ['liftOver', f'-minMatch={mm}', str(inp), str(mapping_CHAIN), str(mapped), str(unmapped)]
    with (mapping_TMP / 'commands.jsonl').open('a') as f:
        f.write(json.dumps(cmd) + '\n')
    with (mapping_TMP / f'{label}.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=log, check=True)
    out = []
    for line in mapped.read_text().splitlines():
        a = line.split('\t')
        out.append(dict(chrom=a[0], start0=int(a[1]), end=int(a[2]), name=a[3], strand=a[5]))
    assert len({r['name'] for r in out}) == len(out), 'No -multiple: expected one output per query'
    return out

def mapping_mappings(targets, candidates):
    core = {}
    extended = {}
    flanks = {}
    core_rows = []
    extended_rows = []
    flank_rows = []
    for mm in mapping_mac.CORE_THRESHOLDS:
        values = mapping_lift(targets, 'core_interval', f'core_{round(mm * 100):02d}', mm)
        core[mm] = {r['name']: r for r in values}
        for t in targets:
            m = core[mm].get(t['stable_gene_id'])
            core_rows.append(dict(human_gene_id=t['stable_gene_id'], min_match=mm, mapped=bool(m), **m or {}))
    for pad in mapping_mac.EXTENDED_PADS:
        prepared = [dict(t, extended_interval=(t['chrom'], max(0, t['start'] - pad), t['end'] + pad, t['stable_gene_id'], t['strand'])) for t in targets]
        values = mapping_lift(prepared, 'extended_interval', f'extended_{pad}', 0.5)
        extended[pad] = {r['name']: r for r in values}
        for t in targets:
            m = extended[pad].get(t['stable_gene_id'])
            extended_rows.append(dict(human_gene_id=t['stable_gene_id'], pad=pad, mapped=bool(m), **m or {}))
    for (inner, outer) in mapping_mac.FLANK_SCHEMES:
        anchors = []
        hcenters = {}
        acenters = {}
        for t in targets:
            (s, e) = (t['start'], t['end'])
            hid = t['stable_gene_id']
            hcenters[hid] = (s + e) / 2
            for (side, a, b) in [('L', max(0, s - outer), max(0, s - inner)), ('R', e + inner, e + outer)]:
                if b <= a:
                    continue
                name = hid + '|' + side
                acenters[name] = (a + b) / 2
                anchors.append(dict(anchor_interval=(t['chrom'], a, b, name, '+')))
        mapped = mapping_lift(anchors, 'anchor_interval', f'flanks_{inner}_{outer}', 0.5)
        bygene = defaultdict(list)
        for m in mapped:
            (hid, side) = m['name'].split('|')
            delta = hcenters[hid] - acenters[m['name']]
            pred = (m['start0'] + m['end']) / 2 + (delta if m['strand'] == '+' else -delta)
            bygene[hid].append(dict(chrom=m['chrom'], predicted_center=pred, chain_strand=m['strand'], side=side))
        preds = {}
        for t in targets:
            hid = t['stable_gene_id']
            bychr = defaultdict(list)
            for m in bygene[hid]:
                bychr[m['chrom']].append(m)
            if not bychr:
                continue
            (chrom, usable) = max(bychr.items(), key=lambda item: len(item[1]))
            centers = [x['predicted_center'] for x in usable]
            trials = centers + ([sum(centers) / len(centers)] if len(centers) > 1 and max(centers) - min(centers) <= 200000 else [])
            choices = []
            length = t['end'] - t['start']
            for center in trials:
                trial = dict(chrom=chrom, start0=max(0, int(round(center - length / 2))), end=max(1, int(round(center + length / 2))), strand=usable[0]['chain_strand'])
                near = mapping_mac.nearest_candidates(trial, t['family'], candidates, 100000)
                choices.append((near[0]['distance_bp'] if near else 10 ** 18, trial, near))
            (distance, pred, near) = min(choices, key=lambda item: item[0])
            preds[hid] = pred
            flank_rows.append(dict(human_gene_id=hid, inner=inner, outer=outer, mapped_anchor_count=len(bygene[hid]), **pred, nearest_distance_bp=distance if near else ''))
        flanks[inner, outer] = preds
    for (name, rows) in [('core_mapping_audit.tsv', core_rows), ('extended_mapping_audit.tsv', extended_rows), ('flank_prediction_audit.tsv', flank_rows)]:
        fields = list(dict.fromkeys((k for r in rows for k in r)))
        mapping_write(mapping_RESULT / name, rows, fields)
    print('Forward mappings complete:', {mm: len(v) for (mm, v) in core.items()}, flush=True)
    return (core[0.2], flanks, extended)

def mapping_sequences(targets, candidates, genes):
    mapping_mac.HUMAN_FASTA = mapping_HFA
    query = mapping_mac.write_query_fasta(targets, genes, mapping_TMP)
    allowed = {c['monkey_gene_id']: c for c in candidates}
    meta = {}
    subject = mapping_TMP / 'chimp_1903_snRNA.fa'
    qc = mapping_read(mapping_OLD_SYN / 'chimp_genome_transcript_sequence_QC.tsv')
    assert len(qc) == 1903 and all((r['exact_sequence_match'] == 'True' for r in qc))
    seqs = {header.split()[0]: seq for (header, seq) in mapping_mac.read_fasta_records(mapping_ROOT / 'primate/tmp/chimp_synteny_20260923/chimp_snRNA_transcribed_1903.fa')}
    with subject.open('w') as f:
        for c in candidates:
            gid = c['monkey_gene_id']
            seq = seqs[gid]
            assert len(seq) == c['end'] - c['start0']
            f.write(f'>{gid}\n{seq}\n')
            meta[gid] = dict(monkey_gene_id=gid, monkey_transcript_id='', monkey_symbol=c['symbol'], family=c['family'])
    print('BLASTN-short: 1480 human queries against 1903 annotated chimpanzee snRNAs', flush=True)
    blast = mapping_mac.run_blast(query, subject, meta, mapping_TMP)
    relaxed = [r for r in blast if r['pident'] >= 65 and r['query_coverage'] >= 0.6]
    mapping_write(mapping_RESULT / 'relaxed_same_family_blast_hits.tsv', relaxed)
    print('Sequence support complete:', len(relaxed), 'accepted HSP records', flush=True)
    return blast

def mapping_finalise(targets, candidates, calls):
    cby = {c['monkey_gene_id']: c for c in candidates}
    keep = []
    pairs = []
    reuse = defaultdict(list)
    for r in calls:
        ids = [x for x in r['monkey_gene_ids'].split(';') if x]
        for cid in ids:
            c = cby[cid]
            expected = r['mapped_strand']
            if r['selected_method'].startswith('independent flank'):
                expected = r['human_strand'] if r['mapped_strand'] == '+' else '+' if r['human_strand'] == '-' else '-'
            reuse[cid].append(r['human_gene_id'])
            pair = dict(human_gene_id=r['human_gene_id'], human_display_name=r['human_display_name'], human_strand=r['human_strand'], analysis_group=r['analysis_group'], chimp_gene_id=cid, chimp_chrom=c['chrom'], chimp_start0=c['start0'], chimp_end=c['end'], chimp_strand=c['strand'], chimp_family=c['family'], selected_method=r['selected_method'], retained=r['retained'], expected_target_strand=expected, strand_concordant=expected == c['strand'] if expected else '')
            pairs.append(pair)
        row = {'chimp_gene_ids': r['monkey_gene_ids'], **{k: v for (k, v) in r.items() if k != 'monkey_gene_ids'}}
        keep.append(row)
    mapping_write(mapping_RESULT / 'all_1480_synteny_calls.tsv', keep)
    mapping_write(mapping_RESULT / 'selected_candidate_pairs.tsv', pairs)
    mapping_write(mapping_RESULT / 'retained_strand_discordance.tsv', [r for r in pairs if r['retained'] and r['strand_concordant'] is False], list(pairs[0]))
    retained = [r for r in keep if r['retained']]
    mapping_write(mapping_RESULT / 'retained_hg38_genes.tsv', retained)
    mapping_write(mapping_RESULT / 'excluded_hg38_genes.tsv', [r for r in keep if not r['retained']])
    assert len({r['human_gene_id'] for r in retained}) == len(retained)
    assert len({r['chimp_gene_ids'] for r in retained}) == len(retained)
    for group in mapping_GROUPS:
        with (mapping_RESULT / (group + '.retained.hg38.bed')).open('w') as f:
            for r in retained:
                if r['analysis_group'] == group:
                    f.write(f"{r['human_chrom']}\t{r['human_start0']}\t{r['human_end']}\t{r['human_gene_id']}\t0\t{r['human_strand']}\n")
    mapping_write(mapping_RESULT / 'target_reuse_audit.tsv', [dict(chimp_gene_id=k, source_count=len(v), human_gene_ids=';'.join(v)) for (k, v) in sorted(reuse.items())])
    summary = []
    for group in mapping_GROUPS:
        rows = [r for r in keep if r['analysis_group'] == group]
        good = [r for r in rows if r['retained']]
        summary.append(dict(analysis_group=group, input_loci=len(rows), retained_loci=len(good), excluded_loci=len(rows) - len(good), direct_core=sum((r['selected_method'].startswith('core ') for r in good)), flank_rescue=sum((r['selected_method'].startswith('independent flank') for r in good)), extended_rescue=sum((r['selected_method'].startswith('extended-locus') for r in good)), excluded_target_reuse=sum((r['relationship_class'].startswith('many-to-one') for r in rows)), excluded_multiple=sum((r['relationship_class'] == 'array-level co-ortholog candidates' for r in rows)), excluded_sequence_only=sum(('orthology unresolved' in r['relationship_class'] for r in rows)), excluded_unresolved=sum((r['relationship_class'] == 'unresolved' for r in rows))))
    mapping_write(mapping_RESULT / 'group_synteny_summary.tsv', summary)
    policy = dict(status='PASS', human_input_count=1480, target_annotated_snrna_count=1903, source_assemblies=['GRCh38', 'Pan_tro_3.0/panTro5'], figure_variant_coordinates='retained human GRCh38 loci', macaque_41_selection_parity=True, shared_selection_thresholds=True, reverse_liftover_required=False, independent_minimap_required=False, strict_80pct_sequence_gate=False, joint_three_group_target_reuse_check=True, target_reuse_includes_sequence_only_candidates_as_in_original_macaque_code=True, group_counts=summary, retained_strand_discordance=sum((r['retained'] and r['strand_concordant'] is False for r in pairs)), strand_gate='Audited, not added: original macaque candidate selection does not enforce strand', family_extension='U6ATAC and U7 are separate families; longest-family annotation-name parsing includes U11 before U1; source group labels fixed', variant_density_used_in_selection=False)
    (mapping_RESULT / 'synteny_validation.json').write_text(json.dumps(policy, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

def mapping_main():
    for p in [mapping_RESULT, mapping_TMP]:
        p.mkdir(parents=True, exist_ok=True)
    mapping_parity_check()
    (targets, candidates, genes) = mapping_prepare()
    (core, flanks, extended) = mapping_mappings(targets, candidates)
    blast = mapping_sequences(targets, candidates, genes)
    mapping_finalise(targets, candidates, mapping_choose(targets, candidates, core, flanks, extended, blast))
    paths = [mapping_MACSCRIPT, mapping_SOURCE / 'chimpanzee_catalog.tsv', mapping_MASTER, mapping_HGTF, mapping_RAW / 'Pan_troglodytes.Pan_tro_3.0.115.gtf.gz', mapping_RAW / 'panTro5.chromAlias.txt', mapping_CHAIN, mapping_ROOT / 'primate/tmp/chimp_synteny_20260923/chimp_snRNA_transcribed_1903.fa', mapping_OLD_SYN / 'chimp_genome_transcript_sequence_QC.tsv']
    mapping_write(mapping_RESULT / 'input_manifest.tsv', [dict(path=str(p), resolved_path=str(p.resolve()), bytes=p.stat().st_size, sha256=mapping_sha(p)) for p in paths])
    (mapping_RESULT / 'reference_macaque_implementation.py').write_text(mapping_MACSCRIPT.read_text())

def mapping_entrypoint():
    mapping_main()
synteny_engine = _component('map_chimpanzee_all_rnu_macaque_20260928')
synteny_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/scripts/reclassify_chimp_per_group_macaque_20260928.py').parents[1]
synteny_EVIDENCE = synteny_ROOT / 'results/chimpanzee_all_rnu_macaque_synteny_20260928'
synteny_OUT = synteny_ROOT / 'results/chimp_three_sets_macaque_per_group_20260928'
synteny_GROUPS = synteny_engine.GROUPS

def synteny_read(name):
    return pd.read_csv(synteny_EVIDENCE / name, sep='\t', float_precision='round_trip').fillna('')

def synteny_mappings():
    core = {r.human_gene_id: dict(chrom=r.chrom, start0=int(r.start0), end=int(r.end), strand=r.strand) for r in synteny_read('core_mapping_audit.tsv').itertuples() if float(r.min_match) == 0.2 and r.mapped}
    flank = {s: {} for s in synteny_engine.mac.FLANK_SCHEMES}
    for r in synteny_read('flank_prediction_audit.tsv').itertuples():
        flank[int(r.inner), int(r.outer)][r.human_gene_id] = dict(chrom=r.chrom, start0=int(r.start0), end=int(r.end), strand=r.strand)
    extended = {p: {} for p in synteny_engine.mac.EXTENDED_PADS}
    for r in synteny_read('extended_mapping_audit.tsv').itertuples():
        if r.mapped:
            extended[int(r.pad)][r.human_gene_id] = dict(chrom=r.chrom, start0=int(r.start0), end=int(r.end), strand=r.strand)
    return (core, flank, extended)

def synteny_main():
    synteny_OUT.mkdir(parents=True, exist_ok=True)
    synteny_engine.RESULT = synteny_OUT
    synteny_engine.parity_check()
    targets = synteny_read('human_input_1480.tsv').to_dict('records')
    candidates = synteny_read('chimp_Ensembl115_candidate_catalog.tsv').rename(columns={'chimp_gene_id': 'monkey_gene_id'}).to_dict('records')
    (core, flank, extended) = synteny_mappings()
    blast = synteny_read('relaxed_same_family_blast_hits.tsv')
    old = synteny_read('all_1480_synteny_calls.tsv').set_index('human_gene_id')
    rows = []
    for group in synteny_GROUPS:
        group_targets = [r for r in targets if r['analysis_group'] == group]
        ids = {r['stable_gene_id'] for r in group_targets}
        group_blast = blast[blast.human_gene_id.isin(ids)].to_dict('records')
        selected = synteny_engine.choose(group_targets, candidates, core, flank, extended, group_blast)
        for r in selected:
            prior = old.loc[r['human_gene_id']]
            assert r['monkey_gene_ids'] == prior.chimp_gene_ids
            assert r['selected_method'] == prior.selected_method
            assert r['relationship_before_reuse'] == prior.relationship_before_reuse
            r['query_class_size'] = len(group_targets)
            r['retained_in_previous_joint_run'] = bool(prior.retained)
            r['relationship_within_group'] = r['relationship_class']
            r['retained_within_group'] = r['retained']
            r['higher_priority_conflict'] = ''
            r['exclusion_reason'] = '' if r['retained'] else r['relationship_class']
        rows.extend(selected)
        print(group, 'within-group retained:', sum((r['retained'] for r in selected)), flush=True)
    occupied = {}
    conflicts = []
    for group in synteny_GROUPS:
        for r in rows:
            if r['analysis_group'] != group or not r['retained']:
                continue
            cid = r['monkey_gene_ids']
            if cid in occupied:
                first = occupied[cid]
                assert group == 'RNU_pseudogenes', ('Unexpected Pol II/U6 overlap', first, r)
                r['retained'] = False
                r['higher_priority_conflict'] = first['analysis_group'] + ':' + first['human_gene_id']
                r['exclusion_reason'] = 'target_already_assigned_to_functional_class'
                r['relationship_class'] = 'excluded_comparator_target_duplicate'
                conflicts.append(dict(chimp_gene_id=cid, retained_class=first['analysis_group'], retained_human_id=first['human_gene_id'], retained_human_name=first['human_display_name'], removed_class=group, removed_human_id=r['human_gene_id'], removed_human_name=r['human_display_name']))
            else:
                occupied[cid] = r
    actual = Counter((r['analysis_group'] for r in rows if r['retained']))
    assert actual == dict(zip(synteny_GROUPS, [28, 110, 1140])), actual
    synteny_engine.finalise(targets, candidates, rows)
    summary = []
    for group in synteny_GROUPS:
        sub = [r for r in rows if r['analysis_group'] == group]
        summary.append(dict(analysis_group=group, input_loci=len(sub), previous_joint_retained=sum((r['retained_in_previous_joint_run'] for r in sub)), within_group_retained=sum((r['retained_within_group'] for r in sub)), comparator_target_duplicates_removed=sum((bool(r['higher_priority_conflict']) for r in sub)), final_retained=sum((r['retained'] for r in sub))))
    synteny_engine.write(synteny_OUT / 'scope_correction_summary.tsv', summary)
    synteny_engine.write(synteny_OUT / 'cross_group_target_resolution.tsv', conflicts)
    final = pd.read_csv(synteny_OUT / 'all_1480_synteny_calls.tsv', sep='\t')
    pol2 = final[final.analysis_group.eq(synteny_GROUPS[0])]
    assert len(pol2) == 41 and pol2.retained.sum() == 28
    pol2.to_csv(synteny_OUT / 'POLII_41_decisions.tsv', sep='\t', index=False)
    proof = json.loads((synteny_OUT / 'synteny_validation.json').read_text())
    proof.update(joint_three_group_target_reuse_check=False, selection_scope='independent per input class: 41 Pol II / 116 U6 / 1323 pseudogenes', target_reuse_includes_sequence_only_candidates_as_in_original_macaque_code='within the same input class only', cross_class_resolution='Retain functional-class assignments; remove duplicate targets from the pseudogene comparator after independent mapping', cross_group_target_duplicates_removed=len(conflicts), macaque_data_changed=False, raw_mapping_and_sequence_evidence_unchanged=True, prior_joint_competition_scope_error_corrected=True, scope_counts=summary)
    (synteny_OUT / 'synteny_validation.json').write_text(json.dumps(proof, indent=2) + '\n')
    inputs = [synteny_EVIDENCE / x for x in ['human_input_1480.tsv', 'chimp_Ensembl115_candidate_catalog.tsv', 'core_mapping_audit.tsv', 'flank_prediction_audit.tsv', 'extended_mapping_audit.tsv', 'relaxed_same_family_blast_hits.tsv', 'all_1480_synteny_calls.tsv']]
    synteny_engine.write(synteny_OUT / 'input_manifest.tsv', [dict(path=str(p), bytes=p.stat().st_size, sha256=synteny_engine.sha(p)) for p in inputs])
    print(json.dumps(summary, indent=2), flush=True)

def synteny_entrypoint():
    synteny_main()
variants_merge = _component('analyze_fig1_baboon_af0p02').merge
variants_original_interval_counts = _component('analyze_fig1_baboon_af0p02').interval_counts
variants_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/primate/scripts/analyze_fig1_chimpanzee.py').parents[1]
variants_TMP = variants_ROOT / 'tmp/fig1_chimpanzee_20260923'
variants_OUT = variants_ROOT / 'data/processed/fig1_chimpanzee_20260923'
variants_BASE = variants_ROOT / 'data/processed/fig1_baboon_af0p02_complement_20260921'
variants_FA = Path('/public5/home/xuejing/human_hg38/coverage/hg38.fa')
variants_GTF = variants_ROOT / 'transpecies_compare/gencode.v44.annotation.gtf.gz'
variants_GROUPS = ['pol2_spliceRNUs', 'functional_U6s', 'RNU_pseudogenes', 'genome_wide_neutral']
variants_CHROMS = [f'chr{i}' for i in range(1, 23)]

def variants_signature(p):
    s = p.stat()
    return dict(path=str(p.resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)

def variants_prepare():
    h = pd.read_csv(variants_BASE / 'panel_a_gene_locus_density.tsv', sep='\t')
    h = h[h.species.eq('Human')]
    ids = set(h.unit_id)
    coords = {}
    with gzip.open(variants_GTF, 'rt') as f:
        for l in f:
            if l.startswith('#'):
                continue
            a = l.rstrip().split('\t')
            if a[2] != 'gene':
                continue
            at = dict(re.findall('(\\w+) "([^"]*)"', a[8]))
            gid = at['gene_id'].split('.')[0]
            if gid in ids:
                assert gid not in coords
                coords[gid] = (a[0], int(a[3]) - 1, int(a[4]), a[6])
    rows = []
    for r in h.itertuples():
        (c, s, e, st) = (r.chrom, int(r.start), int(r.end), '+') if r.analysis_group == variants_GROUPS[-1] else coords[r.unit_id]
        assert c in variants_CHROMS
        rows.append(dict(unit_id=r.unit_id, display_name=r.display_name, analysis_group=r.analysis_group, chrom=c, refseq_chrom=c, start=s, end=e, strand=st))
    cat = pd.DataFrame(rows)
    assert cat.groupby('analysis_group').size().to_dict() == dict(zip(variants_GROUPS, [41, 116, 1323, 1000]))
    cat.to_csv(variants_OUT / 'chimpanzee_catalog.tsv', sep='\t', index=False)
    fa = pysam.FastaFile(str(variants_FA))
    regions = defaultdict(list)
    for r in cat.itertuples():
        flank = 0 if r.analysis_group == variants_GROUPS[-1] else 2000
        assert 0 <= r.start - flank < r.end + flank <= fa.get_reference_length(r.chrom)
        regions[r.chrom].append((r.start - flank, r.end + flank))
    for c in variants_CHROMS:
        with (variants_TMP / f'{c}.targets.bed').open('w') as f:
            for (s, e) in variants_merge(regions[c]):
                f.write(f'{c}\t{s}\t{e}\n')
    for g in variants_GROUPS:
        cat[cat.analysis_group.eq(g)][['chrom', 'start', 'end', 'unit_id', 'strand']].assign(score=0)[['chrom', 'start', 'end', 'unit_id', 'score', 'strand']].to_csv(variants_OUT / f'{g}.hg38.bed', sep='\t', header=False, index=False)
    fa.close()
    contract = dict(assembly='GRCh38/hg38; chimpanzee reads aligned to human reference', annotation='GENCODE v44 coordinates; exact frozen Fig1 human IDs', groups=variants_GROUPS, group_sizes=[41, 116, 1323, 1000], AF_threshold=0.01, source_cohort_samples=198, current_vcf_samples=0, genotype_QC='Source header: GQ<20 set missing, AC/AN recalculated, AN>300, 0<ALT_AF<1, PASS; genotypes stripped', callable_mask=False, denominator='hg38 ACGT minus disqualified polymorphic positions; absent records are not independently proven callable', controls='same 1000 frozen human autosomal coordinates', count='observed allele types with 0<AC/AN<=0.01, including REF; no carrier weighting', position_mask='PASS/canonical/AN valid and all non-major observed alleles <=0.01; dominant allele is the baseline, irrespective of REF/ALT; common minor allele masks entire position', multiallelic='Recombine split ALT rows using pre-PASS parent VCF; REF count=AN-sum(ALT AC); reject inconsistent AN, partial missing sibling alleles, or mixed non-SNV sites', source_input=str(variants_ROOT / 'chimpanzee_variant'), baseline=str(variants_BASE), legacy_column='For Chimpanzee only, rare_snv_alt_count is a compatibility alias for total rare REF+ALT; explicit rare_ref_count, rare_alt_count and rare_snv_allele_count also provided')
    (variants_OUT / 'analysis_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    print('Prepared', cat.groupby('analysis_group').size().to_dict(), flush=True)

def variants_extract_chrom(c):
    source = variants_ROOT / 'chimpanzee_variant' / f'{c}.198chimpanzee.GQ20.GTonly.SNPs.vcf.gz'
    parent = source.resolve().parent.parent / f'{c}.198chimpanzee.GQ20.SNPs.filtering.vcf.gz'
    target = variants_TMP / f'{c}.targets.bed'
    dest = variants_TMP / f'{c}.targets.bcf'
    pout = variants_TMP / f'{c}.parent.bcf'
    mark = variants_TMP / f'{c}.extract.json'
    sig = dict(source=variants_signature(source), parent=variants_signature(parent), targets_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    if mark.exists() and json.loads(mark.read_text()).get('signature') == sig:
        return json.loads(mark.read_text())
    commands = [['bcftools', 'view', '-T', str(target), '-Ob', '-o', str(dest), str(source)], ['bcftools', 'view', '-R', str(target), '-G', '-Ob', '-o', str(pout), str(parent)]]
    with (variants_TMP / f'{c}.extraction.log').open('w') as log:
        for cmd in commands:
            log.write(json.dumps(cmd) + '\n')
            log.flush()
            subprocess.run(cmd, stdout=log, stderr=log, check=True)
    assert variants_signature(source) == sig['source'] and variants_signature(parent) == sig['parent']
    with pysam.VariantFile(str(dest)) as v:
        assert len(v.header.samples) == 0
        fa = pysam.FastaFile(str(variants_FA))
        assert v.header.contigs[c].length == fa.get_reference_length(c)
        fa.close()
        (variants_TMP / f'{c}.header.txt').write_text(str(v.header))
    rec = dict(chrom=c, status='PASS', signature=sig, gzip_crc='NOT_RUN', bgzf_eof=None, source_unchanged=True, commands=commands)
    mark.write_text(json.dumps(rec, indent=2) + '\n')
    print(c, 'target extraction PASS', flush=True)
    return rec

def variants_extract():
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(variants_extract_chrom, variants_CHROMS))
    (variants_OUT / 'extraction_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')

def variants_assess_site(ref, alts, an, afmax=0.01):
    """alts maps nucleotide to integer AC; single dominant allele is baseline."""
    if ref not in 'ACGT' or len(ref) != 1 or any((len(a) != 1 or a not in 'ACGT' or a == ref for a in alts)):
        return ([], 'noncanonical_SNV', {})
    if an <= 300 or any((ac < 0 or ac > an for ac in alts.values())) or sum(alts.values()) > an:
        return ([], 'invalid_AC_AN', {})
    counts = {ref: an - sum(alts.values()), **alts}
    observed = {a: n for (a, n) in counts.items() if n > 0}
    if not observed:
        return ([], 'no_observed_alleles', counts)
    major = max(observed, key=lambda a: (observed[a], a))
    if any((n / an > afmax for (a, n) in observed.items() if a != major)):
        return ([], 'common_minor_allele', counts)
    return ([a for (a, n) in observed.items() if 0 < n / an <= afmax], '', counts)

def variants_read_sites(path):
    sites = defaultdict(dict)
    with pysam.VariantFile(str(path)) as v:
        for r in v:
            for (i, a) in enumerate(r.alts or []):
                ref = r.ref
                pos = r.start
                while len(ref) > 1 and len(a) > 1 and (ref[-1] == a[-1]):
                    ref = ref[:-1]
                    a = a[:-1]
                while len(ref) > 1 and len(a) > 1 and (ref[0] == a[0]):
                    ref = ref[1:]
                    a = a[1:]
                    pos += 1
                val = (ref, int(r.info['AC'][i]), int(r.info['AN']), tuple(r.filter.keys()), float(r.info['AF'][i]))
                if a in sites[pos]:
                    assert sites[pos][a] == val
                sites[pos][a] = val
    return sites

def variants_classify_positions(chrom, af_max):
    sites = variants_read_sites(variants_TMP / f'{chrom}.targets.bcf')
    parents = variants_read_sites(variants_TMP / f'{chrom}.parent.bcf')
    rare = []
    ns = []
    refs = []
    masked = []
    audit = []
    qc = Counter()
    fa = pysam.FastaFile(str(variants_FA))
    for (pos, als) in sorted(sites.items()):
        qc['target_positions'] += 1
        up = parents[pos]
        assert set(als) <= set(up)
        for (a, v) in als.items():
            assert v[:4] == up[a][:4]
            assert abs(v[4] - v[1] / v[2]) < 1e-05
        ref = next(iter(als.values()))[0]
        assert fa.fetch(chrom, pos, pos + len(ref)).upper() == ref
        qc['REF_matches_hg38'] += 1
        reasons = []
        ans = {v[2] for v in up.values()}
        refset = {v[0] for v in up.values()}
        if len(ans) != 1 or len(refset) != 1:
            reasons.append('inconsistent_split_site')
        if set(up) != set(als):
            reasons.append('missing_filtered_or_fixed_sibling_ALT')
        if any((v[3] != ('PASS',) for v in up.values())):
            reasons.append('non_PASS')
        an = next(iter(ans))
        (qual, reason, counts) = variants_assess_site(ref, {a: v[1] for (a, v) in up.items()}, an, af_max)
        if reason:
            reasons.append(reason)
        if reasons:
            masked.append(pos)
        elif qual:
            rare.append(pos)
            ns.append(len(qual))
            refs.append(int(ref in qual))
        qc['multiallelic_positions'] += int(len(up) > 1)
        qc['counted_REF'] += int(not reasons and ref in qual)
        qc['counted_ALT'] += sum((a != ref for a in qual)) if not reasons else 0
        qc.update(set(reasons))
        for (a, n) in counts.items():
            audit.append(dict(chrom=chrom, pos1=pos + 1, ref=ref, allele=a, allele_role='REF' if a == ref else 'ALT', AC=n, AN=an, AF=n / an, counted=not reasons and a in qual, position_mask_reason=';'.join(sorted(set(reasons))), source_ALT_count=len(up)))
    pd.DataFrame(audit).to_csv(variants_OUT / f'{chrom}.classified_target_alleles.tsv.gz', sep='\t', index=False)
    fa.close()
    return (np.array(rare, dtype=int), np.r_[0, np.cumsum(ns)], np.array(masked, dtype=int), np.r_[0, np.cumsum(refs)], dict(qc))

def variants_interval_counts(s, e, seq, positions):
    r = variants_original_interval_counts(s, e, seq, (positions[0], positions[1], positions[2], positions[-1]))
    (l, u) = np.searchsorted(positions[0], [s, e])
    nr = int(positions[3][u] - positions[3][l])
    r.update(rare_ref_count=nr, rare_alt_count=r['rare_snv_alt_count'] - nr, rare_snv_allele_count=r['rare_snv_alt_count'])
    return r

def variants_count(af_max):
    manifest = json.loads((variants_OUT / 'extraction_manifest.json').read_text())
    assert len(manifest) == 22 and all((r['status'] == 'PASS' for r in manifest)), 'All 22 autosomes required'
    cat = pd.read_csv(variants_OUT / 'chimpanzee_catalog.tsv', sep='\t', dtype={'chrom': str})
    positions = {c: variants_classify_positions(c, af_max) for c in variants_CHROMS}
    fa = pysam.FastaFile(str(variants_FA))
    density = []
    bins = []
    for r in cat.itertuples():
        (s, e) = (int(r.start), int(r.end))
        seq = fa.fetch(r.refseq_chrom, s, e).upper()
        vals = variants_interval_counts(s, e, seq, positions[r.chrom])
        assert vals['effective_unmasked_bp'] > 0
        density.append(dict(species='Chimpanzee', assembly='GRCh38', analysis_group=r.analysis_group, unit_id=r.unit_id, display_name=r.display_name, chrom=r.chrom, start=s, end=e, strand=r.strand, **vals, callable_bp=np.nan, effective_callable_bp=np.nan, af_threshold=af_max, denominator_basis='reference_ACGT_minus_position_masks; no callable filter', source_note='198 samples upstream; source GQ20/AN>300; frozen human sets; rare REF+ALT; no chimp callable mask'))
        if r.analysis_group == variants_GROUPS[-1]:
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
            values = variants_interval_counts(a, b, window[a - s + 2000:b - s + 2000], positions[r.chrom])
            bins.append(dict(species='Chimpanzee', analysis_group=r.analysis_group, unit_id=r.unit_id, bin_index=k, region=region, relative_start=rs, relative_end=re_, chrom=r.chrom, genomic_start=a, genomic_end=b, **values))
    fa.close()
    d = pd.DataFrame(density)
    gb = pd.DataFrame(bins)
    body = gb[gb.region.eq('body')].groupby('unit_id')[['rare_snv_alt_count', 'effective_unmasked_bp']].sum()
    for r in d[d.analysis_group.ne(variants_GROUPS[-1])].itertuples():
        assert r.rare_snv_alt_count == body.loc[r.unit_id, 'rare_snv_alt_count'] and r.effective_unmasked_bp == body.loc[r.unit_id, 'effective_unmasked_bp']
    d.to_csv(variants_OUT / 'chimpanzee_gene_locus_density.tsv', sep='\t', index=False)
    gb.to_csv(variants_OUT / 'chimpanzee_profile_gene_bins.tsv.gz', sep='\t', index=False)
    prof = gb.groupby(['species', 'analysis_group', 'bin_index', 'region', 'relative_start', 'relative_end'], as_index=False).agg(gene_count=('unit_id', 'size'), contributing_genes=('effective_unmasked_bp', lambda v: (v > 0).sum()), total_rare_snv_alt_count=('rare_snv_alt_count', 'sum'), rare_REF=('rare_ref_count', 'sum'), rare_ALT=('rare_alt_count', 'sum'), total_effective_unmasked_bp=('effective_unmasked_bp', 'sum'))
    prof['pooled_density_per_kb'] = 1000 * prof.total_rare_snv_alt_count / prof.total_effective_unmasked_bp
    prof['total_effective_callable_bp'] = np.nan
    control = d[d.analysis_group.eq(variants_GROUPS[-1])]
    mean = control.rare_snv_density_per_kb.mean()
    pooled = 1000 * control.rare_snv_alt_count.sum() / control.effective_unmasked_bp.sum()
    assert mean > 0 and pooled > 0
    prof['density_ratio_to_control'] = prof.pooled_density_per_kb / pooled
    bars = []
    stats = []
    for g in variants_GROUPS:
        x = d[d.analysis_group.eq(g)].rare_snv_density_per_kb
        bars.append(dict(species='Chimpanzee', analysis_group=g, n_units=len(x), mean_density_per_kb=x.mean(), mean_density_ratio_to_control=x.mean() / mean))
    for (g1, g2) in combinations(variants_GROUPS, 2):
        x = d[d.analysis_group.eq(g1)].rare_snv_density_per_kb
        y = d[d.analysis_group.eq(g2)].rare_snv_density_per_kb
        t = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic', use_continuity=True)
        stats.append(dict(species='Chimpanzee', group_1=g1, group_2=g2, n1=len(x), n2=len(y), U=t.statistic, p_value=t.pvalue, mean_density_fold=x.mean() / y.mean() if y.mean() else np.nan))
    newtables = {'panel_a_gene_locus_density.tsv': d, 'panel_a_pairwise_statistics.tsv': pd.DataFrame(stats), 'panel_b_control_mean_ratios.tsv': pd.DataFrame(bars), 'panel_c_profile_control_ratios.tsv': prof, 'control_denominators.tsv': pd.DataFrame([dict(species='Chimpanzee', control_mean_density_per_kb=mean, control_pooled_density_per_kb=pooled, denominator_column='effective_unmasked_bp')])}
    for (name, new) in newtables.items():
        old = pd.read_csv(variants_BASE / name, sep='\t', float_precision='round_trip')
        joined = pd.concat([old, new], ignore_index=True)
        joined.to_csv(variants_OUT / name, sep='\t', index=False)
        pd.testing.assert_frame_equal(joined[joined.species.ne('Chimpanzee')][old.columns].reset_index(drop=True), old, check_dtype=False, check_exact=True)
    summary = d.groupby('analysis_group').agg(n=('unit_id', 'size'), count=('rare_snv_alt_count', 'sum'), rare_REF=('rare_ref_count', 'sum'), rare_ALT=('rare_alt_count', 'sum'), effective_bp=('effective_unmasked_bp', 'sum'), mean_density=('rare_snv_density_per_kb', 'mean'), zero_count=('rare_snv_alt_count', lambda x: (x == 0).sum()))
    summary.to_csv(variants_OUT / 'chimpanzee_group_summary.tsv', sep='\t')
    (variants_OUT / 'chimpanzee_variant_QC.json').write_text(json.dumps({c: p[-1] for (c, p) in positions.items()}, indent=2) + '\n')
    contract = json.loads((variants_OUT / 'analysis_contract.json').read_text())
    contract['AF_threshold'] = af_max
    (variants_OUT / 'analysis_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(summary.to_string(), flush=True)

def variants_main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'extract', 'count', 'all'])
    p.add_argument('--af', type=float, default=0.01)
    args = p.parse_args()
    variants_TMP.mkdir(parents=True, exist_ok=True)
    variants_OUT.mkdir(parents=True, exist_ok=True)
    actions = {'prepare': variants_prepare, 'extract': variants_extract,
               'count': lambda: variants_count(args.af)}
    for stage in ['prepare', 'extract', 'count'] if args.stage == 'all' else [args.stage]:
        actions[stage]()

def variants_entrypoint():
    variants_main()
reference_task = _component('revise_fig1_gene_rules_20261004')
reference_engine = _component('map_chimpanzee_all_rnu_macaque_20260928')
reference_prior = _component('reclassify_chimp_per_group_macaque_20260928')
reference_OUT = reference_task.RES / 'chimp_synteny'
reference_TMP = reference_task.TMP / 'chimp_synteny'
reference_read = reference_task.read
reference_save = reference_task.save

def reference_main():
    reference_OUT.mkdir(exist_ok=True)
    reference_TMP.mkdir(exist_ok=True)
    reference_engine.RESULT = reference_OUT
    reference_engine.TMP = reference_TMP
    oldtargets = reference_read(reference_prior.EVIDENCE / 'human_input_1480.tsv')
    native = reference_task.catalogs()
    human = reference_read(reference_task.OUT / 'nonchimp_locus_density.tsv')
    human = human[human.species.eq('Human') & human.analysis_group.ne(reference_task.GROUPS[-1])]
    wanted = set(human.unit_id)
    targets = []
    for r in human.itertuples():
        g = native['Human'].loc[r.unit_id]
        targets.append(dict(stable_gene_id=r.unit_id, display_name=g.gene_name, family=reference_engine.family(g.gene_name), chrom=g.chrom, start=int(g.start), end=int(g.end), strand=g.strand, analysis_group=r.analysis_group))
    extra = [dict(r, core_interval=(r['chrom'], r['start'], r['end'], r['stable_gene_id'], r['strand'])) for r in targets if r['stable_gene_id'] not in set(oldtargets.stable_gene_id)]
    assert len(extra) >= 5
    candidates = reference_read(reference_prior.EVIDENCE / 'chimp_Ensembl115_candidate_catalog.tsv').fillna('').rename(columns={'chimp_gene_id': 'monkey_gene_id'}).to_dict('records')
    (core, flanks, extended) = reference_prior.mappings()
    if not (reference_OUT / 'relaxed_same_family_blast_hits.tsv').exists():
        (ec, ef, ee) = reference_engine.mappings(extra, candidates)
        eb = reference_engine.sequences(extra, candidates, {r['stable_gene_id']: SimpleNamespace(strand=r['strand']) for r in extra})
    else:
        old_evidence = reference_prior.EVIDENCE
        reference_prior.EVIDENCE = reference_OUT
        (ec, ef, ee) = reference_prior.mappings()
        reference_prior.EVIDENCE = old_evidence
        eb = reference_read(reference_OUT / 'relaxed_same_family_blast_hits.tsv').to_dict('records')
    core.update(ec)
    for (k, v) in ef.items():
        flanks[k].update(v)
    for (k, v) in ee.items():
        extended[k].update(v)
    blast = reference_read(reference_prior.EVIDENCE / 'relaxed_same_family_blast_hits.tsv').to_dict('records') + eb
    rows = []
    occupied = set()
    for group in reference_task.GROUPS[:3]:
        subset = [r for r in targets if r['analysis_group'] == group]
        ids = {r['stable_gene_id'] for r in subset}
        selected = reference_engine.choose(subset, candidates, core, flanks, extended, [r for r in blast if r['human_gene_id'] in ids])
        for r in selected:
            r['exclusion_reason'] = '' if r['retained'] else r['relationship_class']
            if r['retained'] and r['monkey_gene_ids'] in occupied:
                r['retained'] = False
                r['exclusion_reason'] = 'target_already_assigned_to_functional_class'
            if r['retained']:
                g = native['Chimpanzee'].loc[r['monkey_gene_ids']]
                if g.resolved_family == 'U7':
                    r['retained'] = False
                    r['exclusion_reason'] = 'U7_annotation'
                elif group == reference_task.GROUPS[2] and g.resolved_family == 'UNKNOWN' and (g.gene_length_bp <= 70):
                    r['retained'] = False
                    r['exclusion_reason'] = 'unresolved_family_native_gene_length_le70'
            if r['retained']:
                occupied.add(r['monkey_gene_ids'])
        rows += selected
    calls = pd.DataFrame(rows).rename(columns={'monkey_gene_ids': 'chimp_gene_ids'})
    reference_save(calls, reference_OUT / 'all_synteny_calls.tsv')
    reference_save(pd.DataFrame(targets), reference_OUT / 'human_input.tsv')
    reference_save(calls[calls.retained], reference_OUT / 'retained_hg38_genes.tsv')
    original = reference_read(reference_task.BASE / 'panel_a_gene_locus_density.tsv')
    old = original[original.species.eq('Chimpanzee')]
    selected = calls[calls.retained]
    keep = set(selected.human_gene_id)
    for group in reference_task.GROUPS[:2]:
        assert set(selected.loc[selected.analysis_group.eq(group), 'human_gene_id']) == set(old.loc[old.analysis_group.eq(group), 'unit_id'])
    added = keep - set(old.unit_id)
    print('Chimp new retained:', selected.groupby('analysis_group').size().to_dict(), 'new counting IDs:', sorted(added), flush=True)
    source = reference_task.ROOT / 'data/processed/fig1_species_AF_restored_20260928/chimpanzee_profile_gene_bins.tsv.gz'
    bins = reference_read(source)
    bins = bins[bins.unit_id.isin(keep)].copy()
    d = old[old.unit_id.isin(keep) | old.analysis_group.eq(reference_task.GROUPS[-1])].copy()
    if added:
        sys.path.insert(0, str(reference_task.ROOT / 'primate/scripts'))
        coremod = _component('analyze_fig1_chimpanzee')
        coremod.TMP = reference_task.TMP / 'chimp_new_count'
        coremod.OUT = reference_task.OUT / 'chimp_variant_audit'
        coremod.TMP.mkdir(exist_ok=True)
        coremod.OUT.mkdir(exist_ok=True)
        regions = {}
        for gid in sorted(added):
            g = native['Human'].loc[gid]
            regions.setdefault(g.chrom, []).append((int(g.start) - 2000, int(g.end) + 2000))
        positions = {}
        for (chrom, intervals) in regions.items():
            (coremod.TMP / f'{chrom}.targets.bed').write_text(''.join((f'{chrom}\t{s}\t{e}\n' for (s, e) in coremod.merge(intervals))))
            coremod.extract_chrom(chrom)
            positions[chrom] = coremod.classify_positions(chrom, 0.01)
        extra_d = []
        extra_b = []
        pe = reference_task.profile_engine()
        with pysam.FastaFile(str(coremod.FA)) as fa:
            for gid in sorted(added):
                g = native['Human'].loc[gid]
                (s, e) = (int(g.start), int(g.end))
                seq = fa.fetch(g.chrom, s - 2000, e + 2000).upper()
                v = coremod.interval_counts(s, e, seq[2000:2000 + e - s], positions[g.chrom])
                assert v['effective_unmasked_bp'] > 0
                extra_d.append(dict(species='Chimpanzee', assembly='GRCh38', analysis_group=reference_task.GROUPS[2], unit_id=gid, display_name=g.gene_name, chrom=g.chrom, start=s, end=e, strand=g.strand, **v, af_threshold=0.01, source_note='Updated human full-length non-U7 pseudogene reference; same per-class synteny; original chimp rare REF+ALT rule'))
                b = pe.build_profile_bins(pd.DataFrame([dict(species='Chimpanzee', analysis_group=reference_task.GROUPS[2], stable_gene_id=gid, display_name=g.gene_name, chrom=g.chrom, start0=s, end=e, strand=g.strand)]))
                for r in b.itertuples():
                    v = coremod.interval_counts(r.genomic_start, r.genomic_end, seq[r.genomic_start - s + 2000:r.genomic_end - s + 2000], positions[g.chrom])
                    extra_b.append(dict(species='Chimpanzee', analysis_group=reference_task.GROUPS[2], unit_id=gid, bin_index=r.bin_index, region=r.region, relative_start=r.relative_start, relative_end=r.relative_end, chrom=g.chrom, genomic_start=r.genomic_start, genomic_end=r.genomic_end, **v))
        d = pd.concat([d, pd.DataFrame(extra_d)], ignore_index=True)
        bins = pd.concat([bins, pd.DataFrame(extra_b)], ignore_index=True)
    assert set(bins.unit_id) == keep and bins.groupby('unit_id').size().eq(100).all()
    body = bins[bins.region.eq('body')].groupby('unit_id')[['rare_snv_alt_count', 'effective_unmasked_bp']].sum()
    q = d[d.unit_id.isin(keep)].set_index('unit_id')
    np.testing.assert_array_equal(body.loc[q.index].to_numpy(), q[['rare_snv_alt_count', 'effective_unmasked_bp']].to_numpy())
    reference_save(d, reference_task.OUT / 'Chimpanzee.locus_density.tsv')
    reference_save(bins, reference_task.OUT / 'Chimpanzee.profile_gene_bins.tsv.gz')
    reference_save(pd.DataFrame([dict(gene_id=g, change='added' if g in added else 'removed') for g in sorted(added | set(old.loc[old.analysis_group.ne(reference_task.GROUPS[-1]), 'unit_id']) - keep)]), reference_OUT / 'membership_changes.tsv')
    (reference_OUT / 'validation.json').write_text(json.dumps(dict(status='PASS', input_gene_count=len(targets), counts=selected.groupby('analysis_group').size().to_dict(), new_counted_loci=sorted(added), functional_sets_unchanged=True), indent=2) + '\n')

def reference_entrypoint():
    reference_main()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    _run_stages({'mapping': 'mapping_entrypoint', 'synteny': 'synteny_entrypoint', 'variants': 'variants_entrypoint', 'reference': 'reference_entrypoint'}, globals(), setup=_prepare_runtime)
