"""Consolidated final processing: classify_rat_rnu_rn7_20260921.

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
import collections
import gzip
import hashlib
import re
import subprocess
annotation_ROOT = (_PROCESSING_ROOT / 'fig1/chicken/primate/transpecies_compare/scripts/classify_rat_rnu_rn7_20260921.py').parents[1]
annotation_RAW = annotation_ROOT / 'tmp/rat_snrna_feasibility_20260921'
annotation_WORK = annotation_ROOT / 'tmp/rat_rnu_rn7_20260921'
annotation_OUT = annotation_ROOT / 'data/processed/rat_rnu_rn7_20260921'
annotation_FINAL = annotation_ROOT / 'final_result/rat_rnu_rn7_20260921'
annotation_INFERNAL = Path('/public/home/xuejing/mambaforge/envs/infernal/bin')
annotation_ATTR = re.compile('(\\w+) "([^"]*)"')
annotation_POL2 = {'U1', 'U2', 'U4', 'U5', 'U11', 'U12', 'U4atac'}
annotation_U6 = {'U6', 'U6atac'}

def annotation_fasta_records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as fh:
        (header, seq) = (None, [])
        for line in fh:
            if line.startswith('>'):
                if header:
                    yield (header, ''.join(seq))
                (header, seq) = (line[1:].strip(), [])
            else:
                seq.append(line.strip())
        if header:
            yield (header, ''.join(seq))

def annotation_gtf_genes(path, biotype=None):
    genes = []
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip().split('\t')
            if len(f) != 9 or f[2] != 'gene':
                continue
            a = dict(annotation_ATTR.findall(f[8]))
            bt = a.get('gene_biotype', a.get('gene_type', ''))
            if biotype and bt != biotype:
                continue
            genes.append(dict(gene_id=a['gene_id'].split('.')[0], gene_name=a.get('gene_name', ''), chrom=f[0] if f[0].startswith('chr') else 'chr' + f[0], start=int(f[3]) - 1, end=int(f[4]), strand=f[6], biotype=bt))
    return genes

def annotation_dump_table(rows, path):
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, sep='\t', index=False)

def annotation_write_bed(rows, path, promoter=False):
    with path.open('w') as fh:
        for r in rows:
            (s, e) = (r['start'], r['end'])
            if promoter:
                tss = s if r['strand'] == '+' else e - 1
                (s, e) = (max(0, tss - 50), tss + 51)
            fh.write(f"{r['chrom']}\t{s}\t{e}\t{r['gene_id']}\t0\t{r['strand']}\n")

def annotation_run(cmd, log_name):
    with (annotation_WORK / (log_name + '.log')).open('w') as fh:
        fh.write('COMMAND: ' + ' '.join(map(str, cmd)) + '\n')
        fh.flush()
        subprocess.run(list(map(str, cmd)), check=True, stdout=fh, stderr=fh)

def annotation_prepare():
    rat = annotation_gtf_genes(annotation_RAW / 'Rattus_norvegicus.mRatBN7.2.112.gtf.gz', 'snRNA')
    ids = {r['gene_id'] for r in rat}
    assert len(ids) == len(rat)
    seqs = collections.defaultdict(list)
    for (h, s) in annotation_fasta_records(annotation_RAW / 'Rattus_norvegicus.mRatBN7.2.ncrna.fa.gz'):
        m = re.search('gene:(\\S+)', h)
        if m and m[1].split('.')[0] in ids:
            seqs[m[1].split('.')[0]].append((h.split()[0], s, h))
    with (annotation_WORK / 'rat_snrna.fa').open('w') as fh:
        for r in rat:
            candidates = seqs[r['gene_id']]
            assert candidates, r
            (tx, seq, desc) = sorted(candidates, key=lambda x: (-len(x[1]), x[0]))[0]
            r.update(transcript_id=tx, transcript_count=len(candidates), sequence_length=len(seq), sequence_header=desc, autosomal=r['chrom'] in {f'chr{i}' for i in range(1, 21)})
            fh.write(f">{r['gene_id']}\n{seq}\n")
    annotation_dump_table(rat, annotation_OUT / 'rat_snRNA_ensembl112.tsv')
    annotation_write_bed(rat, annotation_WORK / 'rat_snRNA.rn7.bed')
    annotation_write_bed(rat, annotation_WORK / 'rat_snRNA_TSS50.rn7.bed', True)
    mouse_path = Path('/public5/home/xuejing/RNU/mouse/rnus_reanno/gencode.vM25.annotation.gtf.gz')
    groups = {}
    for (group, name) in [('pol2_spliceRNUs', 'spliceRNUs_minusU6'), ('functional_U6s', 'RNUs_U6'), ('RNU_pseudogenes', 'pseudoRNUs')]:
        with (annotation_ROOT / f'mouse_bed/mm10.gencodevM25.{name}_exon.bed').open() as fh:
            for line in fh:
                gid = line.split()[3].split('.')[0]
                assert gid not in groups or groups[gid] == group
                groups[gid] = group
    mouse = annotation_gtf_genes(mouse_path, 'snRNA')
    for r in mouse:
        r['mouse_group'] = groups.get(r['gene_id'], 'outside_current_three_sets')
    annotation_dump_table(mouse, annotation_OUT / 'mouse_source_snRNA.tsv')
    annotation_write_bed(mouse, annotation_WORK / 'mouse_snRNA.mm10.bed')
    print('Prepared rat genes:', len(rat), 'autosomes:', sum((r['autosomal'] for r in rat)), 'mouse:', len(mouse), collections.Counter(groups.values()))
    models = [annotation_RAW / 'rfam_U1.cm', *sorted((annotation_WORK / 'models').glob('RF*.cm'))]
    assert len(models) == 10
    with (annotation_WORK / 'snrna_models.cm').open('wb') as fh:
        for m in models:
            data = m.read_bytes()
            assert data.startswith(b'INFERNAL'), m
            fh.write(data)
    annotation_run([annotation_INFERNAL / 'cmpress', '-F', annotation_WORK / 'snrna_models.cm'], 'cmpress')

def annotation_scan():
    names = {re.search('^NAME\\s+(\\S+)', p.read_text(), re.M)[1] for p in [annotation_RAW / 'rfam_U1.cm', *sorted((annotation_WORK / 'models').glob('RF*.cm'))]}
    with (annotation_WORK / 'snrna.clanin').open('w') as fh:
        for line in (annotation_WORK / 'models/Rfam.clanin').read_text().splitlines():
            fields = line.split()
            kept = [x for x in fields[1:] if x in names]
            if kept:
                fh.write('\t'.join([fields[0], *kept]) + '\n')
    annotation_run([annotation_INFERNAL / 'cmscan', '--cpu', '4', '--cut_ga', '--rfam', '--nohmmonly', '--fmt', '2', '--clanin', annotation_WORK / 'snrna.clanin', '--tblout', annotation_WORK / 'rat_snrna.cmscan.tbl', '-o', annotation_WORK / 'rat_snrna.cmscan.out', annotation_WORK / 'snrna_models.cm', annotation_WORK / 'rat_snrna.fa'], 'cmscan')

def annotation_map_intervals():
    for (label, source, chain, mm) in [('mouse_to_rat', 'mouse_snRNA.mm10.bed', 'mm10ToRn7.over.chain.gz', 0.5), ('rat_to_mouse', 'rat_snRNA.rn7.bed', 'rn7ToMm10.over.chain.gz', 0.5), ('rat_promoter_to_rn6', 'rat_snRNA_TSS50.rn7.bed', 'rn7ToRn6.over.chain.gz', 0.95)]:
        annotation_run(['liftOver', f'-minMatch={mm}', annotation_WORK / source, annotation_RAW / chain, annotation_WORK / (label + '.bed'), annotation_WORK / (label + '.unmapped.bed')], label)

def annotation_intersect(a, b, output, options=()):
    """Write only BEDTools output to data files; keep diagnostics separate."""
    cmd = ['bedtools', 'intersect', '-a', str(a), '-b', str(b), *options]
    with output.open('w') as fh, (annotation_WORK / (output.name + '.log')).open('w') as log:
        log.write(' '.join(cmd) + '\n')
        log.flush()
        subprocess.run(cmd, stdout=fh, stderr=log, check=True)

def annotation_family_evidence(frame):
    table = annotation_WORK / 'rat_snrna.cmscan.tbl'
    assert table.read_text().rstrip().endswith('# [ok]'), 'Incomplete Infernal scan'
    hits = collections.defaultdict(list)
    for line in table.read_text().splitlines():
        if line.startswith('#'):
            continue
        f = line.split()
        if f[18] != '!' or f[19] == '=' or f[11] != '+':
            continue
        hits[f[3]].append(dict(cm_family=f[1], cm_accession=f[2], cm_score=float(f[16]), cm_evalue=float(f[17]), cm_model_coverage=(int(f[8]) - int(f[7]) + 1) / int(f[26]), cm_sequence_coverage=(int(f[10]) - int(f[9]) + 1) / int(f[27])))
    rows = []
    for r in frame.to_dict('records'):
        desc = r['sequence_header'].split('description:', 1)[-1]
        m = re.search('\\b(U(?:[46]atac|11|12|[124567]))\\b', desc, re.I)
        annotated = m[1].upper().replace('ATAC', 'atac') if m else ''
        h = sorted(hits[r['gene_id']], key=lambda x: -x['cm_score'])
        best = h[0] if h else {}
        cm = best.get('cm_family', '')
        conflict = bool(annotated and cm and (annotated != cm))
        family = cm or annotated or 'unclassified'
        r.update(annotated_family=annotated, family=family, family_conflict=conflict, family_source='Rfam_CM_GA' if cm else 'Ensembl_RGD_description' if annotated else 'unclassified', cm_pass=bool(cm), cm_model_coverage=0, cm_sequence_coverage=0)
        r.update(best)
        rows.append(r)
    return rows

def annotation_synteny_evidence(mouse, rat):
    import pandas as pd
    forward = annotation_WORK / 'mouse_rat_forward_overlap.bed'
    reverse = annotation_WORK / 'mouse_rat_reverse_overlap.bed'
    opts = ['-s', '-f', '0.3', '-F', '0.3', '-wo']
    annotation_intersect(annotation_WORK / 'mouse_to_rat.bed', annotation_WORK / 'rat_snRNA.rn7.bed', forward, opts)
    annotation_intersect(annotation_WORK / 'rat_to_mouse.bed', annotation_WORK / 'mouse_snRNA.mm10.bed', reverse, opts)
    rev = {}
    for l in reverse.read_text().splitlines():
        f = l.split()
        rev[f[9], f[3]] = min(int(f[12]) / (int(f[2]) - int(f[1])), int(f[12]) / (int(f[8]) - int(f[7])))
    md = mouse.set_index('gene_id').to_dict('index')
    pairs = []
    for l in forward.read_text().splitlines():
        f = l.split()
        key = (f[3], f[9])
        pairs.append(dict(mouse_gene_id=key[0], rat_gene_id=key[1], mouse_gene_name=md[key[0]]['gene_name'], mouse_group=md[key[0]]['mouse_group'], forward_min_overlap=min(int(f[12]) / (int(f[2]) - int(f[1])), int(f[12]) / (int(f[8]) - int(f[7]))), reverse_min_overlap=rev.get(key, 0), reciprocal=key in rev))
    pair_table = pd.DataFrame(pairs)
    reciprocal = pair_table[pair_table.reciprocal]
    mouse_degree = reciprocal.groupby('mouse_gene_id').rat_gene_id.nunique().to_dict()
    rat_degree = reciprocal.groupby('rat_gene_id').mouse_gene_id.nunique().to_dict()
    pair_table['one_to_one_in_catalogue'] = [bool(r.reciprocal and mouse_degree.get(r.mouse_gene_id) == 1 and (rat_degree.get(r.rat_gene_id) == 1)) for r in pair_table.itertuples()]
    pair_table.to_csv(annotation_OUT / 'mouse_rat_synteny_pairs.tsv', sep='\t', index=False)
    return pair_table

def annotation_classify(allow_single_atlas=False):
    import pandas as pd
    rat = pd.read_csv(annotation_OUT / 'rat_snRNA_ensembl112.tsv', sep='\t').fillna('')
    mouse = pd.read_csv(annotation_OUT / 'mouse_source_snRNA.tsv', sep='\t').fillna('')
    rat = pd.DataFrame(annotation_family_evidence(rat))
    pairs = annotation_synteny_evidence(mouse, rat)
    multiorgan = annotation_RAW / 'multiorgan_consensus_peaks.rn6.bed'
    brain = annotation_WORK / 'nervous_system_peaks.rn6.bed'
    if not brain.exists() and (not allow_single_atlas):
        raise FileNotFoundError('Brain peaks unavailable; use --allow-single-atlas only for explicitly preliminary output')
    sources = {'multiorgan': multiorgan}
    if brain.exists():
        sources['nervous_system'] = brain
    promoter_rn6 = {l.split()[3] for l in (annotation_WORK / 'rat_promoter_to_rn6.bed').read_text().splitlines()}
    qc = []
    for (label, source) in sources.items():
        mapped = annotation_WORK / f'{label}_peaks.rn7.bed'
        unmapped = annotation_WORK / f'{label}_peaks.unmapped.bed'

        def count_bed(path):
            if not path.exists():
                return 0
            with path.open() as handle:
                return sum((bool(l.strip()) and (not l.startswith('#')) for l in handle))
        source_count = count_bed(source)
        if count_bed(mapped) + count_bed(unmapped) != source_count:
            annotation_run(['liftOver', '-minMatch=0.95', source, annotation_RAW / 'rn6ToRn7.over.chain.gz', mapped, unmapped], f'{label}_peaks_liftover')
        assert count_bed(mapped) + count_bed(unmapped) == source_count
        hitfile = annotation_WORK / f'{label}_promoter_overlap.bed'
        annotation_intersect(annotation_WORK / 'rat_snRNA_TSS50.rn7.bed', mapped, hitfile, ['-wo'])
        hitids = set()
        max_overlap = collections.defaultdict(int)
        for line in hitfile.read_text().splitlines():
            f = line.split()
            hitids.add(f[3])
            max_overlap[f[3]] = max(max_overlap[f[3]], int(f[-1]))
        rat[f'{label}_TSS50_overlap'] = rat.gene_id.isin(hitids)
        rat[f'{label}_max_overlap_bp'] = rat.gene_id.map(max_overlap).fillna(0).astype(int)
        qc.append(dict(source=label, source_peaks=source_count, mapped_peaks=count_bed(mapped), unmapped_peaks=count_bed(unmapped), genes_overlapping=len(hitids)))
    if 'nervous_system' not in sources:
        rat['nervous_system_TSS50_overlap'] = pd.NA
        rat['nervous_system_max_overlap_bp'] = pd.NA
    rat['promoter_rn6_assessable'] = rat.gene_id.isin(promoter_rn6)
    rat['atac_any'] = rat[[f'{x}_TSS50_overlap' for x in sources]].any(axis=1)
    rat['atac_both'] = rat.multiorgan_TSS50_overlap & rat.nervous_system_TSS50_overlap.fillna(False).astype(bool) if len(sources) == 2 else pd.NA
    if len(sources) == 2:
        shared = annotation_WORK / 'two_atlas_shared_intervals.rn7.bed'
        annotation_intersect(annotation_WORK / 'multiorgan_peaks.rn7.bed', annotation_WORK / 'nervous_system_peaks.rn7.bed', shared)
        overlap = annotation_WORK / 'shared_atlas_promoter_overlap.bed'
        annotation_intersect(annotation_WORK / 'rat_snRNA_TSS50.rn7.bed', shared, overlap, ['-u'])
        shared_ids = {l.split()[3] for l in overlap.read_text().splitlines()}
        rat['atac_shared_interval'] = rat.gene_id.isin(shared_ids)
    else:
        rat['atac_shared_interval'] = pd.NA
    recip = pairs[pairs.reciprocal]
    pol2_recip = recip[recip.mouse_group == 'pol2_spliceRNUs']
    u6_recip = recip[recip.mouse_group == 'functional_U6s']
    rat['mouse_pol2_reciprocal'] = rat.gene_id.isin(pol2_recip.rat_gene_id)
    rat['mouse_u6_reciprocal'] = rat.gene_id.isin(u6_recip.rat_gene_id)
    rat['mouse_pol2_one_to_one'] = rat.gene_id.isin(pol2_recip[pol2_recip.one_to_one_in_catalogue].rat_gene_id)
    rat['mouse_reciprocal_gene_ids'] = rat.gene_id.map(recip.groupby('rat_gene_id').mouse_gene_id.agg(';'.join)).fillna('')
    rat['mouse_reciprocal_groups'] = rat.gene_id.map(recip.groupby('rat_gene_id').mouse_group.agg(lambda s: ';'.join(sorted(set(s))))).fillna('')
    groups = []
    for r in rat.itertuples():
        if r.family == 'U7':
            group = 'excluded_nonsplice_U7'
        elif r.family_conflict or r.family not in annotation_POL2 | annotation_U6:
            group = 'unresolved_family'
        elif r.atac_any:
            group = 'pol2_spliceRNUs' if r.family in annotation_POL2 else 'functional_U6s'
        elif not r.promoter_rn6_assessable:
            group = 'unresolved_coordinate_transfer'
        elif r.mouse_pol2_reciprocal or r.mouse_u6_reciprocal:
            group = 'syntenic_without_rat_ATAC_support'
        else:
            group = 'putative_RNU_pseudogenes'
        groups.append(group)
    rat['classification'] = groups
    rat['high_confidence_native_evidence'] = rat.atac_any & rat.cm_pass & (rat.cm_model_coverage >= 0.7) & ~rat.family_conflict
    rat['pol2_syntenic_ATAC_supported'] = rat.classification.eq('pol2_spliceRNUs') & rat.mouse_pol2_reciprocal
    rat['callable_filter_applied'] = False
    tag = 'two_atlas' if len(sources) == 2 else 'preliminary_multiorgan_only'
    dest = annotation_FINAL / tag
    dest.mkdir(exist_ok=True)
    rat.to_csv(dest / 'rat_rn7_RNU_classification.tsv', sep='\t', index=False)
    pairs.to_csv(dest / 'mouse_rat_synteny_pairs.tsv', sep='\t', index=False)
    summary = rat.groupby(['classification', 'autosomal']).size().reset_index(name='n_genes')
    family = rat.groupby(['classification', 'family']).size().reset_index(name='n_genes')
    summary.to_csv(dest / 'classification_counts.tsv', sep='\t', index=False)
    family.to_csv(dest / 'classification_by_family.tsv', sep='\t', index=False)
    pd.DataFrame(qc).to_csv(dest / 'peak_liftover_qc.tsv', sep='\t', index=False)
    for (group, part) in rat.groupby('classification'):
        annotation_write_bed(part.to_dict('records'), dest / (group + '.all_chromosomes.bed'))
        annotation_write_bed(part[part.autosomal].to_dict('records'), dest / (group + '.autosomal.bed'))
    annotation_write_bed(rat[rat.pol2_syntenic_ATAC_supported & rat.autosomal].to_dict('records'), dest / 'pol2_syntenic_ATAC_supported.autosomal.bed')
    annotation_write_bed(rat[rat.high_confidence_native_evidence & rat.classification.eq('pol2_spliceRNUs') & rat.autosomal].to_dict('records'), dest / 'pol2_native_high_confidence.autosomal.bed')
    if len(sources) == 2:
        annotation_write_bed(rat[rat.atac_shared_interval & rat.classification.eq('pol2_spliceRNUs') & rat.autosomal].to_dict('records'), dest / 'pol2_shared_atlas_interval.autosomal.bed')

def annotation_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'scan', 'map', 'classify'])
    p.add_argument('--allow-single-atlas', action='store_true')
    args = p.parse_args()
    for d in [annotation_WORK, annotation_OUT, annotation_FINAL]:
        d.mkdir(parents=True, exist_ok=True)
    if args.stage == 'classify':
        annotation_classify(args.allow_single_atlas)
    else:
        {'prepare': annotation_prepare, 'scan': annotation_scan, 'map': annotation_map_intervals}[args.stage]()

def annotation_entrypoint():
    annotation_main()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    _run_stages({'annotation': 'annotation_entrypoint'}, globals(), setup=_prepare_runtime)
