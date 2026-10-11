"""Consolidated final processing: analyze_rnu_pol2_orthology.

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
import argparse
import csv
import gzip
import hashlib
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
macaque_ROOT = (_PROCESSING_ROOT / 'fig1/AN_validation/scripts/analyze_rnu_pol2_orthology.py').parents[1]
macaque_STRICT41 = Path('/public5/home/xuejing/RNU/codex/transpecies_compare/rnu_three_gene_sets_20260810/POL2_spliceRNUs_strict41_gene_list.tsv')
macaque_HUMAN_GTF = Path('/public5/home/xuejing/RNU/asymmetry/data/gencode.v44.annotation.gtf.gz')
macaque_HUMAN_FASTA = Path('/public5/home/xuejing/human_hg38/coverage/hg38.fa')
macaque_MONKEY_GTF = Path('/public5/home/xuejing/RNU/monkey/reannotate/Macaca_mulatta.Mmul_10.115.gtf.gz')
macaque_MONKEY_NCRNA_FASTA = Path('/public5/home/xuejing/RNU/monkey/reannotate/Macaca_mulatta.Mmul_10.ncrna.fa.gz')
macaque_CHAIN = Path('/public5/home/xuejing/RNU/codex/transpecies_compare/tmp/rhesus_rnu_function_report_20260812/sources/hg38ToRheMac10.over.chain.gz')
macaque_MONKEY_BEDS = {'spliceRNUs_minusU6': Path('/public5/home/xuejing/RNU/monkey/signature/gene_group_region_bed/rheMac10.Mmul_10.spliceRNUs_minusU6_exon.bed'), 'RNUs_U6': Path('/public5/home/xuejing/RNU/monkey/signature/gene_group_region_bed/rheMac10.Mmul_10.RNUs_U6_exon.bed'), 'pseudoRNUs': Path('/public5/home/xuejing/RNU/monkey/signature/gene_group_region_bed/rheMac10.Mmul_10.pseudoRNUs_exon.bed')}
macaque_CORE_THRESHOLDS = (0.95, 0.8, 0.5, 0.2)
macaque_EXTENDED_PADS = (500, 2000, 10000, 50000)
macaque_FLANK_SCHEMES = ((200, 1000), (500, 5000), (2000, 20000), (10000, 50000))
macaque_DIRECT_RECIPROCAL_MIN = 0.3
macaque_MAX_NEAREST_BP = 100000

def macaque_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=macaque_ROOT / 'results/rnu_pol2_orthology')
    parser.add_argument('--work-dir', type=Path, default=macaque_ROOT / 'tmp/rnu_pol2_orthology_20260814/work/analysis')
    return parser.parse_args()

def macaque_require_inputs() -> None:
    paths = [macaque_STRICT41, macaque_HUMAN_GTF, macaque_HUMAN_FASTA, macaque_HUMAN_FASTA.with_suffix('.fa.fai')]
    paths += [macaque_MONKEY_GTF, macaque_MONKEY_NCRNA_FASTA, macaque_CHAIN, *macaque_MONKEY_BEDS.values()]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing inputs:\n' + '\n'.join(missing))
    for command in ('liftOver', 'makeblastdb', 'blastn'):
        if shutil.which(command) is None:
            raise RuntimeError(f'Required executable not found: {command}')

def macaque_read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))

def macaque_write_tsv(path: Path, rows: list[dict], fieldnames: list[str] | None=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f'Cannot infer columns for empty table: {path}')
        fieldnames = list(rows[0])
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def macaque_parse_attributes(text: str) -> dict[str, str]:
    return {key: value for (key, value) in re.findall('(\\S+) "([^"]*)";', text)}

def macaque_normalize_chrom(chrom: str) -> str:
    if chrom.startswith('chr'):
        return chrom
    if chrom == 'MT':
        return 'chrM'
    return f'chr{chrom}'

def macaque_normalize_family(symbol: str) -> str:
    value = symbol.upper().replace('-', '').replace('_', '')
    if 'U4ATAC' in value:
        return 'U4ATAC'
    for family in ('U11', 'U12', 'U1', 'U2', 'U4', 'U5', 'U6'):
        if re.search(f'(^|RNU|RNV){family}', value) or value.startswith(family):
            return family
    return 'UNKNOWN'

@dataclass
class macaque_Gene:
    gene_id: str
    name: str
    chrom: str
    start0: int
    end: int
    strand: str
    biotype: str

    @property
    def center(self) -> float:
        return (self.start0 + self.end) / 2

def macaque_read_gtf_genes(path: Path) -> list[macaque_Gene]:
    genes: list[macaque_Gene] = []
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9 or fields[2] != 'gene':
                continue
            attrs = macaque_parse_attributes(fields[8])
            gene_id = attrs.get('gene_id', '').split('.')[0]
            if not gene_id:
                continue
            genes.append(macaque_Gene(gene_id=gene_id, name=attrs.get('gene_name', gene_id), chrom=macaque_normalize_chrom(fields[0]), start0=int(fields[3]) - 1, end=int(fields[4]), strand=fields[6], biotype=attrs.get('gene_type', attrs.get('gene_biotype', ''))))
    return genes

class macaque_IndexedFasta:

    def __init__(self, fasta: Path):
        self.fasta = fasta
        self.index: dict[str, tuple[int, int, int, int]] = {}
        with Path(str(fasta) + '.fai').open() as handle:
            for line in handle:
                (name, length, offset, line_bases, line_width, *_) = line.rstrip().split('\t')
                self.index[name] = (int(length), int(offset), int(line_bases), int(line_width))

    def fetch(self, chrom: str, start0: int, end: int) -> str:
        (length, offset, line_bases, line_width) = self.index[chrom]
        start0 = max(0, start0)
        end = min(length, end)
        if end <= start0:
            return ''
        byte_start = offset + start0 // line_bases * line_width + start0 % line_bases
        last = end - 1
        byte_end = offset + last // line_bases * line_width + last % line_bases + 1
        with self.fasta.open('rb') as handle:
            handle.seek(byte_start)
            raw = handle.read(byte_end - byte_start)
        return raw.replace(b'\n', b'').replace(b'\r', b'').decode().upper()[:end - start0]

def macaque_reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]

def macaque_read_monkey_candidates(monkey_genes: dict[str, macaque_Gene]) -> list[dict]:
    merged: dict[str, dict] = {}
    for (group, path) in macaque_MONKEY_BEDS.items():
        with path.open() as handle:
            for line in handle:
                fields = line.rstrip().split('\t')
                if len(fields) < 4:
                    continue
                (chrom, start, end, gene_id) = fields[:4]
                (start0, end_i) = (int(start), int(end))
                if gene_id not in merged:
                    merged[gene_id] = {'monkey_gene_id': gene_id, 'current_group': group, 'chrom': macaque_normalize_chrom(chrom), 'start0': start0, 'end': end_i}
                else:
                    merged[gene_id]['start0'] = min(merged[gene_id]['start0'], start0)
                    merged[gene_id]['end'] = max(merged[gene_id]['end'], end_i)
    rows = []
    for (gene_id, row) in merged.items():
        meta = monkey_genes.get(gene_id)
        row['symbol'] = meta.name if meta else gene_id
        row['strand'] = meta.strand if meta else '.'
        row['biotype'] = meta.biotype if meta else ''
        row['family'] = macaque_normalize_family(row['symbol'])
        row['center'] = (row['start0'] + row['end']) / 2
        rows.append(row)
    return sorted(rows, key=lambda x: (x['chrom'], x['start0'], x['end'], x['monkey_gene_id']))

def macaque_write_bed(path: Path, rows: list[dict], interval_key: str) -> None:
    with path.open('w') as handle:
        for row in rows:
            (chrom, start0, end, name, strand) = row[interval_key]
            handle.write(f'{chrom}\t{start0}\t{end}\t{name}\t0\t{strand}\n')

def macaque_run_liftover(path: Path, chain: Path, out_prefix: Path, min_match: float) -> list[dict]:
    mapped = out_prefix.with_suffix('.mapped.bed')
    unmapped = out_prefix.with_suffix('.unmapped.bed')
    subprocess.run(['liftOver', f'-minMatch={min_match}', str(path), str(chain), str(mapped), str(unmapped)], check=True)
    rows: list[dict] = []
    with mapped.open() as handle:
        for line in handle:
            fields = line.rstrip().split('\t')
            rows.append({'chrom': fields[0], 'start0': int(fields[1]), 'end': int(fields[2]), 'name': fields[3], 'strand': fields[5] if len(fields) > 5 else '.'})
    return rows

def macaque_overlap_metrics(a_start: int, a_end: int, b_start: int, b_end: int) -> tuple[int, float]:
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    if overlap == 0:
        return (0, 0.0)
    reciprocal = min(overlap / (a_end - a_start), overlap / (b_end - b_start))
    return (overlap, reciprocal)

def macaque_candidate_overlaps(mapped: dict, family: str, candidates: list[dict]) -> list[dict]:
    hits = []
    for cand in candidates:
        if cand['chrom'] != mapped['chrom'] or cand['family'] not in {family, 'UNKNOWN'}:
            continue
        (overlap, reciprocal) = macaque_overlap_metrics(mapped['start0'], mapped['end'], cand['start0'], cand['end'])
        if overlap:
            hits.append({**cand, 'overlap_bp': overlap, 'reciprocal_overlap': reciprocal})
    return sorted(hits, key=lambda x: (-x['reciprocal_overlap'], -x['overlap_bp'], x['monkey_gene_id']))

def macaque_nearest_candidates(mapped: dict, family: str, candidates: list[dict], max_distance: int) -> list[dict]:
    center = (mapped['start0'] + mapped['end']) / 2
    hits = []
    for cand in candidates:
        if cand['chrom'] != mapped['chrom'] or cand['family'] != family:
            continue
        distance = abs(cand['center'] - center)
        if distance <= max_distance:
            hits.append({**cand, 'distance_bp': int(round(distance))})
    return sorted(hits, key=lambda x: (x['distance_bp'], x['monkey_gene_id']))

def macaque_write_query_fasta(targets: list[dict], human_by_id: dict[str, macaque_Gene], work_dir: Path) -> Path:
    fasta = macaque_IndexedFasta(macaque_HUMAN_FASTA)
    output = work_dir / 'human_strict41.core.fa'
    with output.open('w') as handle:
        for row in targets:
            gene = human_by_id[row['stable_gene_id']]
            seq = fasta.fetch(row['chrom'], int(row['start']), int(row['end']))
            if gene.strand == '-':
                seq = macaque_reverse_complement(seq)
            handle.write(f">{row['stable_gene_id']}|{row['display_name']}|{row['family']}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i:i + 80] + '\n')
    return output

def macaque_read_fasta_records(path: Path) -> list[tuple[str, str]]:
    opener = gzip.open if path.suffix == '.gz' else open
    records: list[tuple[str, str]] = []
    header = ''
    chunks: list[str] = []
    with opener(path, 'rt') as handle:
        for line in handle:
            if line.startswith('>'):
                if header:
                    records.append((header, ''.join(chunks).upper()))
                header = line[1:].rstrip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header:
            records.append((header, ''.join(chunks).upper()))
    return records

def macaque_write_monkey_candidate_fasta(candidates: list[dict], work_dir: Path) -> tuple[Path, dict[str, dict]]:
    allowed = {row['monkey_gene_id'] for row in candidates}
    meta: dict[str, dict] = {}
    output = work_dir / 'rhesus_current_rnu_candidates.fa'
    with output.open('w') as handle:
        for (header, seq) in macaque_read_fasta_records(macaque_MONKEY_NCRNA_FASTA):
            match = re.search('\\bgene:(ENSMMUG\\d+)(?:\\.\\d+)?', header)
            if not match or match.group(1) not in allowed:
                continue
            gene_id = match.group(1)
            transcript_id = header.split()[0].split('.')[0]
            symbol_match = re.search('\\bgene_symbol:([^ ]+)', header)
            symbol = symbol_match.group(1) if symbol_match else gene_id
            subject_id = f'{gene_id}|{transcript_id}|{symbol}'
            handle.write(f'>{subject_id}\n')
            for i in range(0, len(seq), 80):
                handle.write(seq[i:i + 80] + '\n')
            meta[subject_id] = {'monkey_gene_id': gene_id, 'monkey_transcript_id': transcript_id, 'monkey_symbol': symbol, 'family': macaque_normalize_family(symbol)}
    return (output, meta)

def macaque_run_blast(query: Path, subject_fasta: Path, subject_meta: dict[str, dict], work_dir: Path) -> list[dict]:
    db = work_dir / 'rhesus_rnu_db'
    subprocess.run(['makeblastdb', '-in', str(subject_fasta), '-dbtype', 'nucl', '-out', str(db)], check=True, capture_output=True, text=True)
    out = work_dir / 'human_vs_rhesus_rnu.blast.tsv'
    fmt = '6 qseqid sseqid pident length qlen slen qstart qend sstart send bitscore evalue'
    subprocess.run(['blastn', '-task', 'blastn-short', '-query', str(query), '-db', str(db), '-word_size', '7', '-reward', '1', '-penalty', '-1', '-gapopen', '2', '-gapextend', '1', '-evalue', '100', '-max_target_seqs', '500', '-outfmt', fmt, '-out', str(out)], check=True)
    columns = fmt.split()[1:]
    rows: list[dict] = []
    with out.open() as handle:
        for line in handle:
            values = line.rstrip().split('\t')
            rec = dict(zip(columns, values))
            (q_gene, q_name, q_family) = rec['qseqid'].split('|', 2)
            s_meta = subject_meta.get(rec['sseqid'])
            if not s_meta or s_meta['family'] not in {q_family, 'UNKNOWN'}:
                continue
            (length, qlen) = (int(rec['length']), int(rec['qlen']))
            query_aligned_bases = abs(int(rec['qend']) - int(rec['qstart'])) + 1
            rows.append({'human_gene_id': q_gene, 'human_display_name': q_name, 'human_family': q_family, **s_meta, 'pident': float(rec['pident']), 'alignment_length': length, 'query_length': qlen, 'query_coverage': query_aligned_bases / qlen, 'bitscore': float(rec['bitscore']), 'evalue': float(rec['evalue'])})
    rows.sort(key=lambda x: (x['human_gene_id'], -x['bitscore'], -x['query_coverage'], -x['pident'], x['monkey_gene_id']))
    return rows
macaque_file_sha256 = _shared.sha256_file

def macaque_build_neighbor_table(targets: list[dict], calls: list[dict], human_genes: list[macaque_Gene], monkey_genes: list[macaque_Gene]) -> list[dict]:
    human_pc = defaultdict(list)
    monkey_pc = defaultdict(list)
    for gene in human_genes:
        if gene.biotype == 'protein_coding':
            human_pc[gene.chrom].append(gene)
    for gene in monkey_genes:
        if gene.biotype == 'protein_coding':
            monkey_pc[gene.chrom].append(gene)
    for values in (*human_pc.values(), *monkey_pc.values()):
        values.sort(key=lambda x: x.start0)
    call_by_id = {row['human_gene_id']: row for row in calls}
    output: list[dict] = []
    for target in targets:
        call = call_by_id[target['stable_gene_id']]
        h_center = (int(target['start']) + int(target['end'])) / 2
        synteny_chrom = call.get('synteny_chrom', call['monkey_chrom'])
        synteny_center_value = call.get('synteny_center', call['monkey_center'])
        m_center = float(synteny_center_value) if synteny_center_value != '' else None
        h_near = sorted(human_pc[target['chrom']], key=lambda g: abs(g.center - h_center))[:12]
        m_near = sorted(monkey_pc[synteny_chrom], key=lambda g: abs(g.center - m_center))[:12] if m_center is not None else []
        shared = set((g.name for g in h_near)) & set((g.name for g in m_near))
        for (species, genes, center) in (('human', h_near, h_center), ('rhesus', m_near, m_center)):
            for (rank, gene) in enumerate(genes, 1):
                output.append({'human_gene_id': target['stable_gene_id'], 'target_name': target['display_name'], 'species': species, 'neighbor_rank': rank, 'gene_id': gene.gene_id, 'gene_name': gene.name, 'chrom': gene.chrom, 'start0': gene.start0, 'end': gene.end, 'strand': gene.strand, 'distance_to_target_bp': int(round(gene.center - center)), 'shared_symbol_anchor': gene.name in shared})
    return output

def macaque_main() -> None:
    args = macaque_parse_args()
    macaque_require_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    targets = macaque_read_tsv(macaque_STRICT41)
    if len(targets) != 41 or len({r['stable_gene_id'] for r in targets}) != 41:
        raise ValueError('The frozen strict input must contain exactly 41 unique genes')
    human_genes = macaque_read_gtf_genes(macaque_HUMAN_GTF)
    monkey_genes = macaque_read_gtf_genes(macaque_MONKEY_GTF)
    human_by_id = {g.gene_id: g for g in human_genes}
    monkey_by_id = {g.gene_id: g for g in monkey_genes}
    missing_human = [r['stable_gene_id'] for r in targets if r['stable_gene_id'] not in human_by_id]
    if missing_human:
        raise ValueError(f'Strict41 IDs missing from human GTF: {missing_human}')
    candidates = macaque_read_monkey_candidates(monkey_by_id)
    for row in targets:
        gene = human_by_id[row['stable_gene_id']]
        row['strand'] = gene.strand
        row['core_interval'] = (row['chrom'], int(row['start']), int(row['end']), row['stable_gene_id'], gene.strand)
    all_core: list[dict] = []
    core_maps: dict[float, dict[str, dict]] = {}
    method_rows: list[dict] = []
    for threshold in macaque_CORE_THRESHOLDS:
        label = f'core_minMatch_{threshold:.2f}'
        bed = args.work_dir / f'{label}.hg38.bed'
        macaque_write_bed(bed, targets, 'core_interval')
        mapped = macaque_run_liftover(bed, macaque_CHAIN, args.work_dir / label, threshold)
        map_by_id = {row['name']: row for row in mapped}
        core_maps[threshold] = map_by_id
        supported = set()
        for target in targets:
            mapping = map_by_id.get(target['stable_gene_id'])
            if mapping is None:
                all_core.append({'method': label, 'min_match': threshold, 'human_gene_id': target['stable_gene_id'], 'human_display_name': target['display_name'], 'human_family': target['family'], 'lifted': False, 'mapped_chrom': '', 'mapped_start0': '', 'mapped_end': '', 'mapped_strand': '', 'candidate_gene_ids': '', 'best_reciprocal_overlap': 0})
                continue
            hits = macaque_candidate_overlaps(mapping, target['family'], candidates)
            qualifying = [h for h in hits if h['reciprocal_overlap'] >= macaque_DIRECT_RECIPROCAL_MIN]
            if qualifying:
                supported.add(target['stable_gene_id'])
            all_core.append({'method': label, 'min_match': threshold, 'human_gene_id': target['stable_gene_id'], 'human_display_name': target['display_name'], 'human_family': target['family'], 'lifted': True, 'mapped_chrom': mapping['chrom'], 'mapped_start0': mapping['start0'], 'mapped_end': mapping['end'], 'mapped_strand': mapping['strand'], 'candidate_gene_ids': ';'.join((h['monkey_gene_id'] for h in qualifying)), 'best_reciprocal_overlap': qualifying[0]['reciprocal_overlap'] if qualifying else 0})
        method_rows.append({'method': label, 'evidence_type': 'core chain + same-family annotation overlap', 'genes_with_coordinate_mapping': len(mapped), 'genes_with_candidate_relationship': len(supported), 'selection_role': 'comparison'})
    extended_maps: dict[int, dict[str, dict]] = {}
    all_extended: list[dict] = []
    for pad in macaque_EXTENDED_PADS:
        label = f'extended_{pad}bp_minMatch_0.50'
        prepared = []
        for target in targets:
            gene = human_by_id[target['stable_gene_id']]
            prepared.append({**target, 'extended_interval': (target['chrom'], max(0, int(target['start']) - pad), int(target['end']) + pad, target['stable_gene_id'], gene.strand)})
        bed = args.work_dir / f'{label}.hg38.bed'
        macaque_write_bed(bed, prepared, 'extended_interval')
        mapped = macaque_run_liftover(bed, macaque_CHAIN, args.work_dir / label, 0.5)
        map_by_id = {row['name']: row for row in mapped}
        extended_maps[pad] = map_by_id
        supported = set()
        for target in targets:
            mapping = map_by_id.get(target['stable_gene_id'])
            if mapping is None:
                all_extended.append({'method': label, 'pad_bp': pad, 'human_gene_id': target['stable_gene_id'], 'human_display_name': target['display_name'], 'human_family': target['family'], 'lifted': False, 'mapped_chrom': '', 'mapped_start0': '', 'mapped_end': '', 'mapped_strand': '', 'nearest_candidate_gene_ids': '', 'nearest_distance_bp': ''})
                continue
            near = macaque_nearest_candidates(mapping, target['family'], candidates, macaque_MAX_NEAREST_BP)
            if near:
                supported.add(target['stable_gene_id'])
            all_extended.append({'method': label, 'pad_bp': pad, 'human_gene_id': target['stable_gene_id'], 'human_display_name': target['display_name'], 'human_family': target['family'], 'lifted': True, 'mapped_chrom': mapping['chrom'], 'mapped_start0': mapping['start0'], 'mapped_end': mapping['end'], 'mapped_strand': mapping['strand'], 'nearest_candidate_gene_ids': ';'.join((h['monkey_gene_id'] for h in near[:5])), 'nearest_distance_bp': near[0]['distance_bp'] if near else ''})
        method_rows.append({'method': label, 'evidence_type': 'extended conserved locus + nearest same-family annotation', 'genes_with_coordinate_mapping': len(mapped), 'genes_with_candidate_relationship': len(supported), 'selection_role': 'comparison'})
    flank_predictions: dict[tuple[int, int], dict[str, dict]] = {}
    all_flank: list[dict] = []
    for (inner, outer) in macaque_FLANK_SCHEMES:
        label = f'independent_flanks_{inner}_{outer}bp_minMatch_0.50'
        anchors = []
        human_anchor_centers: dict[str, float] = {}
        human_core_centers: dict[str, float] = {}
        for target in targets:
            (start0, end) = (int(target['start']), int(target['end']))
            core_center = (start0 + end) / 2
            intervals = {'L': (max(0, start0 - outer), max(0, start0 - inner)), 'R': (end + inner, end + outer)}
            for (side, (anchor_start, anchor_end)) in intervals.items():
                if anchor_end <= anchor_start:
                    continue
                name = f"{target['stable_gene_id']}|{side}"
                human_anchor_centers[name] = (anchor_start + anchor_end) / 2
                human_core_centers[target['stable_gene_id']] = core_center
                anchors.append({'anchor_interval': (target['chrom'], anchor_start, anchor_end, name, '+')})
        bed = args.work_dir / f'{label}.hg38.bed'
        macaque_write_bed(bed, anchors, 'anchor_interval')
        mapped_anchors = macaque_run_liftover(bed, macaque_CHAIN, args.work_dir / label, 0.5)
        predicted_by_gene: dict[str, list[dict]] = defaultdict(list)
        for anchor in mapped_anchors:
            (human_id, side) = anchor['name'].split('|', 1)
            h_anchor = human_anchor_centers[anchor['name']]
            h_core = human_core_centers[human_id]
            m_anchor = (anchor['start0'] + anchor['end']) / 2
            signed_offset = h_core - h_anchor
            predicted_center = m_anchor + signed_offset if anchor['strand'] == '+' else m_anchor - signed_offset
            predicted_by_gene[human_id].append({'chrom': anchor['chrom'], 'predicted_center': predicted_center, 'chain_strand': anchor['strand'], 'side': side})
        predictions: dict[str, dict] = {}
        supported = set()
        for target in targets:
            human_id = target['stable_gene_id']
            anchors_for_gene = predicted_by_gene.get(human_id, [])
            usable = []
            by_chrom: dict[str, list[dict]] = defaultdict(list)
            for rec in anchors_for_gene:
                by_chrom[rec['chrom']].append(rec)
            if by_chrom:
                (best_chrom, usable) = max(by_chrom.items(), key=lambda item: len(item[1]))
                centers = [x['predicted_center'] for x in usable]
                trial_centers = centers + ([sum(centers) / len(centers)] if len(centers) > 1 and max(centers) - min(centers) <= 200000 else [])
                trials = []
                core_len = int(target['end']) - int(target['start'])
                for center in trial_centers:
                    trial = {'chrom': best_chrom, 'start0': max(0, int(round(center - core_len / 2))), 'end': max(1, int(round(center + core_len / 2))), 'strand': usable[0]['chain_strand']}
                    near = macaque_nearest_candidates(trial, target['family'], candidates, macaque_MAX_NEAREST_BP)
                    trials.append((near[0]['distance_bp'] if near else 10 ** 18, trial, near))
                (_, prediction, near) = min(trials, key=lambda item: item[0])
                predictions[human_id] = prediction
                if near:
                    supported.add(human_id)
                all_flank.append({'method': label, 'inner_gap_bp': inner, 'outer_gap_bp': outer, 'human_gene_id': human_id, 'human_display_name': target['display_name'], 'human_family': target['family'], 'mapped_anchor_count': len(anchors_for_gene), 'predicted_chrom': prediction['chrom'], 'predicted_start0': prediction['start0'], 'predicted_end': prediction['end'], 'chain_strand': prediction['strand'], 'nearest_candidate_gene_ids': ';'.join((h['monkey_gene_id'] for h in near[:5])), 'nearest_distance_bp': near[0]['distance_bp'] if near else ''})
            else:
                all_flank.append({'method': label, 'inner_gap_bp': inner, 'outer_gap_bp': outer, 'human_gene_id': human_id, 'human_display_name': target['display_name'], 'human_family': target['family'], 'mapped_anchor_count': 0, 'predicted_chrom': '', 'predicted_start0': '', 'predicted_end': '', 'chain_strand': '', 'nearest_candidate_gene_ids': '', 'nearest_distance_bp': ''})
        flank_predictions[inner, outer] = predictions
        method_rows.append({'method': label, 'evidence_type': 'independent conserved flanks + nearest same-family annotation', 'genes_with_coordinate_mapping': len(predictions), 'genes_with_candidate_relationship': len(supported), 'selection_role': 'comparison'})
    query = macaque_write_query_fasta(targets, human_by_id, args.work_dir)
    (monkey_fasta, subject_meta) = macaque_write_monkey_candidate_fasta(candidates, args.work_dir)
    blast_rows = macaque_run_blast(query, monkey_fasta, subject_meta, args.work_dir)
    relaxed_blast = [r for r in blast_rows if r['pident'] >= 65 and r['query_coverage'] >= 0.6]
    blast_by_query: dict[str, list[dict]] = defaultdict(list)
    blast_by_pair: dict[tuple[str, str], dict] = {}
    for row in relaxed_blast:
        if row['family'] == row['human_family']:
            blast_by_query[row['human_gene_id']].append(row)
        key = (row['human_gene_id'], row['monkey_gene_id'])
        if key not in blast_by_pair or row['bitscore'] > blast_by_pair[key]['bitscore']:
            blast_by_pair[key] = row
    method_rows.append({'method': 'relaxed_BLASTN_65pctID_60pctQcov', 'evidence_type': 'same-family core-sequence similarity only', 'genes_with_coordinate_mapping': 0, 'genes_with_candidate_relationship': len(blast_by_query), 'selection_role': 'comparison; not sufficient alone for orthology'})
    final_calls: list[dict] = []
    for target in targets:
        human_id = target['stable_gene_id']
        fallback_synteny_mapping = next((flank_predictions[scheme][human_id] for scheme in macaque_FLANK_SCHEMES if human_id in flank_predictions[scheme]), None)
        chosen: list[dict] = []
        method = ''
        confidence = ''
        relationship = ''
        mapping = core_maps[0.2].get(human_id)
        if mapping:
            direct = [h for h in macaque_candidate_overlaps(mapping, target['family'], candidates) if h['reciprocal_overlap'] >= macaque_DIRECT_RECIPROCAL_MIN]
            if direct:
                best_overlap = direct[0]['reciprocal_overlap']
                chosen = [h for h in direct if h['reciprocal_overlap'] >= best_overlap - 0.05]
                method = 'core liftOver minMatch=0.20 + same-family overlap'
                confidence = 'high' if best_overlap >= 0.7 else 'moderate'
                relationship = 'one-to-one candidate' if len(chosen) == 1 else 'array-level co-ortholog candidates'
        if not chosen:
            for (inner, outer) in macaque_FLANK_SCHEMES:
                mapping = flank_predictions[inner, outer].get(human_id)
                if not mapping:
                    continue
                near = macaque_nearest_candidates(mapping, target['family'], candidates, macaque_MAX_NEAREST_BP)
                if near:
                    min_dist = near[0]['distance_bp']
                    shortlist = [h for h in near if h['distance_bp'] <= min_dist + 2000]
                    with_seq = [h for h in shortlist if (human_id, h['monkey_gene_id']) in blast_by_pair]
                    chosen = with_seq if with_seq else shortlist[:1]
                    method = f'independent flank-anchor liftOver ({inner:,}–{outer:,} bp) + nearest same-family annotation'
                    confidence = 'moderate' if with_seq and len(flank_predictions[inner, outer]) else 'low'
                    relationship = 'one-to-one candidate' if len(chosen) == 1 else 'array-level co-ortholog candidates'
                    break
        if not chosen:
            for pad in macaque_EXTENDED_PADS:
                mapping = extended_maps[pad].get(human_id)
                if not mapping:
                    continue
                near = macaque_nearest_candidates(mapping, target['family'], candidates, macaque_MAX_NEAREST_BP)
                if near:
                    min_dist = near[0]['distance_bp']
                    shortlist = [h for h in near if h['distance_bp'] <= min_dist + 2000]
                    with_seq = [h for h in shortlist if (human_id, h['monkey_gene_id']) in blast_by_pair]
                    chosen = with_seq if with_seq else shortlist[:1]
                    method = f'extended-locus liftOver ±{pad:,} bp + nearest same-family annotation'
                    confidence = 'moderate' if with_seq else 'low'
                    relationship = 'one-to-one candidate' if len(chosen) == 1 else 'array-level co-ortholog candidates'
                    break
        if not chosen:
            sequence_hits = blast_by_query.get(human_id, [])
            if sequence_hits:
                best_score = sequence_hits[0]['bitscore']
                top = [h for h in sequence_hits if h['bitscore'] >= best_score * 0.98]
                chosen_ids = []
                for hit in top:
                    if hit['monkey_gene_id'] not in chosen_ids:
                        chosen_ids.append(hit['monkey_gene_id'])
                cand_by_id = {c['monkey_gene_id']: c for c in candidates}
                chosen = [cand_by_id[x] for x in chosen_ids[:10] if x in cand_by_id]
                method = 'relaxed same-family BLASTN only (no positional support)'
                confidence = 'low'
                relationship = 'family-level homolog candidates; orthology unresolved'
                mapping = None
        if chosen:
            if mapping:
                (m_chrom, m_start, m_end) = (mapping['chrom'], mapping['start0'], mapping['end'])
                m_center = (m_start + m_end) / 2
                m_strand = mapping['strand']
            else:
                m_chrom = chosen[0]['chrom']
                (m_start, m_end) = (chosen[0]['start0'], chosen[0]['end'])
                m_center = chosen[0]['center']
                m_strand = chosen[0]['strand']
            seq_evidence = [blast_by_pair[human_id, h['monkey_gene_id']] for h in chosen if (human_id, h['monkey_gene_id']) in blast_by_pair]
            best_seq = max(seq_evidence, key=lambda x: x['bitscore'], default=None)
            selected_synteny_mapping = mapping if 'BLASTN only' not in method else fallback_synteny_mapping
            final_calls.append({'human_gene_id': human_id, 'human_display_name': target['display_name'], 'human_family': target['family'], 'human_chrom': target['chrom'], 'human_start0': target['start'], 'human_end': target['end'], 'human_strand': target['strand'], 'relationship_class': relationship, 'confidence': confidence, 'selected_method': method, 'monkey_gene_ids': ';'.join((h['monkey_gene_id'] for h in chosen)), 'monkey_symbols': ';'.join((h['symbol'] for h in chosen)), 'monkey_chrom': m_chrom, 'monkey_start0': int(m_start), 'monkey_end': int(m_end), 'monkey_center': int(round(m_center)), 'monkey_mapped_strand': m_strand, 'candidate_count': len(chosen), 'synteny_chrom': selected_synteny_mapping['chrom'] if selected_synteny_mapping else '', 'synteny_start0': selected_synteny_mapping['start0'] if selected_synteny_mapping else '', 'synteny_end': selected_synteny_mapping['end'] if selected_synteny_mapping else '', 'synteny_center': int(round((selected_synteny_mapping['start0'] + selected_synteny_mapping['end']) / 2)) if selected_synteny_mapping else '', 'synteny_evidence': 'selected positional orthology' if 'BLASTN only' not in method else 'flank-projected region without a core/candidate match', 'best_sequence_identity_pct': best_seq['pident'] if best_seq else '', 'best_sequence_query_coverage': best_seq['query_coverage'] if best_seq else '', 'notes': 'Positional orthology is array-level where copy number or tandem repetition prevents a unique pair.' if 'array' in relationship else ''})
        else:
            final_calls.append({'human_gene_id': human_id, 'human_display_name': target['display_name'], 'human_family': target['family'], 'human_chrom': target['chrom'], 'human_start0': target['start'], 'human_end': target['end'], 'human_strand': target['strand'], 'relationship_class': 'unresolved', 'confidence': 'unresolved', 'selected_method': 'no method passed relaxed criteria', 'monkey_gene_ids': '', 'monkey_symbols': '', 'monkey_chrom': '', 'monkey_start0': '', 'monkey_end': '', 'monkey_center': '', 'monkey_mapped_strand': '', 'candidate_count': 0, 'synteny_chrom': fallback_synteny_mapping['chrom'] if fallback_synteny_mapping else '', 'synteny_start0': fallback_synteny_mapping['start0'] if fallback_synteny_mapping else '', 'synteny_end': fallback_synteny_mapping['end'] if fallback_synteny_mapping else '', 'synteny_center': int(round((fallback_synteny_mapping['start0'] + fallback_synteny_mapping['end']) / 2)) if fallback_synteny_mapping else '', 'synteny_evidence': 'flank-projected region without a core/candidate match' if fallback_synteny_mapping else '', 'best_sequence_identity_pct': '', 'best_sequence_query_coverage': '', 'notes': ''})
    monkey_to_human: dict[str, set[str]] = defaultdict(set)
    for row in final_calls:
        for monkey_id in row['monkey_gene_ids'].split(';'):
            if monkey_id:
                monkey_to_human[monkey_id].add(row['human_gene_id'])
    reused = {monkey_id for (monkey_id, human_ids) in monkey_to_human.items() if len(human_ids) > 1}
    for row in final_calls:
        ids = {x for x in row['monkey_gene_ids'].split(';') if x}
        if ids & reused and 'orthology unresolved' not in row['relationship_class']:
            row['relationship_class'] = 'many-to-one / array-level co-ortholog candidate'
            row['notes'] = 'The selected rhesus locus is shared by multiple human array members; a unique one-to-one ortholog is not supported.'
    local_search_path = args.output_dir / 'rheMac10_chr13_local_sequence_search.tsv'
    if local_search_path.exists():
        local_rows = macaque_read_tsv(local_search_path)
        focal = [row for row in local_rows if row.get('human_display_name') == 'RNVU1-24']
        if focal:
            best_qcov = max((float(row['query_coverage']) for row in focal))
            for row in final_calls:
                if row['human_display_name'] == 'RNVU1-24':
                    row['notes'] = f'Both independent flanks project to rheMac10 chr13 near 103.147 Mb, but the ±101-kb local sequence search found no core match at >=60% query coverage (best HSP qcov={best_qcov:.3f}); the reported chr18 U1 hit is family homology only, not a resolved positional ortholog.'
    final_supported = sum((row['relationship_class'] != 'unresolved' for row in final_calls))
    final_synteny = sum(('BLASTN only' not in row['selected_method'] and row['relationship_class'] != 'unresolved' for row in final_calls))
    method_rows.append({'method': 'selected_synteny_first_ensemble', 'evidence_type': 'core/flank/extended synteny, annotation and sequence tie-break', 'genes_with_coordinate_mapping': final_synteny, 'genes_with_candidate_relationship': final_supported, 'selection_role': 'selected final method'})
    neighbor_rows = macaque_build_neighbor_table(targets, final_calls, human_genes, monkey_genes)
    macaque_write_tsv(args.output_dir / 'human_strict41_input.tsv', [{k: v for (k, v) in r.items() if k != 'core_interval'} for r in targets])
    macaque_write_tsv(args.output_dir / 'monkey_rnu_candidate_catalog.tsv', [{k: v for (k, v) in r.items() if k != 'center'} for r in candidates])
    macaque_write_tsv(args.output_dir / 'method_comparison.tsv', method_rows)
    macaque_write_tsv(args.output_dir / 'core_liftover_all_thresholds.tsv', all_core)
    macaque_write_tsv(args.output_dir / 'extended_liftover_all_pads.tsv', all_extended)
    macaque_write_tsv(args.output_dir / 'independent_flank_liftover_all_schemes.tsv', all_flank)
    macaque_write_tsv(args.output_dir / 'relaxed_blastn_same_family_hits.tsv', relaxed_blast, fieldnames=list(blast_rows[0]) if blast_rows else [])
    macaque_write_tsv(args.output_dir / 'orthology_calls.tsv', final_calls)
    macaque_write_tsv(args.output_dir / 'synteny_neighbor_genes.tsv', neighbor_rows)
    input_paths = [macaque_STRICT41, macaque_HUMAN_GTF, macaque_HUMAN_FASTA, Path(str(macaque_HUMAN_FASTA) + '.fai'), macaque_MONKEY_GTF, macaque_MONKEY_NCRNA_FASTA, macaque_CHAIN, *macaque_MONKEY_BEDS.values()]
    local_window_json = macaque_ROOT / 'tmp/rnu_pol2_orthology_20260814/source/rheMac10_chr13_103046000_103248000.json'
    if local_window_json.exists():
        input_paths.append(local_window_json)
    inventory = []
    for path in input_paths:
        inventory.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': macaque_file_sha256(path), 'read_only_source': True})
    macaque_write_tsv(args.output_dir / 'input_inventory.tsv', inventory)
    summary = [{'metric': 'human_strict41_loci', 'value': len(targets)}, {'metric': 'rhesus_current_candidate_loci', 'value': len(candidates)}, {'metric': 'final_candidate_relationships', 'value': final_supported}, {'metric': 'final_with_positional_synteny', 'value': final_synteny}, {'metric': 'final_sequence_only', 'value': final_supported - final_synteny}, {'metric': 'final_unresolved', 'value': len(targets) - final_supported}]
    macaque_write_tsv(args.output_dir / 'analysis_summary.tsv', summary)
    print(f'Strict human loci: {len(targets)}')
    print(f'Current rhesus candidate loci: {len(candidates)}')
    print(f'Final candidate relationships: {final_supported}/41')
    print(f'Final calls with positional synteny: {final_synteny}/41')
    print(f'Outputs: {args.output_dir}')

def macaque_entrypoint():
    macaque_main()

def _prepare_runtime():
    pass

if __name__ == '__main__':
    _run_stages({'macaque': 'macaque_entrypoint'}, globals(), setup=_prepare_runtime)
