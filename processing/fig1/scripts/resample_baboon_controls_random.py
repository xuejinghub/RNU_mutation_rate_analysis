#!/usr/bin/env python3
"""Draw Baboon controls once, without selecting on mutation outcomes.

The frozen candidate pool and current Fig. 1 genes are reused. No VCFs are
recounted. This is a retrospective reanalysis, not an independent validation.
"""

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import platform
import secrets

import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT_AUDIT = HERE.parent / "data/processed/fig1_baboon_callable_20260927/baboon_control_candidate_audit.tsv"
DEFAULT_DATA = ROOT / "figures/Fig1/data"
DEFAULT_OUTPUT = HERE.parent / "data/processed/fig1_baboon_random_control"
GROUPS = ["pol2_spliceRNUs", "functional_U6s", "RNU_pseudogenes", "genome_wide_neutral"]
CONTROL = GROUPS[-1]
TABLES = ["panel_a_gene_locus_density.tsv", "panel_a_pairwise_statistics.tsv",
          "panel_b_control_mean_ratios.tsv", "panel_c_profile_control_ratios.tsv",
          "control_denominators.tsv", "group_counts.tsv"]
ELIGIBILITY = ["unit_id", "refseq_chrom", "start", "end", "raw_callable_bp",
               "ACGT_only", "outside_gene_buffer10kb", "fully_callable",
               "effective_callable_bp"]
POLICY = {"control_count": 1000, "length_bp": 150, "minimum_effective_bp": 50,
          "candidate_order": "unit_id ascending", "bit_generator": "PCG64",
          "sampling": "first 1000 of one permutation; without replacement",
          "outcome_based_selection": False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, **kwargs):
    return pd.read_csv(path, sep="\t", float_precision="round_trip", **kwargs)


def save_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def save_table(path, table):
    table.to_csv(path, sep="\t", index=False, na_rep="NA", mode="x")


def eligible_candidates(path):
    # Deliberately exclude mutation counts, densities and old selection ranks.
    candidates = read(path, usecols=ELIGIBILITY)
    require(candidates.unit_id.notna().all() and candidates.unit_id.is_unique,
            "Candidate IDs must be nonempty and unique.")
    for name in ["ACGT_only", "outside_gene_buffer10kb", "fully_callable"]:
        parsed = candidates[name].astype(str).str.lower().map({"true": True, "false": False})
        require(parsed.notna().all(), f"Invalid boolean values: {name}")
        candidates[name] = parsed
    for name in ["start", "end", "raw_callable_bp", "effective_callable_bp"]:
        candidates[name] = pd.to_numeric(candidates[name], errors="raise")
    selected = candidates.loc[
        candidates.fully_callable & candidates.ACGT_only & candidates.outside_gene_buffer10kb
        & candidates.raw_callable_bp.eq(POLICY["length_bp"])
        & (candidates.end - candidates.start).eq(POLICY["length_bp"])
        & candidates.effective_callable_bp.ge(POLICY["minimum_effective_bp"])
    ].sort_values("unit_id").reset_index(drop=True)
    require(len(selected) >= POLICY["control_count"], "Not enough eligible controls.")
    require(selected.refseq_chrom.notna().all(), "Missing reference contigs.")
    require(np.isfinite(selected[["start", "end", "effective_callable_bp"]]).all().all(),
            "Nonfinite coordinates or effective lengths.")
    require((selected.start >= 0).all() and (selected.effective_callable_bp <= 150).all(),
            "Invalid coordinates or effective lengths.")
    require((selected[["start", "end"]] % 1 == 0).all().all(), "Noninteger coordinates.")
    for _, group in selected.sort_values(["refseq_chrom", "start"]).groupby("refseq_chrom"):
        require(np.all(group.start.to_numpy()[1:] >= group.end.to_numpy()[:-1]),
                "Eligible candidate intervals overlap.")
    return candidates, selected


def choose_ids(pool, seed):
    # Exactly one random permutation. No density, ratio or P value is an input.
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(pool))
    return pool.iloc[order[:POLICY["control_count"]]].unit_id.tolist()


def new_directory(path):
    target = Path(path).absolute()
    for part in [target, *target.parents]:
        require(not part.is_symlink(), f"Refusing output through a symlink: {part}")
    # Reserve the directory before obtaining entropy. Existing runs never reroll.
    target.mkdir(parents=True, exist_ok=False)
    return target


def seed_plan(path, replay_plan=None):
    """Persist one seed before selection; later stages reuse the same seed."""
    path = Path(path)
    if path.exists():
        if replay_plan is not None:
            raise FileExistsError(f"Seed plan already exists: {path}")
        return json.loads(path.read_text())
    previous = json.loads(Path(replay_plan).read_text()) if replay_plan else None
    seed = int(previous["seed"]) if previous else secrets.randbits(128)
    plan = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": str(seed), "seed_source": "recorded_plan_replay" if previous else "system_entropy_128_bit",
            "policy": POLICY, "numpy_version": np.__version__,
            "replay_of": str(Path(replay_plan).resolve()) if previous else None}
    save_json(path, plan)
    return plan


def select_controls(candidates, seed):
    """Select from eligible controls once, using only eligibility and a seed."""
    pool = candidates.loc[candidates.effective_callable_bp.ge(POLICY["minimum_effective_bp"])].copy()
    for column in ("fully_callable", "ACGT_only", "outside_gene_buffer10kb"):
        if column in pool:
            pool = pool[pool[column].astype(str).str.lower().eq("true")]
    pool = pool[(pool.end - pool.start).eq(POLICY["length_bp"])]
    if "raw_callable_bp" in pool:
        pool = pool[pool.raw_callable_bp.eq(POLICY["length_bp"])]
    pool = pool.sort_values("unit_id").reset_index(drop=True)
    require(pool.unit_id.is_unique and len(pool) >= POLICY["control_count"], "Insufficient unique eligible controls.")
    ids = choose_ids(pool, seed)
    selected = pool.set_index("unit_id", drop=False).loc[ids].copy().reset_index(drop=True)
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["status"] = "selected_by_single_random_draw"
    return selected


def freeze_draw(audit, data, output, replay_plan=None):
    candidates, pool = eligible_candidates(audit)
    candidate_plan = Path(audit).parent / "seed_plan.json"
    if replay_plan is None and candidate_plan.is_file():
        replay_plan = candidate_plan
    output = new_directory(output)
    plan = seed_plan(output / "seed_plan.json", replay_plan)
    ids = choose_ids(pool, int(plan["seed"]))
    save_table(output / "selected_ids.tsv", pd.DataFrame({"unit_id": ids, "draw_rank": range(1, len(ids) + 1)}))
    return output, plan, {"candidate_count": len(candidates), "eligible_count": len(pool)}


def replace_baboon(source, destination, frame):
    header = read(source, nrows=0).columns
    payload = frame.reindex(columns=header).to_csv(sep="\t", header=False, index=False, na_rep="NA").encode()
    before = Path(source).read_bytes().splitlines(keepends=True)
    written = False
    with Path(destination).open("xb") as handle:
        for line in before:
            if line.startswith(b"Baboon\t"):
                if not written:
                    handle.write(payload)
                    written = True
            else:
                handle.write(line)
    require(written, f"No Baboon rows in {source}")


def divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else np.nan


def aggregate(audit_path, data, output, plan):
    data, output = Path(data), Path(output)
    ids = read(output / "selected_ids.tsv").unit_id.tolist()
    audit = read(audit_path, dtype={"chrom": str})
    controls = audit.set_index("unit_id", drop=False).loc[ids].copy().reset_index(drop=True)
    require(len(controls) == 1000 and controls.unit_id.is_unique, "Control selection is invalid.")
    require(controls.analysis_group.eq(CONTROL).all(), "Unexpected control class.")
    for col in ["rare_snv_alt_count", "rare_snv_density_per_kb", "effective_callable_bp"]:
        require(np.isfinite(controls[col]).all() and controls[col].ge(0).all(), f"Invalid values: {col}")
    require(controls.rare_snv_alt_count.mod(1).eq(0).all(), "Noninteger mutation counts.")
    controls["selection_rank"] = range(1, 1001)
    controls["status"] = "selected_by_single_random_draw"
    controls["species"] = "Baboon"
    controls["assembly"] = "Panubis1.0"
    controls["af_threshold"] = .02
    controls["denominator_basis"] = "coverage_intersect_mappability_ACGT_minus_position_masks"
    controls["source_note"] = (f"Single random draw; seed={plan['seed']}; no outcome-based selection; "
                               "frozen callable candidate pool; GQ20; AF<=0.02")
    controls = controls.sort_values(["refseq_chrom", "start"]).reset_index(drop=True)
    save_table(output / "baboon_control_1000.tsv", controls)
    controls[["refseq_chrom", "start", "end", "unit_id"]].astype({"start": int, "end": int}).to_csv(
        output / "baboon.Panubis1.0.random_control1000.bed", sep="\t", index=False, header=False, mode="x")
    _, pool = eligible_candidates(audit_path)
    audit["selection_rank"] = audit.unit_id.map(dict(zip(ids, range(1, 1001))))
    audit["status"] = "ineligible"
    audit.loc[audit.unit_id.isin(pool.unit_id), "status"] = "eligible_not_selected"
    audit.loc[audit.unit_id.isin(ids), "status"] = "selected_by_single_random_draw"
    save_table(output / "baboon_control_candidate_audit.tsv", audit)

    original = read(data / TABLES[0])
    genes = original.loc[original.species.eq("Baboon") & original.analysis_group.ne(CONTROL)].copy()
    d = pd.concat([genes, controls.reindex(columns=genes.columns)], ignore_index=True)
    save_table(output / "baboon_gene_locus_density.tsv", d)
    cm = float(controls.rare_snv_density_per_kb.mean())
    cp = float(1000 * controls.rare_snv_alt_count.sum() / controls.effective_callable_bp.sum())
    bars, stats = [], []
    for group in GROUPS:
        values = d.loc[d.analysis_group.eq(group), "rare_snv_density_per_kb"]
        bars.append(dict(species="Baboon", analysis_group=group, n_units=len(values),
                         mean_density_per_kb=values.mean(), mean_density_ratio_to_control=divide(values.mean(), cm)))
    for g1, g2 in combinations(GROUPS, 2):
        x, y = [d.loc[d.analysis_group.eq(g), "rare_snv_density_per_kb"] for g in (g1, g2)]
        test = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic", use_continuity=True)
        stats.append(dict(species="Baboon", group_1=g1, group_2=g2, n1=len(x), n2=len(y),
                          U=test.statistic, p_value=test.pvalue, mean_density_fold=divide(x.mean(), y.mean())))
    profile = read(data / TABLES[3]).query('species == "Baboon"').copy()
    profile["density_ratio_to_control"] = profile.pooled_density_per_kb / cp if cp > 0 else np.nan
    save_table(output / "baboon_profile_summary.tsv", profile)
    denominator = pd.DataFrame([dict(species="Baboon", control_mean_density_per_kb=cm,
                                    control_pooled_density_per_kb=cp, denominator_column="effective_callable_bp")])
    counts = d.groupby(["species", "analysis_group"]).size().rename("n").reset_index()
    frames = [d, pd.DataFrame(stats), pd.DataFrame(bars), profile, denominator, counts]
    for name, frame in zip(TABLES, frames):
        replace_baboon(data / name, output / name, frame)
    group_summary = d.groupby("analysis_group").agg(n_after=("unit_id", "size"),
        rare_ALT_SNVs=("rare_snv_alt_count", "sum"), effective_callable_bp=("effective_callable_bp", "sum"))
    save_table(output / "baboon_group_summary.tsv", group_summary.reset_index())
    summary = {"analysis_type": "single_random_draw_control", "seed": plan["seed"], "draws": 1,
               "outcome_based_selection": False, "control_count": 1000,
               "control_mean_density_per_kb": cm, "control_pooled_density_per_kb": cp,
               "ratios_defined": bool(cm > 0 and cp > 0),
               "gene_counts": genes.groupby("analysis_group").size().to_dict(),
               "raw_vcf_recounted": False,
               "candidate_audit": str(Path(audit_path).resolve()), "figure_data": str(data.resolve()),
               "limitation": "One retrospective random draw conditional on the historical candidate pool. "
                             "Do not rerun to select a preferred outcome. Not independent validation.",
               "python": platform.python_version(), "numpy": np.__version__,
               "pandas": pd.__version__, "scipy": scipy.__version__}
    save_json(output / "analysis_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--figure-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay-plan", type=Path, help="Replay an existing seed_plan.json, not a new draw.")
    args = parser.parse_args()
    output, plan, _ = freeze_draw(args.candidate_audit, args.figure_data, args.output, args.replay_plan)
    summary = aggregate(args.candidate_audit, args.figure_data, output, plan)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
