"""Command-line interface for the Asymmetric Balance Search Laboratory."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from hybrid.core.config import VariantConfig
from .metrics import compute_tier0_metrics
from .design import generate_screening_matrix, get_standard_reference_points, FACTORS
from .screening import run_paired_tournament, TournamentConfig
from .analysis import estimate_causal_effects


def main():
    parser = argparse.ArgumentParser(
        description="Asymmetric Board-Game Rule Design and Balance Search Laboratory"
    )
    parser.add_argument(
        "--mode",
        choices=["metrics", "reference", "screen", "full"],
        default="reference",
        help="Execution mode: metrics (Tier 0 only), reference (screen reference presets), screen (32-run factorial matrix), full (Tier 0 + 32-run screen)",
    )
    parser.add_argument("--num-pairs", type=int, default=5, help="Number of paired games per variant (each pair = 2 games, sides swapped)")
    parser.add_argument("--max-plies", type=int, default=150, help="Maximum plies per game before draw truncation")
    parser.add_argument("--agent-a", type=str, default="ab_fast", help="Agent A (default: ab_fast)")
    parser.add_argument("--agent-b", type=str, default="greedy", help="Agent B (default: greedy)")
    parser.add_argument("--sprt", action="store_true", help="Enable SPRT early stopping on severe asymmetry")
    parser.add_argument("--output", type=str, default="runs/balance_screening_results.json", help="Path to write JSON telemetry")
    parser.add_argument("--report", type=str, default="docs/BALANCE_SCREENING_REPORT.md", help="Path to write markdown causal report")

    args = parser.parse_args()

    print(f"=== Hybrid Chess Balance Laboratory (Mode: {args.mode}) ===")

    if args.mode in ("reference", "screen", "full"):
        if args.mode == "reference":
            points = get_standard_reference_points()
        elif args.mode == "screen":
            points = generate_screening_matrix(32)
        else:  # full
            points = get_standard_reference_points() + generate_screening_matrix(32)

        print(f"Loaded {len(points)} configurations for evaluation.")
        print(f"Tournament Setup: {args.num_pairs} pairs ({args.num_pairs * 2} games/variant) | {args.agent_a} vs {args.agent_b} | Max plies: {args.max_plies}")

        cfg = TournamentConfig(
            num_pairs=args.num_pairs,
            max_plies=args.max_plies,
            agent_a=args.agent_a,
            agent_b=args.agent_b,
            sprt_enabled=args.sprt,
        )

        results = []
        for idx, p in enumerate(points):
            print(f"[{idx+1}/{len(points)}] Screening: {p.name} ... ", end="", flush=True)
            res = run_paired_tournament(p.variant, cfg, variant_name=p.name, factors=p.factors)
            results.append(res)
            print(f"Done (Chess: {res.chess_score_rate*100:.1f}%, XQ: {res.xiangqi_score_rate*100:.1f}%, Disparity: {res.balance_disparity*100:.1f}%)")

        # Causal effect analysis
        print("\nComputing causal effect attribution...")
        report = estimate_causal_effects(results)
        summary_md = report.summary_table()
        print(summary_md)

        # Write output JSON
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "analysis": report.to_dict(),
                "results": [r.to_dict() for r in results],
            }, f, indent=2)
        print(f"\n[Saved JSON results to {out_path}]")

        # Write report MD
        if args.report:
            rep_path = Path(args.report)
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rep_path, "w", encoding="utf-8") as f:
                f.write(summary_md)
            print(f"[Saved Markdown report to {rep_path}]")

    elif args.mode == "metrics":
        points = get_standard_reference_points()
        print(f"\nTier 0 Metrics for Standard Presets:")
        print("| Variant | Chess Moves | XQ Moves | Mobility Ratio | Royal Escapes (C/X) | Knight Obstruction | Material Ratio |")
        print("|---|---|---|---|---|---|---|")
        for p in points:
            m = compute_tier0_metrics(p.variant, name=p.name)
            print(
                f"| `{m.variant_name}` | {m.chess_initial_moves} | {m.xiangqi_initial_moves} | {m.mobility_ratio:.2f} | "
                f"{m.chess_royal_escapes} / {m.xiangqi_royal_escapes} | {m.knight_block_obstruction_ratio*100:.1f}% | {m.material_ratio:.2f} |"
            )


if __name__ == "__main__":
    main()
