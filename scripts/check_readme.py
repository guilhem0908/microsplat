"""Keep the README in step with the committed results.

Every measured number of the README lives in a block delimited by HTML comments,
``<!-- begin:NAME -->`` ... ``<!-- end:NAME -->``. This script rebuilds each block from
``results/*.json`` and compares it with the text in the README.

    python scripts/check_readme.py           # exit status 1 if a block is stale
    python scripts/check_readme.py --write   # rewrite the blocks in place
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from microsplat import verify

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
RESULTS = REPO_ROOT / "results"
BLOCK = re.compile(r"(<!-- begin:(\w+) -->\n)(.*?)(<!-- end:\2 -->)", re.DOTALL)
VARIANT_NAMES = {"adaptive": "adaptive (clone / split / prune)", "fixed": "none (fixed set)"}


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _sci(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _group(summary: dict, scene: str, variant: str) -> dict:
    return next(g for g in summary["groups"] if g["scene"] == scene and g["variant"] == variant)


def verdict_block(summary: dict) -> str:
    shapes, knot = (_group(summary, scene, "adaptive") for scene in ("shapes", "knot"))
    shapes_fixed, knot_fixed = (_group(summary, scene, "fixed") for scene in ("shapes", "knot"))
    iterations = summary["configs"]["adaptive"]["iterations"]
    return (
        f"**Starting from random Gaussians, {iterations:,} iterations reach "
        f"{shapes['psnr_mean']:.1f} dB PSNR on held-out views of the ray-traced `shapes` scene "
        f"and {knot['psnr_mean']:.1f} dB on the `knot` scene (mean of {shapes['runs']} seeds); "
        f"without density control the same loop stops at {shapes_fixed['psnr_mean']:.1f} dB and "
        f"{knot_fixed['psnr_mean']:.1f} dB with more Gaussians.**\n"
    )


def setup_block(summary: dict) -> str:
    size = summary["resolution"]
    return (
        f"Setup: {size} x {size} images, {summary['training_views']} training views and "
        f"{summary['held_out_views']} held-out views per scene, "
        f"{summary['configs']['adaptive']['iterations']:,} iterations, {len(summary['seeds'])} "
        f"seeds per row (mean ± sample standard deviation), "
        f"{summary['threads_per_run']} CPU threads per run. The {summary['parallel_runs']} runs "
        f"executed in parallel took {summary['wall_clock_minutes']:.0f} minutes in total.\n"
    )


def fit_table(summary: dict) -> str:
    lines = [
        "| Scene | Density control | Final Gaussians | PSNR (dB) | SSIM "
        "| Median depth error | Depth pixels |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for group in summary["groups"]:
        lines.append(
            f"| `{group['scene']}` | {VARIANT_NAMES[group['variant']]} "
            f"| {group['gaussians_mean']:,.0f} ± {group['gaussians_std']:,.0f} "
            f"| {group['psnr_mean']:.2f} ± {group['psnr_std']:.2f} "
            f"| {group['ssim_mean']:.4f} ± {group['ssim_std']:.4f} "
            f"| {100 * group['depth_rel_err_median_mean']:.2f} % "
            f"| {group['depth_pixels_mean']:,.0f} |"
        )
    return "\n".join(lines) + "\n"


def verification_table(checks: dict) -> str:
    parity, gradients = checks["parity"], checks["gradient_max_relative_error"]
    small, large = checks["monte_carlo_covariance"][0], checks["monte_carlo_covariance"][-1]
    scenes = f"{parity['scenes']} scenes of {parity['gaussians']:,} Gaussians"
    ties = parity["scenes"] - parity["float32_scenes_with_reference_depth_order"]
    rows = [
        (
            "Projection Jacobian vs central differences, max relative error",
            _sci(checks["jacobian_max_relative_error"]),
            f"< {_sci(verify.JACOBIAN_TOLERANCE)}",
        ),
        (
            f"EWA covariance vs Monte-Carlo ({small['samples']:,} samples), Gaussian of "
            f"{100 * small['scale_over_distance']:.1f} % of the viewing distance",
            f"{100 * small['relative_error']:.2f} %",
            f"< {100 * verify.MONTE_CARLO_TOLERANCE:.0f} %",
        ),
        (
            f"Same, Gaussian of {100 * large['scale_over_distance']:.0f} % of the viewing distance",
            f"{100 * large['relative_error']:.1f} %",
            "grows, by design",
        ),
        (
            f"NumPy vs PyTorch float64, max abs. colour difference ({scenes})",
            _sci(parity["float64_max_abs_error"]["color"]),
            f"< {_sci(verify.PARITY_FLOAT64_TOLERANCE)}",
        ),
        (
            "NumPy vs PyTorch float32, the "
            f"{parity['float32_scenes_with_reference_depth_order']} scenes sorted as in float64",
            _sci(parity["float32_max_abs_error_same_depth_order"]["color"]),
            f"< {_sci(verify.PARITY_FLOAT32_TOLERANCE)}",
        ),
        (
            f"NumPy vs PyTorch float32, all {parity['scenes']} scenes ({ties} with a depth tie)",
            _sci(parity["float32_max_abs_error_all_scenes"]["color"]),
            "explained, not bounded",
        ),
        (
            f"Autograd vs central differences, 5 parameter groups, {checks['gradient_seeds']} "
            "scenes, max relative error",
            _sci(max(gradients["without_threshold"].values())),
            f"< {_sci(verify.GRADIENT_TOLERANCE)}",
        ),
    ]
    lines = ["| Check | Measured | Bound in the tests |", "|:--|--:|--:|"]
    lines += [f"| {name} | {value} | {bound} |" for name, value, bound in rows]
    return "\n".join(lines) + "\n"


def timing_table(timing: dict) -> str:
    size = timing["resolution"]
    has_cuda = timing["cuda_name"] is not None
    header = (
        "| Scene state | Gaussians | NumPy reference render | PyTorch CPU render "
        "| PyTorch CPU step |"
    )
    rule = "|:--|--:|--:|--:|--:|"
    if has_cuda:
        header += " PyTorch CUDA step |"
        rule += "--:|"
    lines = [header, rule]
    for label, state in timing["states"].items():
        row = (
            f"| {label} | {state['gaussians']:,} "
            f"| {state['numpy_reference_forward_ms_median']:,.0f} ms "
            f"| {state['cpu']['forward_ms_median']:.0f} ms "
            f"| {state['cpu']['step_ms_median']:.0f} ms |"
        )
        if has_cuda:
            row += f" {state['cuda']['step_ms_median']:.0f} ms |"
        lines.append(row)
    devices = f"{timing['cpu_name']}, {timing['cpu_threads']} threads"
    if has_cuda:
        devices += f"; {timing['cuda_name']}"
    note = (
        f"\nMedian of {timing['timed_steps']} measurements at {size} x {size}. A step is "
        f"render + loss + backward + Adam update. {devices}; PyTorch {timing['torch']}.\n"
    )
    return "\n".join(lines) + "\n" + note


def build_blocks() -> dict[str, str]:
    summary = _load("fit_summary.json")
    return {
        "verdict": verdict_block(summary),
        "setup": setup_block(summary),
        "fit_table": fit_table(summary),
        "verification_table": verification_table(_load("verification.json")),
        "timing_table": timing_table(_load("timing.json")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="rewrite the blocks in place")
    args = parser.parse_args()

    blocks = build_blocks()
    text = README.read_text(encoding="utf-8")
    found = {match.group(2): match.group(3) for match in BLOCK.finditer(text)}
    missing = sorted(set(blocks) - set(found))
    unknown = sorted(set(found) - set(blocks))
    if missing or unknown:
        print(f"README blocks do not match the generators: missing {missing}, unknown {unknown}")
        return 1

    if args.write:
        updated = BLOCK.sub(lambda m: m.group(1) + blocks[m.group(2)] + m.group(4), text)
        with open(README, "w", newline="\n", encoding="utf-8") as handle:
            handle.write(updated)
        print("README blocks rewritten from results/")
        return 0

    stale = [name for name in blocks if found[name] != blocks[name]]
    for name in stale:
        print(f"stale block '{name}':\n--- README\n{found[name]}--- results\n{blocks[name]}")
    if stale:
        return 1
    print(f"README matches results/ ({len(blocks)} generated blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
