"""Keep the README in step with the committed results.

Every measured number of the README lives in a block delimited by HTML comments,
``<!-- begin:NAME -->`` ... ``<!-- end:NAME -->``. This script rebuilds each block from
``results/*.json`` and compares it with the text in the README.

    python scripts/check_readme.py           # exit status 1 if a block is stale
    python scripts/check_readme.py --write   # rewrite the blocks in place
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from microsplat import verify
from microsplat.scenes import KNOT_GAUSSIANS

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
RESULTS = REPO_ROOT / "results"
BLOCK = re.compile(r"(<!-- begin:(\w+) -->\n)(.*?)(<!-- end:\2 -->)", re.DOTALL)
SCENES = ("shapes", "knot")
VARIANT_NAMES = {"adaptive": "adaptive (clone / split / prune)", "fixed": "none (fixed set)"}


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _sci(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _group(summary: dict, scene: str, variant: str) -> dict:
    return next(g for g in summary["groups"] if g["scene"] == scene and g["variant"] == variant)


def _load_history() -> dict[tuple[str, str, int], dict[str, float]]:
    """Mean over the seeds of the held-out PSNR and of the Gaussian count, per curve point."""
    with (RESULTS / "fit_history.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    points: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        points.setdefault((row["scene"], row["variant"], int(row["iteration"])), []).append(row)
    return {
        key: {
            "psnr": sum(float(r["psnr"]) for r in group) / len(group),
            "gaussians": sum(int(r["gaussians"]) for r in group) / len(group),
        }
        for key, group in points.items()
    }


def _require(condition: bool, message: str) -> None:
    """The prose around the tables states a ranking; fail loudly if new results reverse it."""
    if not condition:
        raise SystemExit(f"results no longer support the README text: {message}")


def _require_phase_lock(config: dict) -> None:
    """The text says that curve points fall right after the densification rounds."""
    density = config["density"]
    _require(
        config["eval_every"] == density["interval"]
        and density["start"] % config["eval_every"] == 0,
        "evaluation points no longer coincide with the densification rounds",
    )


def verdict_block(summary: dict) -> str:
    adaptive = {scene: _group(summary, scene, "adaptive") for scene in SCENES}
    fixed = {scene: _group(summary, scene, "fixed") for scene in SCENES}
    delta = {scene: adaptive[scene]["psnr_mean"] - fixed[scene]["psnr_mean"] for scene in SCENES}
    _require(delta["shapes"] > 0 > delta["knot"], "adaptive wins on shapes and loses on knot")
    configs = summary["configs"]
    return (
        f"**Starting from random Gaussians, {configs['adaptive']['iterations']:,} iterations "
        f"reach {adaptive['shapes']['psnr_mean']:.1f} dB PSNR on held-out views of the "
        f"ray-traced `shapes` scene and {adaptive['knot']['psnr_mean']:.1f} dB on the `knot` "
        f"scene (mean of {adaptive['shapes']['runs']} seeds; adaptive density control, "
        f"{configs['adaptive']['initial_gaussians']:,} initial Gaussians). A baseline without "
        f"density control, which has {configs['fixed']['initial_gaussians']:,} Gaussians from "
        f"the start, scores {delta['shapes']:.1f} dB lower on `shapes` and "
        f"{-delta['knot']:.1f} dB higher on `knot`. The two arms also differ in size, so the "
        "`knot` result is not an isolated effect of density control (see below).**\n"
    )


def setup_block(summary: dict) -> str:
    size = summary["resolution"]
    runs = sum(group["runs"] for group in summary["groups"])
    return (
        f"Setup: {size} x {size} images, {summary['training_views']} training views and "
        f"{summary['held_out_views']} held-out views per scene, "
        f"{summary['configs']['adaptive']['iterations']:,} iterations, {len(summary['seeds'])} "
        f"seeds per row (mean ± sample standard deviation), "
        f"{summary['threads_per_run']} CPU threads per run. Adaptive runs start from "
        f"{summary['configs']['adaptive']['initial_gaussians']:,} random Gaussians, fixed runs "
        f"from {summary['configs']['fixed']['initial_gaussians']:,}. The {runs} runs, "
        f"{summary['parallel_runs']} at a time, took {summary['wall_clock_minutes']:.0f} "
        "minutes in total. Depth error: median of |z - z_true| / z_true over the pixels "
        f"counted in the last column, pooled over the held-out views.\n"
    )


def reading_block(summary: dict, history: dict) -> str:
    adaptive = {scene: _group(summary, scene, "adaptive") for scene in SCENES}
    fixed = {scene: _group(summary, scene, "fixed") for scene in SCENES}
    delta = {scene: adaptive[scene]["psnr_mean"] - fixed[scene]["psnr_mean"] for scene in SCENES}
    _require(delta["shapes"] > 0 > delta["knot"], "density control helps on shapes, not on knot")
    depth = {key: 100 * group["depth_rel_err_median_mean"] for key, group in (
        ("adaptive", adaptive["shapes"]), ("fixed", fixed["shapes"]))}  # fmt: skip
    _require(depth["fixed"] < depth["adaptive"], "fixed set has the lower depth error on shapes")
    configs = summary["configs"]
    iterations = configs["adaptive"]["iterations"]
    density = configs["adaptive"]["density"]
    _require_phase_lock(configs["adaptive"])

    # Facts about the curves that the text quotes; each is checked before it is stated.
    shapes_points = [
        it for scene, variant, it in history if (scene, variant) == ("shapes", "fixed")
    ]
    _require(
        all(
            history["shapes", "adaptive", it]["gaussians"]
            < history["shapes", "fixed", it]["gaussians"]
            for it in shapes_points
        ),
        "the adaptive runs hold fewer Gaussians than the fixed set at every point of shapes",
    )
    first, last, every = density["start"], density["stop"], configs["adaptive"]["eval_every"]
    before, after = first - every, last + every
    knot = {
        (variant, it): history["knot", variant, it]
        for variant in ("adaptive", "fixed")
        for it in (before, last, after)
    }
    _require(
        knot["adaptive", before]["psnr"] < knot["fixed", before]["psnr"],
        "the fixed set is already ahead on knot before the first densification round",
    )
    _require(
        knot["adaptive", last]["gaussians"] == knot["adaptive", after]["gaussians"]
        and knot["adaptive", after]["psnr"] > knot["adaptive", last]["psnr"],
        "the adaptive knot PSNR rises between the last round and the next point at constant size",
    )
    return (
        "**How to read the table.**\n"
        f"- `shapes`: density control gains {delta['shapes']:.2f} dB with "
        f"{adaptive['shapes']['gaussians_mean']:,.0f} Gaussians instead of "
        f"{fixed['shapes']['gaussians_mean']:,.0f}, starting from "
        f"{configs['adaptive']['initial_gaussians']:,} instead of "
        f"{configs['fixed']['initial_gaussians']:,}. The adaptive runs hold fewer Gaussians than "
        "the fixed set at every point of the curve, so the gain is not bought with more of them. "
        "The depth metric ranks the two variants the other way round, narrowly (median error "
        f"{depth['fixed']:.2f} % for the fixed set against {depth['adaptive']:.2f} %, seed "
        f"standard deviation {100 * adaptive['shapes']['depth_rel_err_median_std']:.2f} %): "
        "image quality and depth accuracy are different measurements.\n"
        f"- `knot` is exactly representable, by the {KNOT_GAUSSIANS} Gaussians that generated "
        f"it, and there the fixed set ends {-delta['knot']:.1f} dB higher. The cause is not "
        "isolated, and three things stand in the way of reading this as an effect of density "
        "control. The arms differ in capacity: the adaptive runs start from "
        f"{configs['adaptive']['initial_gaussians']:,} Gaussians and end with "
        f"{adaptive['knot']['gaussians_mean']:,.0f}, the fixed runs have "
        f"{fixed['knot']['gaussians_mean']:,.0f} from iteration 0. The gap exists before any "
        f"densification: at iteration {before}, ahead of the first round at iteration {first}, "
        f"the adaptive runs are at {knot['adaptive', before]['psnr']:.1f} dB against "
        f"{knot['fixed', before]['psnr']:.1f} dB. And the curve points are locked to the "
        f"densification rounds (note under the plot): between iterations {last:,} and "
        f"{after:,}, with {knot['adaptive', after]['gaussians']:,.0f} Gaussians at both, the "
        f"adaptive PSNR goes from {knot['adaptive', last]['psnr']:.1f} to "
        f"{knot['adaptive', after]['psnr']:.1f} dB, so the shape of the adaptive curve, far below "
        f"until iteration {last:,} and then a steep climb, is partly an artefact of when it is "
        f"sampled. The final gap is measured {iterations - last:,} iterations after the last "
        "round and does not suffer from that, but no experiment here separates density control "
        "from initial and final size. "
        f"The comparison holds for this schedule ({iterations:,} iterations, one hand-set "
        "gradient threshold), not in general.\n"
    )


def curve_note_block(summary: dict) -> str:
    config = summary["configs"]["adaptive"]
    density = config["density"]
    _require_phase_lock(config)
    return (
        f"Curve points are taken every {config['eval_every']} iterations and the adaptive runs "
        f"densify every {density['interval']} iterations from iteration {density['start']:,} to "
        f"{density['stop']:,}: each adaptive point in that range is measured right after a "
        "clone / split / prune round. The fixed runs have no rounds.\n"
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
            f"Autograd vs central differences, {len(gradients['with_threshold'])} parameter "
            f"groups, {checks['gradient_seeds']} scenes, with and without the alpha threshold, "
            "max relative error",
            _sci(max(max(errors.values()) for errors in gradients.values())),
            f"< {_sci(verify.GRADIENT_TOLERANCE)}",
        ),
    ]
    lines = ["| Check | Measured | Bound in the tests |", "|:--|--:|--:|"]
    lines += [f"| {name} | {value} | {bound} |" for name, value, bound in rows]
    return "\n".join(lines) + "\n"


def timing_table(timing: dict, summary: dict) -> str:
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
        f"render + loss + backward + Adam update. {devices}; PyTorch {timing['torch']}."
    )
    fitted = timing["states"]["fitted"]
    ratio = fitted["numpy_reference_forward_ms_median"] / fitted["cpu"]["forward_ms_median"]
    note += (
        f" The NumPy reference is the specification, not a fast renderer: it takes {ratio:.0f} "
        "times longer than the PyTorch render on the fitted scene."
    )
    if has_cuda:
        note += (
            " The GPU is a laptop part that was not reserved for the measurement, so read the "
            "CUDA column as indicative."
        )
    # The study of the section above is much slower per iteration than these steps suggest.
    seconds = [group["seconds_mean"] for group in summary["groups"]]
    iterations = summary["configs"]["adaptive"]["iterations"]
    per_iteration = [1000 * value / iterations for value in seconds]
    note += (
        " The steps were timed alone, by a separate stage that runs after the fits. The fits of "
        f"the study ran {summary['parallel_runs']} at a time with {summary['threads_per_run']} "
        f"threads each on the same CPU and took {min(seconds):.0f} to {max(seconds):.0f} s per "
        f"run ({min(per_iteration):.0f} to {max(per_iteration):.0f} ms per iteration, "
        "densification and the changing Gaussian count included), so the two sets of times "
        "should not be compared."
    )
    return "\n".join(lines) + "\n" + note + "\n"


def build_blocks() -> dict[str, str]:
    summary = _load("fit_summary.json")
    return {
        "verdict": verdict_block(summary),
        "setup": setup_block(summary),
        "fit_table": fit_table(summary),
        "curve_note": curve_note_block(summary),
        "reading": reading_block(summary, _load_history()),
        "verification_table": verification_table(_load("verification.json")),
        "timing_table": timing_table(_load("timing.json"), summary),
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
