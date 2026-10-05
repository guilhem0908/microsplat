"""Regenerate every number and figure quoted in the README.

    python scripts/reproduce.py            # full budget, rewrites results/ and docs/
    python scripts/reproduce.py --quick    # small budget for a smoke run, writes out/quick/

Stages, selectable with ``--only``:

* ``verify``  numerical cross-checks                    -> results/verification.json
* ``fit``     optimisation runs on both scenes          -> results/fit_*.{csv,json},
              results/shapes_adaptive_seed0.ply
* ``gif``     four-panel optimisation and orbit GIF     -> docs/hero.gif
* ``timing``  time per optimisation step and per render -> results/timing.json
* ``plot``    held-out PSNR against iteration           -> docs/psnr_curve.png

All seeds are fixed. Fitted metrics repeat exactly for a given machine and thread count; a
different CPU or thread count changes the floating-point summation order and with it the
last digits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from microsplat import reference, verify, viz
from microsplat.density import DensityConfig
from microsplat.fit import FitConfig, fit
from microsplat.gaussians import GaussianCloud, random_cloud
from microsplat.metrics import photometric_loss, psnr
from microsplat.optim import Adam
from microsplat.ply import load_ply, save_ply
from microsplat.scenes import (
    SCENE_NAMES,
    Scene,
    View,
    held_out_cameras,
    make_scene,
    training_cameras,
)
from microsplat.settings import RasterSettings
from microsplat.torch_renderer import PARAM_NAMES, cloud_to_params, params_to_cloud, render

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("verify", "fit", "gif", "timing", "plot")
VARIANTS = ("adaptive", "fixed")
HERO_RUN = ("shapes", "adaptive", 0)
HERO_PLY = "{}_{}_seed{}.ply".format(*HERO_RUN)
HERO_SNAPSHOTS = "hero_snapshots.npz"  # intermediate clouds for the GIF, not versioned
FIT_THREADS = 4
TIMING_THREADS = 8
TIMING_WARMUP = 5
HERO_FRAME_MS = 110
HERO_HOLD_MS = 1200
ORBIT_FRAME_MS = 90
MONTE_CARLO_SCALES = (0.02, 0.1, 0.3, 0.6)
GRADIENT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Budget:
    """Everything that differs between the full run and the smoke run."""

    resolution: int
    seeds: tuple[int, ...]
    adaptive: FitConfig
    fixed: FitConfig
    hero_frames: int
    orbit_frames: int
    monte_carlo_samples: int
    parity_scenes: int
    parity_gaussians: int
    timing_iterations: int


FULL = Budget(
    resolution=128,
    seeds=(0, 1, 2),
    adaptive=FitConfig(),
    fixed=FitConfig(initial_gaussians=DensityConfig().max_gaussians, density=None),
    hero_frames=34,
    orbit_frames=40,
    monte_carlo_samples=1_000_000,
    parity_scenes=5,
    parity_gaussians=1000,
    timing_iterations=40,
)

QUICK = Budget(
    resolution=48,
    seeds=(0,),
    adaptive=FitConfig(
        iterations=150,
        initial_gaussians=300,
        density=DensityConfig(start=50, stop=120, interval=25, max_gaussians=600),
        eval_every=50,
    ),
    fixed=FitConfig(iterations=150, initial_gaussians=600, density=None, eval_every=50),
    hero_frames=8,
    orbit_frames=8,
    monte_carlo_samples=100_000,
    parity_scenes=2,
    parity_gaussians=200,
    timing_iterations=5,
)


@dataclass(frozen=True)
class FitJob:
    scene: str
    variant: str
    seed: int
    resolution: int
    config: FitConfig
    threads: int
    snapshot_iterations: tuple[int, ...] = ()


def _observe(scene: Scene, resolution: int) -> tuple[list, list]:
    train = [scene.observe(camera) for camera in training_cameras(resolution)]
    held_out = [scene.observe(camera) for camera in held_out_cameras(resolution)]
    return train, held_out


def run_fit(job: FitJob) -> dict:
    """One optimisation run; executed in a worker process."""
    torch.set_num_threads(job.threads)
    scene = make_scene(job.scene)
    train, held_out = _observe(scene, job.resolution)
    snapshots: dict[int, GaussianCloud] = {}

    def keep_snapshot(iteration: int, params: dict[str, torch.Tensor]) -> None:
        if iteration in job.snapshot_iterations:
            snapshots[iteration] = params_to_cloud(params)

    result = fit(scene, train, held_out, replace(job.config, seed=job.seed), keep_snapshot)
    return {"job": job, "result": result, "snapshots": snapshots}


# --------------------------------------------------------------------------- verify


def stage_verify(budget: Budget, results_dir: Path) -> None:
    distance = float(
        np.linalg.norm(verify.check_camera(2, 2).position - np.array(verify.MONTE_CARLO_MEAN))
    )
    monte_carlo = [
        {
            "scale": scale,
            "scale_over_distance": scale / distance,
            "samples": budget.monte_carlo_samples,
            "relative_error": verify.monte_carlo_covariance_error(
                scale, budget.monte_carlo_samples, seed=0
            ),
        }
        for scale in MONTE_CARLO_SCALES
    ]

    size = {"width": 128, "height": 128}

    def parity_runs(dtype: torch.dtype) -> list[dict]:
        return [
            verify.parity_errors(budget.parity_gaussians, seed, dtype, **size)
            for seed in range(budget.parity_scenes)
        ]

    def worst(runs: list[dict]) -> dict[str, float]:
        return {buffer: max(run[buffer] for run in runs) for buffer in verify.PARITY_BUFFERS}

    float32_runs = parity_runs(torch.float32)
    same_order = [run for run in float32_runs if run["same_order"]]
    parity = {
        "scenes": budget.parity_scenes,
        "gaussians": budget.parity_gaussians,
        "resolution": [size["width"], size["height"]],
        "float64_max_abs_error": worst(parity_runs(torch.float64)),
        "float32_scenes_with_reference_depth_order": len(same_order),
        "float32_max_abs_error_same_depth_order": worst(same_order) if same_order else None,
        "float32_max_abs_error_all_scenes": worst(float32_runs),
    }

    gradients = {}
    thresholds = (("without_threshold", 0.0), ("with_threshold", RasterSettings().alpha_min))
    for label, alpha_min in thresholds:
        per_seed = [verify.gradient_errors(alpha_min, seed) for seed in GRADIENT_SEEDS]
        gradients[label] = {name: max(run[name] for run in per_seed) for name in per_seed[0]}

    _write_json(
        results_dir / "verification.json",
        {
            "jacobian_max_relative_error": verify.jacobian_error(seed=0),
            "monte_carlo_covariance": monte_carlo,
            "parity": parity,
            "gradient_seeds": len(GRADIENT_SEEDS),
            "gradient_max_relative_error": gradients,
        },
    )


# ------------------------------------------------------------------------------ fit


def _hero_iterations(iterations: int, frames: int) -> tuple[int, ...]:
    """Roughly geometric schedule: most of the visible change happens early."""
    steps = np.unique(np.round(np.geomspace(1, iterations, frames - 1)).astype(int))
    return (0, *steps.tolist())


def stage_fit(budget: Budget, results_dir: Path, docs_dir: Path, jobs: int) -> None:
    started = time.perf_counter()
    todo = []
    for scene in SCENE_NAMES:
        for variant in VARIANTS:
            config = getattr(budget, variant)
            for seed in budget.seeds:
                hero = (scene, variant, seed) == HERO_RUN
                snapshots = _hero_iterations(config.iterations, budget.hero_frames) if hero else ()
                todo.append(
                    FitJob(scene, variant, seed, budget.resolution, config, FIT_THREADS, snapshots)
                )
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outputs = list(pool.map(run_fit, todo))
    else:
        outputs = [run_fit(job) for job in todo]
    wall_clock = time.perf_counter() - started

    history_rows, run_rows = [], []
    for output in outputs:
        job, result = output["job"], output["result"]
        key = {"scene": job.scene, "variant": job.variant, "seed": job.seed}
        history_rows += [{**key, **record} for record in result.history]
        final = result.history[-1]
        run_rows.append(
            {
                **key,
                "iterations": job.config.iterations,
                "gaussians": final["gaussians"],
                "psnr": final["psnr"],
                "ssim": final["ssim"],
                "depth_rel_err_median": final["depth_rel_err_median"],
                "depth_pixels": final["depth_pixels"],
                "seconds": result.seconds,
            }
        )
    _write_csv(results_dir / "fit_history.csv", history_rows)
    _write_csv(results_dir / "fit_runs.csv", run_rows)

    summary = {
        "resolution": budget.resolution,
        "seeds": list(budget.seeds),
        "training_views": len(training_cameras(budget.resolution)),
        "held_out_views": len(held_out_cameras(budget.resolution)),
        "threads_per_run": FIT_THREADS,
        "parallel_runs": jobs,
        "wall_clock_minutes": wall_clock / 60.0,
        "configs": {variant: asdict(getattr(budget, variant)) for variant in VARIANTS},
        "groups": [],
    }
    for scene in SCENE_NAMES:
        for variant in VARIANTS:
            rows = [r for r in run_rows if (r["scene"], r["variant"]) == (scene, variant)]
            group = {"scene": scene, "variant": variant, "runs": len(rows)}
            for column in ("psnr", "ssim", "depth_rel_err_median", "gaussians", "seconds"):
                values = [row[column] for row in rows]
                group[f"{column}_mean"] = statistics.fmean(values)
                group[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            group["depth_pixels_mean"] = statistics.fmean(row["depth_pixels"] for row in rows)
            summary["groups"].append(group)
    _write_json(results_dir / "fit_summary.json", summary)

    hero = next(o for o in outputs if (o["job"].scene, o["job"].variant, o["job"].seed) == HERO_RUN)
    save_ply(results_dir / HERO_PLY, hero["result"].cloud)
    np.savez_compressed(
        results_dir / HERO_SNAPSHOTS,
        **{
            f"{iteration:06d}/{name}": getattr(cloud, name).astype(np.float32)
            for iteration, cloud in hero["snapshots"].items()
            for name in PARAM_NAMES
        },
    )


# ------------------------------------------------------------------------------ gif


def stage_gif(budget: Budget, results_dir: Path, docs_dir: Path) -> None:
    """Four panels through the optimisation of one held-out view, then a held-out orbit."""
    cache = results_dir / HERO_SNAPSHOTS
    if not cache.exists():
        raise SystemExit(f"{cache.name} is missing: run the fit stage first")
    arrays = np.load(cache)
    iterations = sorted({key.split("/")[0] for key in arrays.files})
    snapshots = [
        (int(key), GaussianCloud(**{name: arrays[f"{key}/{name}"] for name in PARAM_NAMES}))
        for key in iterations
    ]
    scene = make_scene(HERO_RUN[0])
    background = torch.as_tensor(scene.background, dtype=torch.float32)

    def frame(cloud: GaussianCloud, view: View, label: str) -> Image.Image:
        with torch.no_grad():
            image = render(cloud_to_params(cloud), view.camera, background).color.clamp(0, 1)
        score = psnr(image, torch.as_tensor(view.image, dtype=torch.float32)).item()
        caption = f"{label}   |   {len(cloud)} Gaussians   |   PSNR {score:.1f} dB"
        return viz.four_panel_frame(view.image, image.numpy(), cloud, view.camera, caption)

    fixed_view = scene.observe(held_out_cameras(budget.resolution)[0])
    frames = [frame(cloud, fixed_view, f"iteration {iteration}") for iteration, cloud in snapshots]
    durations = [HERO_FRAME_MS] * (len(frames) - 1) + [HERO_HOLD_MS]

    final = load_ply(results_dir / HERO_PLY)
    for camera in held_out_cameras(budget.resolution, budget.orbit_frames):
        frames.append(frame(final, scene.observe(camera), "held-out orbit"))
        durations.append(ORBIT_FRAME_MS)
    viz.save_gif(frames, durations, docs_dir / "hero.gif")
    print("wrote hero.gif")


# --------------------------------------------------------------------------- timing


def _cpu_name() -> str:
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _median_ms(samples: list[float]) -> float:
    return 1000.0 * statistics.median(samples)


def _time_steps(cloud: GaussianCloud, scene: Scene, views: list, device: str, count: int) -> dict:
    """Median time of ``render + loss + backward + Adam update`` and of a render alone."""
    params = cloud_to_params(cloud, device=device, requires_grad=True)
    background = torch.as_tensor(scene.background, dtype=torch.float32, device=device)
    targets = [torch.as_tensor(v.image, dtype=torch.float32, device=device) for v in views]
    config = FitConfig()
    rates = {
        "means": config.lr_means * scene.extent * config.lr_means_final_ratio,
        "log_scales": config.lr_log_scales,
        "quats": config.lr_quats,
        "opacity_logits": config.lr_opacity_logits,
        "sh_dc": config.lr_sh_dc,
    }
    adam = Adam(params)

    def synchronise() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    step_times, forward_times = [], []
    for i in range(TIMING_WARMUP + count):
        view = views[i % len(views)]
        synchronise()
        tick = time.perf_counter()
        loss = photometric_loss(
            render(params, view.camera, background).color, targets[i % len(views)]
        )
        loss.backward()
        adam.step(params, rates)
        synchronise()
        step_times.append(time.perf_counter() - tick)

        tick = time.perf_counter()
        with torch.no_grad():
            render(params, view.camera, background)
        synchronise()
        forward_times.append(time.perf_counter() - tick)
    return {
        "step_ms_median": _median_ms(step_times[TIMING_WARMUP:]),
        "forward_ms_median": _median_ms(forward_times[TIMING_WARMUP:]),
    }


def stage_timing(budget: Budget, results_dir: Path) -> None:
    torch.set_num_threads(min(TIMING_THREADS, os.cpu_count() or 1))
    scene = make_scene(HERO_RUN[0])
    views = [scene.observe(camera) for camera in training_cameras(budget.resolution)[:6]]
    clouds = {
        "fitted": load_ply(results_dir / HERO_PLY),
        "initial": random_cloud(
            budget.adaptive.initial_gaussians,
            scene.box_min,
            scene.box_max,
            np.random.default_rng(0),
        ),
    }

    states = {}
    for label, cloud in clouds.items():
        entry = {
            "gaussians": len(cloud),
            "cpu": _time_steps(cloud, scene, views, "cpu", budget.timing_iterations),
        }
        numpy_times = []
        for view in views[:3]:
            tick = time.perf_counter()
            reference.render(cloud, view.camera, scene.background)
            numpy_times.append(time.perf_counter() - tick)
        entry["numpy_reference_forward_ms_median"] = _median_ms(numpy_times)
        if torch.cuda.is_available():
            entry["cuda"] = _time_steps(cloud, scene, views, "cuda", budget.timing_iterations)
        states[label] = entry

    _write_json(
        results_dir / "timing.json",
        {
            "resolution": budget.resolution,
            "timed_steps": budget.timing_iterations,
            "cpu_name": _cpu_name(),
            "cpu_threads": torch.get_num_threads(),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.system(),
            "states": states,
        },
    )


# ----------------------------------------------------------------------------- plot


def stage_plot(results_dir: Path, docs_dir: Path) -> None:
    with open(results_dir / "fit_history.csv", newline="") as handle:
        rows = [
            {**row, "seed": int(row["seed"]), "iteration": int(row["iteration"]),
             "psnr": float(row["psnr"])}
            for row in csv.DictReader(handle)
        ]  # fmt: skip
    viz.plot_psnr_curves(rows, SCENE_NAMES, docs_dir / "psnr_curve.png")


# ------------------------------------------------------------------------------ i/o


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path.name}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="small budget, writes out/quick/")
    parser.add_argument("--only", choices=STAGES, action="append")
    parser.add_argument("--jobs", type=int, default=1, help="optimisation runs in parallel")
    parser.add_argument("--out", type=Path, help="output root (default: repository, or out/quick)")
    args = parser.parse_args()

    budget = QUICK if args.quick else FULL
    root = args.out or (REPO_ROOT / "out" / "quick" if args.quick else REPO_ROOT)
    results_dir, docs_dir = root / "results", root / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(FIT_THREADS)

    stages = args.only or STAGES
    if "verify" in stages:
        stage_verify(budget, results_dir)
    if "fit" in stages:
        stage_fit(budget, results_dir, docs_dir, args.jobs)
    if "gif" in stages:
        stage_gif(budget, results_dir, docs_dir)
    if "timing" in stages:
        stage_timing(budget, results_dir)
    if "plot" in stages:
        stage_plot(results_dir, docs_dir)


if __name__ == "__main__":
    main()
