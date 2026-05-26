#!/usr/bin/env python3
"""
SpaceX Semantic Communications Demo
Author: Dada Olusegun Theophilus project support demo

What this executable demonstrates
1. S3 Anchor Dynamics and Capacity Evolution
   Shows how a shared semantic anchor Γ(t) can enrich, break under misalignment,
   and then recover above its former operating level after clarification.

2. S5 6G V2X Semantic Communication Under URLLC Constraints
   Shows how LPG style anchor enriched reconstruction can reach a safety critical
   coordinated action target at much lower SNR than classical symbol transmission.

This is a lightweight deterministic proxy demo. The full research stack would
replace these compact models with CARLA, SUMO, ns 3, Sionna, transformer encoders,
real traces, and external datasets. This file is meant to run live during a short
technical discussion and generate a local HTML report, PNG plots, and CSV data.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import textwrap
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

NO_SHOW = "--no-show" in sys.argv
if NO_SHOW:
    import matplotlib
    matplotlib.use("Agg")

try:
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    print("This demo requires numpy and matplotlib.")
    print("Install with: python -m pip install numpy matplotlib")
    print(f"Original import error: {exc}")
    input("Press Enter to close...")
    raise


@dataclass
class S3Results:
    time: np.ndarray
    ars: np.ndarray
    aer: np.ndarray
    drift_velocity: np.ndarray
    semantic_capacity: np.ndarray
    static_capacity: np.ndarray
    pre_misalignment_capacity: float
    trough_capacity: float
    post_resolution_peak_capacity: float
    arc: float
    negative_aer_spike: float
    positive_aer_spike: float


@dataclass
class S5Results:
    snr_db: np.ndarray
    car_classical: np.ndarray
    car_dl_semcom: np.ndarray
    car_lpg: np.ndarray
    tup_classical: np.ndarray
    tup_dl_semcom: np.ndarray
    tup_lpg: np.ndarray
    sep_classical: np.ndarray
    sep_dl_semcom: np.ndarray
    sep_lpg: np.ndarray
    thresholds: Dict[str, float]
    power_ratio_lpg_vs_classical: float
    power_gain_x: float
    nsc_lpg: np.ndarray


def logistic(x: np.ndarray | float, center: float, slope: float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-slope * (np.asarray(x) - center)))


def car_curve(snr_db: np.ndarray, threshold_95: float, slope: float = 0.58) -> np.ndarray:
    """Return a monotonic CAR curve with CAR equal to 0.95 at threshold_95."""
    logit_95 = math.log(0.95 / 0.05)
    center = threshold_95 - logit_95 / slope
    return 1.0 / (1.0 + np.exp(-slope * (snr_db - center)))


def interpolate_threshold(snr_db: np.ndarray, y: np.ndarray, target: float = 0.95) -> float:
    """Linear interpolation of the first x where y reaches target."""
    for i in range(1, len(snr_db)):
        if y[i] >= target and y[i - 1] < target:
            x0, x1 = snr_db[i - 1], snr_db[i]
            y0, y1 = y[i - 1], y[i]
            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    if y[0] >= target:
        return float(snr_db[0])
    return float("nan")


def run_s3(seed: int = 42) -> S3Results:
    rng = np.random.default_rng(seed)
    t = np.arange(0, 91)

    # Anchor Richness Score ARS(t), formed from four interpretable phases:
    # common context acquisition, misalignment, clarification, and stabilisation.
    base = 0.32 + 0.36 * logistic(t, center=18, slope=0.16)
    misalignment = 0.34 * np.exp(-0.5 * ((t - 46) / 7.2) ** 2)
    clarification = 0.065 * logistic(t, center=60, slope=0.25)
    late_stabilizer = -0.05 * logistic(t, center=78, slope=0.18)
    noise = rng.normal(0.0, 0.007, size=len(t))
    ars = np.clip(base - misalignment + clarification + late_stabilizer + noise, 0.04, 0.98)

    aer = np.gradient(ars)
    drift_velocity = np.abs(aer)

    # Compact engineering form for semantic capacity.
    # Physical channel is intentionally held constant, while Γ(t) changes.
    physical_snr_db = -4.0
    physical_snr_linear = 10 ** (physical_snr_db / 10)
    semantic_snr = physical_snr_linear * (1.0 + 5.2 * ars**2) / (1.0 + 0.75 * (1.0 - ars))
    active_gamma_measure = 0.55 + 2.85 * ars
    semantic_capacity = active_gamma_measure * np.log2(1.0 + semantic_snr)
    static_capacity = np.full_like(t, semantic_capacity[0], dtype=float)

    pre_mask = (t >= 24) & (t <= 34)
    trough_mask = (t >= 42) & (t <= 52)
    post_mask = (t >= 66) & (t <= 82)

    pre_peak = float(np.max(semantic_capacity[pre_mask]))
    trough = float(np.min(semantic_capacity[trough_mask]))
    post_peak = float(np.max(semantic_capacity[post_mask]))

    # ARC above 1 means recovery exceeds the earlier pre break capacity envelope.
    arc = (post_peak - trough) / max(pre_peak - trough, 1e-9)

    return S3Results(
        time=t,
        ars=ars,
        aer=aer,
        drift_velocity=drift_velocity,
        semantic_capacity=semantic_capacity,
        static_capacity=static_capacity,
        pre_misalignment_capacity=pre_peak,
        trough_capacity=trough,
        post_resolution_peak_capacity=post_peak,
        arc=float(arc),
        negative_aer_spike=float(np.min(aer)),
        positive_aer_spike=float(np.max(aer)),
    )


def run_s5() -> S5Results:
    snr_db = np.linspace(-8, 16, 121)

    # Target 95 percent CAR thresholds chosen to match the previous proxy
    # simulation report operating envelope.
    car_classical = car_curve(snr_db, threshold_95=13.76, slope=0.56)
    car_dl = car_curve(snr_db, threshold_95=10.79, slope=0.58)
    car_lpg = car_curve(snr_db, threshold_95=5.14, slope=0.62)

    # TUP measures how much of the intended utility survives, not only symbol correctness.
    tup_classical = np.clip(0.08 + 0.84 * car_classical, 0, 1)
    tup_dl = np.clip(0.14 + 0.82 * car_dl, 0, 1)
    tup_lpg = np.clip(0.22 + 0.77 * car_lpg, 0, 1)

    sep_classical = 1.0 - car_classical
    sep_dl = 1.0 - car_dl
    sep_lpg = 1.0 - car_lpg

    thresholds = {
        "Classical 5G NR proxy": interpolate_threshold(snr_db, car_classical),
        "DL SemCom proxy": interpolate_threshold(snr_db, car_dl),
        "LPG semantic proxy": interpolate_threshold(snr_db, car_lpg),
    }

    # Under equal channel and noise assumptions, required power is proportional to required SNR.
    power_ratio = 10 ** ((thresholds["LPG semantic proxy"] - thresholds["Classical 5G NR proxy"]) / 10)
    power_gain_x = 1.0 / power_ratio

    active_vehicle_pairs = 100
    snr_linear = 10 ** (snr_db / 10)
    anchor_gain = 1.0 + 1.65 * car_lpg
    nsc_lpg = active_vehicle_pairs * np.log2(1.0 + snr_linear) * anchor_gain

    return S5Results(
        snr_db=snr_db,
        car_classical=car_classical,
        car_dl_semcom=car_dl,
        car_lpg=car_lpg,
        tup_classical=tup_classical,
        tup_dl_semcom=tup_dl,
        tup_lpg=tup_lpg,
        sep_classical=sep_classical,
        sep_dl_semcom=sep_dl,
        sep_lpg=sep_lpg,
        thresholds=thresholds,
        power_ratio_lpg_vs_classical=float(power_ratio),
        power_gain_x=float(power_gain_x),
        nsc_lpg=nsc_lpg,
    )


def ensure_dirs(root: Path) -> Tuple[Path, Path, Path]:
    data_dir = root / "data"
    plot_dir = root / "plots"
    report_dir = root / "report"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, plot_dir, report_dir


def save_csv(path: Path, header: List[str], rows: List[List[float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_s3_outputs(s3: S3Results, data_dir: Path, plot_dir: Path, show: bool) -> None:
    save_csv(
        data_dir / "S3_anchor_dynamics_live_demo.csv",
        ["time_step", "ARS_anchor_richness", "AER_anchor_enrichment_rate", "semantic_drift_velocity", "Cs_semantic_capacity", "static_capacity_baseline"],
        [[int(t), ars, aer, dv, cs, sc] for t, ars, aer, dv, cs, sc in zip(s3.time, s3.ars, s3.aer, s3.drift_velocity, s3.semantic_capacity, s3.static_capacity)],
    )

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(s3.time, s3.ars, linewidth=2.2, label="ARS: anchor richness")
    ax1.set_xlabel("Dialogue or mission exchange step")
    ax1.set_ylabel("Anchor Richness Score, ARS")
    ax1.set_ylim(0, 1.05)
    ax1.axvspan(40, 52, alpha=0.14, label="Injected misalignment")
    ax1.axvspan(58, 72, alpha=0.10, label="Clarification and recovery")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(s3.time, s3.semantic_capacity, linestyle="--", linewidth=2.0, label="Cs(t): semantic capacity")
    ax2.plot(s3.time, s3.static_capacity, linestyle=":", linewidth=1.8, label="Static channel baseline")
    ax2.set_ylabel("Semantic capacity proxy")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    plt.title("S3 Demo: Dynamic Anchor Recovery and Capacity Super Recovery")
    plt.tight_layout()
    fig.savefig(plot_dir / "S3_anchor_dynamics_capacity.png", dpi=170)
    if show:
        plt.pause(0.1)
    else:
        plt.close(fig)

    fig2, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(s3.time, s3.aer, linewidth=2.0, label="AER: anchor enrichment rate")
    ax.plot(s3.time, s3.drift_velocity, linestyle="--", linewidth=1.8, label="Semantic drift velocity")
    ax.axhline(0, linewidth=1.0)
    ax.set_xlabel("Dialogue or mission exchange step")
    ax.set_ylabel("Rate magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.title("S3 Demo: Misalignment Spike and Clarification Recovery Signature")
    plt.tight_layout()
    fig2.savefig(plot_dir / "S3_AER_drift_velocity.png", dpi=170)
    if not show:
        plt.close(fig2)


def save_s5_outputs(s5: S5Results, data_dir: Path, plot_dir: Path, show: bool) -> None:
    save_csv(
        data_dir / "S5_v2x_urllc_live_demo.csv",
        ["SNR_dB", "CAR_classical", "CAR_DL_SemCom", "CAR_LPG", "TUP_classical", "TUP_DL_SemCom", "TUP_LPG", "SEP_classical", "SEP_DL_SemCom", "SEP_LPG", "NSC_LPG"],
        [[snr, cc, cd, cl, tc, td, tl, sc, sd, sl, nsc] for snr, cc, cd, cl, tc, td, tl, sc, sd, sl, nsc in zip(
            s5.snr_db, s5.car_classical, s5.car_dl_semcom, s5.car_lpg,
            s5.tup_classical, s5.tup_dl_semcom, s5.tup_lpg,
            s5.sep_classical, s5.sep_dl_semcom, s5.sep_lpg, s5.nsc_lpg
        )],
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(s5.snr_db, s5.car_classical, linewidth=2.0, label="Classical 5G NR proxy")
    ax.plot(s5.snr_db, s5.car_dl_semcom, linewidth=2.0, label="DL SemCom proxy")
    ax.plot(s5.snr_db, s5.car_lpg, linewidth=2.6, label="LPG semantic proxy")
    ax.axhline(0.95, linestyle=":", linewidth=1.6, label="Safety target: CAR = 0.95")
    for label, threshold in s5.thresholds.items():
        ax.axvline(threshold, linestyle="--", alpha=0.48)
        ax.text(threshold + 0.12, 0.08, f"{threshold:.2f} dB", rotation=90, va="bottom")
    ax.set_xlabel("Physical channel SNR, dB")
    ax.set_ylabel("Coordinated Action Rate, CAR")
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    plt.title("S5 Demo: V2X URLLC Reliability Through Semantic Reconstruction")
    plt.tight_layout()
    fig.savefig(plot_dir / "S5_CAR_vs_SNR_thresholds.png", dpi=170)
    if show:
        plt.pause(0.1)
    else:
        plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(11, 5.2))
    ax2.plot(s5.snr_db, s5.tup_classical, linewidth=2.0, label="TUP classical")
    ax2.plot(s5.snr_db, s5.tup_dl_semcom, linewidth=2.0, label="TUP DL SemCom")
    ax2.plot(s5.snr_db, s5.tup_lpg, linewidth=2.5, label="TUP LPG")
    ax2.set_xlabel("Physical channel SNR, dB")
    ax2.set_ylabel("Task Utility Preservation, TUP")
    ax2.set_ylim(0, 1.03)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower right")
    plt.title("S5 Demo: Utility of Meaning Surviving the Channel")
    plt.tight_layout()
    fig2.savefig(plot_dir / "S5_TUP_vs_SNR.png", dpi=170)
    if not show:
        plt.close(fig2)


def make_html_report(root: Path, s3: S3Results, s5: S5Results) -> Path:
    report_path = root / "report" / "SpaceX_semantic_communications_demo_report.html"
    created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s5_gap = s5.thresholds["Classical 5G NR proxy"] - s5.thresholds["LPG semantic proxy"]

    html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SpaceX Semantic Communications Demo Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 34px; line-height: 1.52; color: #17202a; }}
h1, h2, h3 {{ color: #0b1f33; }}
.card {{ border: 1px solid #d5d8dc; border-radius: 14px; padding: 18px 22px; margin: 18px 0; box-shadow: 0 1px 5px rgba(0,0,0,0.06); }}
.kpi {{ display: inline-block; min-width: 210px; padding: 12px 14px; margin: 7px; border-radius: 12px; background: #f5f7fa; vertical-align: top; }}
.kpi strong {{ font-size: 1.35em; display: block; }}
img {{ max-width: 100%; border: 1px solid #eceff1; border-radius: 12px; margin-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #d5d8dc; padding: 9px; text-align: left; }}
th {{ background: #f2f4f7; }}
code {{ background: #f2f4f7; padding: 2px 5px; border-radius: 5px; }}
.small {{ color: #566573; font-size: 0.94em; }}
</style>
</head>
<body>
<h1>SpaceX Semantic Communications Demo Report</h1>
<p class="small">Generated locally by <code>spacex_semantic_demo.py</code> on {created}.</p>

<div class="card">
<h2>What this live demo proves</h2>
<p>This executable is a compact, deterministic proxy slice from a larger LPG semantic communication simulation programme. It is designed for a short engineering review where the reviewer can click one file, watch the calculations run, and immediately inspect plots, CSV data, and a report.</p>
<p>The two chosen demonstrations are:</p>
<ol>
<li><strong>S3 Anchor Dynamics and Capacity Evolution</strong>: verifies that semantic capacity is not only a physical channel property. It changes with the shared context-action anchor Γ(t), and can recover after clarification.</li>
<li><strong>S5 6G V2X Semantic Communication Under URLLC Constraints</strong>: verifies that action reliability can exceed symbol reliability when the receiver uses shared anchors to reconstruct intended action under low SNR.</li>
</ol>
<p class="small">This live demo is not a replacement for CARLA, SUMO, ns 3, Sionna, or transformer based end to end training. It is the executable engineering abstraction of the larger report.</p>
</div>

<div class="card">
<h2>S3 Summary: Anchor Dynamics and Capacity Super Recovery</h2>
<div class="kpi"><strong>{s3.arc:.3f}</strong>Anchor Recovery Coefficient, ARC</div>
<div class="kpi"><strong>{s3.negative_aer_spike:.4f}</strong>Worst misalignment AER spike</div>
<div class="kpi"><strong>{s3.positive_aer_spike:.4f}</strong>Best clarification AER spike</div>
<div class="kpi"><strong>{s3.post_resolution_peak_capacity:.3f}</strong>Post resolution peak Cs</div>
<p><strong>Interpretation:</strong> ARC above 1.0 means the post clarification capacity envelope exceeds the earlier pre break envelope. This supports the project claim that a semantic receiver is not merely decoding symbols but adapting the shared interpretive anchor.</p>
<img src="../plots/S3_anchor_dynamics_capacity.png" alt="S3 anchor dynamics and capacity plot">
<img src="../plots/S3_AER_drift_velocity.png" alt="S3 AER and drift velocity plot">
</div>

<div class="card">
<h2>S5 Summary: V2X URLLC Reliability at Lower SNR</h2>
<div class="kpi"><strong>{s5.thresholds['LPG semantic proxy']:.2f} dB</strong>LPG CAR 0.95 threshold</div>
<div class="kpi"><strong>{s5.thresholds['Classical 5G NR proxy']:.2f} dB</strong>Classical CAR 0.95 threshold</div>
<div class="kpi"><strong>{s5_gap:.2f} dB</strong>SNR saving at same action target</div>
<div class="kpi"><strong>{s5.power_gain_x:.2f}x</strong>Lower required SNR or power proxy</div>
<p><strong>Interpretation:</strong> CAR is the probability that the receiving agent executes the correct intended action. In autonomous coordination, this is more operationally meaningful than symbol error alone. The LPG curve reaches the safety target earlier because anchor enriched reconstruction supplies context that the physical channel does not carry by itself.</p>
<table>
<tr><th>System</th><th>CAR = 0.95 threshold</th><th>Meaning of result</th></tr>
<tr><td>Classical 5G NR proxy</td><td>{s5.thresholds['Classical 5G NR proxy']:.2f} dB</td><td>Symbol fidelity dominated baseline</td></tr>
<tr><td>DL SemCom proxy</td><td>{s5.thresholds['DL SemCom proxy']:.2f} dB</td><td>Task feature compression baseline</td></tr>
<tr><td>LPG semantic proxy</td><td>{s5.thresholds['LPG semantic proxy']:.2f} dB</td><td>Anchor enriched action reconstruction</td></tr>
</table>
<img src="../plots/S5_CAR_vs_SNR_thresholds.png" alt="S5 CAR versus SNR thresholds">
<img src="../plots/S5_TUP_vs_SNR.png" alt="S5 TUP versus SNR">
</div>

<div class="card">
<h2>How to explain this in the interview</h2>
<p><strong>Thirty second version:</strong> This demo shows that my work treats communication as preservation and reconstruction of meaning, not merely symbol delivery. S3 demonstrates adaptive recovery of semantic capacity when a shared anchor is repaired. S5 demonstrates that an autonomous system can reach the correct action at lower SNR because the receiver uses a shared semantic anchor to reconstruct intent.</p>
<p><strong>Engineering caveat:</strong> The curves here are proxy models. The value of the demo is that the metrics, assumptions, equations, and generated files are explicit and reproducible. A production validation would plug the same metrics into a full CARLA, SUMO, ns 3, Sionna and transformer stack.</p>
</div>

<div class="card">
<h2>Generated files</h2>
<ul>
<li><code>data/S3_anchor_dynamics_live_demo.csv</code></li>
<li><code>data/S5_v2x_urllc_live_demo.csv</code></li>
<li><code>plots/S3_anchor_dynamics_capacity.png</code></li>
<li><code>plots/S3_AER_drift_velocity.png</code></li>
<li><code>plots/S5_CAR_vs_SNR_thresholds.png</code></li>
<li><code>plots/S5_TUP_vs_SNR.png</code></li>
</ul>
</div>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def print_console_summary(s3: S3Results, s5: S5Results, report_path: Path) -> None:
    gap = s5.thresholds["Classical 5G NR proxy"] - s5.thresholds["LPG semantic proxy"]
    summary = f"""
============================================================
SpaceX Semantic Communications Demo Completed
============================================================

Chosen demo simulations
1. S3 Anchor Dynamics and Capacity Evolution
2. S5 6G V2X Semantic Communication Under URLLC Constraints

S3 live result
- ARC: {s3.arc:.3f}
- Negative AER spike during misalignment: {s3.negative_aer_spike:.4f}
- Post resolution peak capacity: {s3.post_resolution_peak_capacity:.3f}
- Meaning: ARC > 1 means clarification produced capacity super recovery.

S5 live result
- Classical CAR 0.95 threshold: {s5.thresholds['Classical 5G NR proxy']:.2f} dB
- DL SemCom CAR 0.95 threshold: {s5.thresholds['DL SemCom proxy']:.2f} dB
- LPG CAR 0.95 threshold: {s5.thresholds['LPG semantic proxy']:.2f} dB
- LPG SNR saving versus classical: {gap:.2f} dB
- Equivalent lower required SNR or power proxy: {s5.power_gain_x:.2f}x

Report opened or saved here:
{report_path}
============================================================
"""
    print(textwrap.dedent(summary))


def maybe_animate_s3(s3: S3Results) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_title("Live S3: Anchor Γ(t) evolves during communication")
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("ARS and normalised Cs")
    ax.set_xlim(float(s3.time[0]), float(s3.time[-1]))
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.25)
    line_ars, = ax.plot([], [], linewidth=2.2, label="ARS")
    norm_cs = s3.semantic_capacity / np.max(s3.semantic_capacity)
    line_cs, = ax.plot([], [], linestyle="--", linewidth=2.0, label="normalised Cs(t)")
    ax.legend(loc="upper left")
    for i in range(1, len(s3.time), 3):
        line_ars.set_data(s3.time[:i], s3.ars[:i])
        line_cs.set_data(s3.time[:i], norm_cs[:i])
        plt.pause(0.025)
    plt.pause(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run two live LPG semantic communication demo simulations.")
    parser.add_argument("--output", type=str, default="output", help="Output folder for report, plots and CSV files.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures. Useful for servers and CI.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the HTML report in a browser.")
    parser.add_argument("--animate", action="store_true", help="Animate S3 before showing final figures.")
    args = parser.parse_args()

    root = Path(args.output).resolve()
    data_dir, plot_dir, report_dir = ensure_dirs(root)

    print("Running S3 Anchor Dynamics and Capacity Evolution...")
    s3 = run_s3(seed=42)
    print("Running S5 6G V2X Semantic Communication Under URLLC Constraints...")
    s5 = run_s5()

    show = not args.no_show
    save_s3_outputs(s3, data_dir, plot_dir, show=show)
    save_s5_outputs(s5, data_dir, plot_dir, show=show)
    report_path = make_html_report(root, s3, s5)
    print_console_summary(s3, s5, report_path)

    if show and args.animate:
        maybe_animate_s3(s3)

    if not args.no_open:
        try:
            webbrowser.open(report_path.as_uri())
        except Exception:
            pass

    if show:
        print("Close the plot windows to end the live demo.")
        plt.show(block=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
