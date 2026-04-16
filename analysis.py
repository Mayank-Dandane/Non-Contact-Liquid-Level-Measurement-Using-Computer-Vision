import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress

CALIB_FILE = "calibration.json"
LOG_FILE   = "level_log.csv"

# ── Ground truth for error analysis ───────────────────────
# Fill this in AFTER doing your physical test
# Format: (actual_mm, measured_mm_from_script)
# Example readings — replace with your real ones after testing
GROUND_TRUTH = [
    (0,   0.0),
    (20,  20.0),
    (40,  40.0),
    (60,  60.0),
    (80,  80.0),
    (100, 100.0),
]
# ──────────────────────────────────────────────────────────

def load_calibration():
    if not os.path.exists(CALIB_FILE):
        print("[WARNING] calibration.json not found. Skipping calib plot.")
        return None
    with open(CALIB_FILE) as f:
        return json.load(f)

def load_log():
    if not os.path.exists(LOG_FILE):
        print("[WARNING] level_log.csv not found. Generating dummy data for preview.")
        # Dummy sine wave data for preview
        times  = [f"00:{i:02d}:00" for i in range(20)]
        levels = [40 + 30 * np.sin(i * 0.4) for i in range(20)]
        return times, levels
    times, levels = [], []
    with open(LOG_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(row["timestamp"])
            levels.append(float(row["level_mm"]))
    return times, levels

def plot_calibration_curve(ax, calib):
    points = calib.get("points", [])
    if not points:
        ax.text(0.5, 0.5, "No calibration points found",
                ha="center", va="center", transform=ax.transAxes)
        return

    px  = np.array([p["pixel_y"] for p in points])
    mm  = np.array([p["real_mm"] for p in points])

    px_line = np.linspace(px.min() - 10, px.max() + 10, 200)
    mm_line = calib["slope"] * px_line + calib["intercept"]

    ax.scatter(px, mm, color="#1D9E75", s=60, zorder=5, label="Calibration points")
    ax.plot(px_line, mm_line, color="#0F6E56", linewidth=1.8,
            label=f"Fit: y = {calib['slope']:.4f}x + {calib['intercept']:.2f}")
    ax.set_xlabel("Pixel Y")
    ax.set_ylabel("Actual height (mm)")
    ax.set_title(f"Calibration curve  (R² = {calib['r_squared']:.4f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_level_vs_time(ax, times, levels):
    x = np.arange(len(levels))
    ax.plot(x, levels, color="#185FA5", linewidth=1.6, label="Measured level")
    ax.fill_between(x, levels, alpha=0.12, color="#185FA5")

    # X-axis: show every ~5th timestamp label
    step = max(1, len(times) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(times[::step], rotation=35, ha="right", fontsize=7)

    ax.set_xlabel("Time")
    ax.set_ylabel("Level (mm)")
    ax.set_title("Liquid level vs time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_error_analysis(ax, ground_truth):
    actuals   = np.array([g[0] for g in ground_truth], dtype=float)
    measured  = np.array([g[1] for g in ground_truth], dtype=float)
    errors    = measured - actuals
    pct_errors = np.where(actuals != 0,
                          np.abs(errors) / actuals * 100,
                          np.zeros_like(errors))

    colors = ["#E24B4A" if abs(e) > 2 else "#1D9E75" for e in errors]
    bars = ax.bar(actuals, pct_errors, width=6, color=colors, edgecolor="white")

    for bar, pct in zip(bars, pct_errors):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Actual level (mm)")
    ax.set_ylabel("Absolute error (%)")
    ax.set_title("Measurement error at each level")
    ax.set_xticks(actuals)
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    legend_els = [Patch(color="#1D9E75", label="Error ≤ 2 mm"),
                  Patch(color="#E24B4A", label="Error > 2 mm")]
    ax.legend(handles=legend_els, fontsize=8)

def plot_accuracy_table(ax, ground_truth):
    actuals  = [g[0] for g in ground_truth]
    measured = [g[1] for g in ground_truth]
    errors   = [round(m - a, 2) for a, m in zip(actuals, measured)]
    pct_errs = [
        f"{abs(e)/a*100:.1f}%" if a != 0 else "—"
        for a, e in zip(actuals, errors)
    ]

    col_labels = ["Actual (mm)", "Measured (mm)", "Error (mm)", "Error (%)"]
    rows = list(zip(
        [str(a) for a in actuals],
        [str(m) for m in measured],
        [str(e) for e in errors],
        pct_errs
    ))

    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Header styling
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#0C447C")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row coloring
    for i in range(1, len(rows) + 1):
        bg = "#F1EFE8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor(bg)

    ax.set_title("Accuracy table", fontweight="bold", pad=12)

def print_summary(ground_truth, calib):
    actuals  = np.array([g[0] for g in ground_truth], dtype=float)
    measured = np.array([g[1] for g in ground_truth], dtype=float)
    errors   = np.abs(measured - actuals)

    print("\n" + "=" * 50)
    print("  PERFORMANCE SUMMARY")
    print("=" * 50)
    if calib:
        print(f"  R² (calibration)   : {calib['r_squared']:.4f}")
        slope_mm_per_px = abs(calib['slope'])
        print(f"  Resolution         : {slope_mm_per_px:.3f} mm/pixel")
    print(f"  Max error          : {errors.max():.2f} mm")
    print(f"  Mean error         : {errors.mean():.2f} mm")
    print(f"  Std deviation      : {errors.std():.2f} mm")
    print("=" * 50)

def main():
    calib         = load_calibration()
    times, levels = load_log()

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Vision-Based Liquid Level Measurement — Performance Analysis",
                 fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    if calib:
        plot_calibration_curve(ax1, calib)
    else:
        ax1.text(0.5, 0.5, "Run calibrate.py first",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=11)
        ax1.set_title("Calibration curve")

    plot_level_vs_time(ax2, times, levels)
    plot_error_analysis(ax3, GROUND_TRUTH)
    plot_accuracy_table(ax4, GROUND_TRUTH)

    print_summary(GROUND_TRUTH, calib)

    plt.savefig("analysis_report.png", dpi=150, bbox_inches="tight")
    print("\n  Report saved → analysis_report.png")
    plt.show()

if __name__ == "__main__":
    main()
