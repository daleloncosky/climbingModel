import argparse
from pathlib import Path
from typing import Sequence
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]
SHOW_PLOTS = False
 
# Helper funcs
def _save_and_show(fig_path: Path) -> None: 
    """Save and close a figure; display it only when requested."""
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()
 
 
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True) 
 
# Plotting and statistics helpers
def scatter_with_trend(x: pd.Series, y: pd.Series, color=None, label=None):
    mask = np.isfinite(x) & np.isfinite(y)
    clean_x, clean_y = x[mask], y[mask]
    plt.scatter(clean_x, clean_y, alpha=0.22, s=12, color=color, label=label)
    if len(clean_x) < 2 or clean_x.nunique() < 2:
        return np.nan, np.nan, np.nan
    m, b = np.polyfit(clean_x, clean_y, 1)
    xs = np.linspace(clean_x.min(), clean_x.max(), 200)
    plt.plot(xs, m * xs + b, color=color)
    correlation = clean_x.corr(clean_y) if clean_y.nunique() > 1 else np.nan
    return m, b, correlation

# hexbin plot
def hexbin_plot(df: pd.DataFrame, outdir: Path) -> None:
    fig_name = "hexbin_bmi_vs_max"
 
    # Plot
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(df["bmi"], df["grades_max"], gridsize=30, cmap="Blues", mincnt=1)
    plt.colorbar(label="Count")
    plt.xlabel("BMI (kg/m²)")
    plt.ylabel("Maximum grade (dataset encoding)")
    plt.title("Hexbin Density: BMI vs Maximum Grade")
    plt.xlim(15, 30)
    _save_and_show(outdir / f"{fig_name}.png")
 
    # Export counts matrix
    counts = hb.get_array().reshape(hb.get_offsets().shape[0])
    offsets = hb.get_offsets()
    hex_df = pd.DataFrame(offsets, columns=["bmi_center", "grade_center"])
    hex_df["count"] = counts
    hex_df.to_csv(outdir / f"{fig_name}_counts.csv", index=False)
 
 
# sex plot
def sex_coloured_plot(df: pd.DataFrame, outdir: Path) -> None:
    fig_name = "bmi_sex"
    plt.figure(figsize=(8, 6))
    stats_lines = []
    for sex_val, sex_name, color in [(0, "Male", "tab:blue"), (1, "Female", "tab:orange")]:
        sub = df[df["sex"] == sex_val]
        if sub.empty:
            continue
        m, b, r = scatter_with_trend(sub["bmi"], sub["grades_max"], color=color)
        plt.scatter([], [], label=f"{sex_name} (r={r:.2f})", color=color)  # legend handle
        stats_lines.append(f"{sex_name}: slope={m:.3f}, intercept={b:.3f}, r={r:.3f}, n={len(sub)}") 
    plt.xlabel("BMI (kg/m²)")
    plt.ylabel("Maximum grade (dataset encoding)")
    plt.title("BMI vs Maximum Grade | coloured by Sex")
    plt.xlim(15, 30)
    plt.legend()
    _save_and_show(outdir / f"{fig_name}.png")
 
    # Write text summary
    (outdir / f"{fig_name}_stats.txt").write_text("\n".join(stats_lines))
 
# age band plot
def age_band_plot(df: pd.DataFrame, age_labels: Sequence[str], outdir: Path) -> None:
    fig_name = "bmi_age"
    plt.figure(figsize=(8, 6))
    stats_lines = []
 
    # Use a consistent color palette
    cmap = plt.get_cmap("tab10")  # or 'Set1', 'Dark2', etc.
    colors = {label: cmap(i) for i, label in enumerate(age_labels)}
 
    for band in age_labels:
        sub = df[df["age_band"] == band]
        if sub.empty:
            continue
        color = colors[band]
        m, b, r = scatter_with_trend(sub["bmi"], sub["grades_max"], color=color)
        plt.scatter([], [], label=f"{band} (r={r:.2f})", color=color)
        stats_lines.append(f"Age {band}: slope={m:.3f}, intercept={b:.3f}, r={r:.3f}, n={len(sub)}")
 
    plt.xlabel("BMI (kg/m²)")
    plt.ylabel("Maximum grade (dataset encoding)")
    plt.title("BMI vs Maximum Grade | coloured by Age Band")
    plt.xlim(15, 30)
    plt.legend(title="Age Band")
    _save_and_show(outdir / f"{fig_name}.png")
 
    (outdir / f"{fig_name}_stats.txt").write_text("\n".join(stats_lines))
 
 
# country boxplot
def boxplot_by_country(df: pd.DataFrame, metric: str, outdir: Path, top_n: int = 10) -> None:
    fig_name = "country_boxplot"
    counts = df["country"].value_counts().nlargest(top_n).index
    subset = df[df["country"].isin(counts)]
 
    if subset.empty:
        return
    subset.boxplot(column=metric, by="country", rot=45, figsize=(10, 6))
    plt.ylabel("Maximum grade (dataset encoding)" if metric == "grades_max" else metric)
    plt.title(f"Recorded maximum grade by country (top {top_n})")
    plt.suptitle("")
    _save_and_show(outdir / f"{fig_name}.png")
 
    country_stats = subset.groupby("country")[metric].agg(["count", "mean", "median"])
    country_stats.to_csv(outdir / f"{fig_name}_stats.csv")
 
# load climber data
def load_climber_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"height", "weight", "age", "sex", "grades_max", "grades_mean", "country"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}") 
 
    for column in ["height", "weight", "age", "sex", "grades_max", "grades_mean"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    valid = (df["height"] > 0) & (df["weight"] > 0)
    valid &= np.isfinite(df[["height", "weight", "grades_max"]]).all(axis=1)
    df = df.loc[valid].copy()
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    df = df.loc[np.isfinite(df["bmi"])].copy()
    df.loc[(df["age"] <= 0) | ~np.isfinite(df["age"]), "age"] = np.nan
    # Left-closed intervals place age 25 in 25-34, age 35 in 35-44, etc.
    df["age_band"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, np.inf], labels=AGE_LABELS, right=False)
    return df

def main() -> None:
    global SHOW_PLOTS
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Climber skill analysis with plot + text exports")
    parser.add_argument("--csv", default=project_dir / "data/climber_df.csv", help="Path to climber CSV")
    parser.add_argument("--outdir", default=project_dir / "figures", help="climbingModel/figures output directory")
    parser.add_argument("--show", action="store_true", help="Also display figures interactively")
    args = parser.parse_args()
    SHOW_PLOTS = args.show
 
    outdir = Path(args.outdir)
    _ensure_dir(outdir)
 
    df = load_climber_data(Path(args.csv))
 
    if df.empty:
        parser.error("No rows contain positive height, positive weight, and a finite maximum grade.")

    # 1. Correlation matrix CSV
    corr = df[["height", "weight", "bmi", "grades_max", "grades_mean"]].corr()
    corr.round(3).to_csv(outdir / "corr_matrix.csv") 
    print("\n=== Pearson Correlation Matrix ===")
    print(corr.round(2).to_string())
 
    # 2‑5. Plots + text/CSV summaries
    hexbin_plot(df, outdir)
    sex_coloured_plot(df, outdir)
    age_labels = AGE_LABELS
    age_band_plot(df, age_labels, outdir)
    boxplot_by_country(df, metric="grades_max", outdir=outdir)
 
if __name__ == "__main__":
    main()