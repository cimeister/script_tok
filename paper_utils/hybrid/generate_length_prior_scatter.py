#!/usr/bin/env python3
"""Generate the README figure of MorphAlign versus compression for the Unigram length-prior runs.

One panel per language (English, German, Finnish). Every baseline method is a single marker; the
geometric-prior runs form a line from Unigram (lambda = 0) through the available lambdas; the learned
end-of-word (EOW) run is a filled marker. Reads only two caches: the compression grid and the MorphAlign cache.
The x axis is inverted so that better compression (fewer tokens) lies to the right and better MorphAlign
lies higher.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS_DIR = Path("results/hybrid")
GRID_PATH = RESULTS_DIR / "compression_train_eval_grid.json"
MORPHALIGN_CACHE_PATH = RESULTS_DIR / "cache_morphalign_scatter.json"
GRID_KEY = "fineweb->fishfood"
OUTPUT_DIR = Path("report/figures")
OUTPUT_STEM = "morphalign_vs_compression"

MORPHALIGN_SCALE = 100.0
FIGURE_SIZE = (12.0, 4.2)
DPI = 300
LAMBDA_LABEL_FONTSIZE = 7.5
LAMBDA_LABEL_OFFSET_POINTS = (0, 7)
MARKER_SIZE = 8
EOW_MARKER_SIZE = 11
LINE_WIDTH = 1.5
BPE_INIT_HALO = 4  # enlarges Unigram-BPE-Init so it stays visible behind the near-identical FSP-BPE-Init
X_MARGIN_FRACTION = 0.06

LANGUAGES = [("eng", "English"), ("deu", "German"), ("fin", "Finnish")]
GEO_LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0]
EOW_KEY = "unigram_eow"

# Light-mode categorical slots 1-7 and 8 from the dataviz reference palette; Unigram and the geometric line use ink.
INK = "#0b0b0b"
GRID_COLOR = "#d9d8d4"
SURFACE = "#fcfcfb"
TEXT_SECONDARY = "#52514e"

# method key: (legend label, color, marker, grid key, MorphAlign cache key prefix after "<lang>/")
BASELINES = {
    "unigram": ("Unigram", INK, "o", None, "ref/Default"),
    "bpe": ("BPE", "#2a78d6", "s", "bpe", "ref/BPE"),
    "fsp": ("FSP", "#eb6834", "^", "fsp", "ref/FSP"),
    "bpe_init": ("Unigram-BPE-Init", "#1baf7a", "v", "bpe_init", "bpe_init/f1.15/"),
    "bpe_init_fsp": ("FSP-BPE-Init", "#eda100", "<", "bpe_init_fsp", "bpe_init_fsp/f1.15/"),
    "mingram": ("MinGram", "#e87ba4", "D", "mingram", "mingram/mingram_f1.15_"),
    "mingram_pp": ("MinGram-PP", "#008300", "P", "mingram_pp", "mingram/mingram_f8.0_"),
    "pathpiece_bpe": ("PathPiece-BPE", "#6250d6", "h", "pathpiece_bpe", "pathpiece_bpe/"),
}
EOW_COLOR = "#e34948"


def lambda_text(value: float) -> str:
    return f"{value:g}"


def resolve_morphalign(cache: dict, lang: str, prefix: str) -> float:
    """MorphAlign x100 for the single non-token cache entry under `<lang>/<prefix>`; raise unless exactly one."""
    full_prefix = f"{lang}/{prefix}"
    matches = [k for k in cache if not k.startswith("tokens/") and k.startswith(full_prefix)]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one MorphAlign cache key starting with {full_prefix!r}, found {matches}")
    return float(cache[matches[0]]) * MORPHALIGN_SCALE


def collect_rows(grid_series: dict, cache: dict) -> list[dict]:
    rows = []
    for lang, _ in LANGUAGES:
        series = grid_series[lang]
        unigram_y = resolve_morphalign(cache, lang, BASELINES["unigram"][4])
        for method, (_, _, _, grid_key, prefix) in BASELINES.items():
            if grid_key is None:
                x = 0.0
            else:
                if grid_key not in series or series[grid_key] is None:
                    raise KeyError(f"baseline {grid_key!r} missing from compression grid for {lang}")
                x = float(series[grid_key])
            rows.append(_row(lang, method, "", x, resolve_morphalign(cache, lang, prefix)))
        rows.append(_row(lang, "unigram_geo", "0", 0.0, unigram_y))
        for lam in GEO_LAMBDAS:
            key = f"unigram_geo_{lambda_text(lam)}"
            x = series.get(key)
            matches = [k for k in cache if k.startswith(f"{lang}/{key}/")]
            if x is None or not matches:
                print(f"note: {lang} has no {key} (grid={x is not None}, morphalign={bool(matches)}); skipped")
                continue
            rows.append(
                _row(lang, "unigram_geo", lambda_text(lam), float(x), resolve_morphalign(cache, lang, f"{key}/"))
            )
        x = series.get(EOW_KEY)
        if x is None:
            print(f"note: {lang} has no {EOW_KEY} in grid; skipped")
        else:
            rows.append(_row(lang, EOW_KEY, "", float(x), resolve_morphalign(cache, lang, f"{EOW_KEY}/")))
    return rows


def _row(lang: str, method: str, lam: str, x: float, y: float) -> dict:
    return {"language": lang, "method": method, "lambda": lam, "compression_pct": x, "morphalign_x100": y}


def draw(rows: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=FIGURE_SIZE, facecolor=SURFACE)
    for ax, (lang, name) in zip(axes, LANGUAGES, strict=True):
        lang_rows = [r for r in rows if r["language"] == lang]
        ax.set_facecolor(SURFACE)
        geo = sorted((r for r in lang_rows if r["method"] == "unigram_geo"), key=lambda r: float(r["lambda"]))
        ax.plot(
            [r["compression_pct"] for r in geo],
            [r["morphalign_x100"] for r in geo],
            color=TEXT_SECONDARY,
            lw=LINE_WIDTH,
            marker="o",
            ms=MARKER_SIZE - 3,
            mfc=SURFACE,
            mec=TEXT_SECONDARY,
            zorder=2,
        )
        for r in geo[1:]:
            ax.annotate(
                lambda_text(float(r["lambda"])),
                (r["compression_pct"], r["morphalign_x100"]),
                textcoords="offset points",
                xytext=LAMBDA_LABEL_OFFSET_POINTS,
                ha="center",
                fontsize=LAMBDA_LABEL_FONTSIZE,
                color=TEXT_SECONDARY,
            )
        for r in lang_rows:
            if r["method"] in BASELINES:
                _, color, marker, _, _ = BASELINES[r["method"]]
                ax.plot(
                    r["compression_pct"],
                    r["morphalign_x100"],
                    marker,
                    color=color,
                    ms=MARKER_SIZE + (BPE_INIT_HALO if r["method"] == "bpe_init" else 0),
                    mec=SURFACE,
                    mew=1.0,
                    zorder=3 + (r["method"] == "bpe_init_fsp"),
                )
            elif r["method"] == EOW_KEY:
                ax.plot(
                    r["compression_pct"],
                    r["morphalign_x100"],
                    "*",
                    color=EOW_COLOR,
                    ms=EOW_MARKER_SIZE,
                    mec=SURFACE,
                    mew=0.8,
                    zorder=4,
                )
                ax.annotate(
                    "EOW",
                    (r["compression_pct"], r["morphalign_x100"]),
                    textcoords="offset points",
                    xytext=(-9, -6),
                    ha="right",
                    fontsize=LAMBDA_LABEL_FONTSIZE + 0.5,
                    color=EOW_COLOR,
                )
        ax.set_title(name, fontsize=12, color=INK)
        ax.set_xlabel("Token count change vs Unigram (%)", fontsize=10, color=TEXT_SECONDARY)
        ax.grid(color=GRID_COLOR, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.margins(x=X_MARGIN_FRACTION, y=0.12)
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
    axes[0].set_ylabel("MorphAlign (x100)", fontsize=10, color=TEXT_SECONDARY)
    handles = [
        Line2D([], [], ls="", marker=m, color=c, ms=MARKER_SIZE, label=label)
        for label, c, m, _, _ in BASELINES.values()
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=TEXT_SECONDARY,
            lw=LINE_WIDTH,
            marker="o",
            ms=MARKER_SIZE - 3,
            mfc=SURFACE,
            label="Unigram, geometric prior (labels: $\\lambda$)",
        )
    )
    handles.append(Line2D([], [], ls="", marker="*", color=EOW_COLOR, ms=EOW_MARKER_SIZE, label="Unigram, learned EOW"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig


def main() -> None:
    with open(GRID_PATH) as f:
        grid_series = json.load(f)[GRID_KEY]["series"]
    with open(MORPHALIGN_CACHE_PATH) as f:
        cache = json.load(f)
    rows = collect_rows(grid_series, cache)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / f"{OUTPUT_STEM}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig = draw(rows)
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.png", dpi=DPI, facecolor=SURFACE)
    print(f"wrote {OUTPUT_DIR / OUTPUT_STEM}.png and .csv ({len(rows)} points)")


if __name__ == "__main__":
    main()
