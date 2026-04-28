import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\ruoch\Desktop\CU\Research\StarDist")
DATA_PATH = ROOT / "results_1path_analysis" / "analysis_data_1path.json"
FIG_DIR = ROOT / "docs" / "figures"


def load_image(b64):
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def save_overview(data):
    overview = load_image(data["overview"]["img"])
    draw = ImageDraw.Draw(overview)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    for idx, key in enumerate(data["cores"], start=1):
        core = data["cores"][key]
        x, y, r = core["cx_overview"], core["cy_overview"], core["r_overview"]
        color = "#334155" if idx == 7 else core["color"]
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=5)
        draw.text((x - 35, max(8, y - r - 28)), f"Core {idx}", fill=color, font=font)

    overview.thumbnail((1600, 1100), Image.Resampling.LANCZOS)
    overview.save(FIG_DIR / "overview_1path_cores.png", quality=95)


def save_core_counts(data):
    rows = data["analysis"]["summary"]
    labels = [f"Core {row['core']}" for row in rows]
    counts = [row["cells"] for row in rows]
    colors = [data["cores"][f"core{row['core']}"]["color"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    ax.bar(labels, counts, color=colors, edgecolor="#172033", linewidth=0.5)
    ax.set_ylabel("Detected nuclei")
    ax.set_title("Nuclear Count by Core")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    for i, value in enumerate(counts):
        ax.text(i, value + max(counts) * 0.015, f"{value:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "core_counts_1path.png", bbox_inches="tight")
    plt.close(fig)


def save_density(data):
    rows = data["analysis"]["summary"]
    labels = [f"Core {row['core']}" for row in rows]
    density = [row["density_per_mm2"] for row in rows]
    colors = [data["cores"][f"core{row['core']}"]["color"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    ax.bar(labels, density, color=colors, edgecolor="#172033", linewidth=0.5)
    ax.set_ylabel("Nuclei / mm2")
    ax.set_title("Nuclear Density by Core")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    for i, value in enumerate(density):
        ax.text(i, value + max(density) * 0.015, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "density_1path.png", bbox_inches="tight")
    plt.close(fig)


def save_slide_comparison(data):
    rows = data["slide_comparison"]
    names = [row["file"].replace(".svs", "") for row in rows]
    mpp = [float(row["mpp"]) for row in rows]
    ten = [float(row["tenengrad_mean_tissue_lowest"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    axes[0].bar(names, mpp, color=["#64748b", "#0ea5e9", "#0f766e"])
    axes[0].set_title("Pixel Size")
    axes[0].set_ylabel("um / pixel")
    axes[0].tick_params(axis="x", labelrotation=18)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].bar(names, ten, color=["#64748b", "#0ea5e9", "#0f766e"])
    axes[1].set_title("Sharpness Proxy")
    axes[1].set_ylabel("Tenengrad mean on tissue")
    axes[1].tick_params(axis="x", labelrotation=18)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Slide Comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "slide_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    save_overview(data)
    save_core_counts(data)
    save_density(data)
    save_slide_comparison(data)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
