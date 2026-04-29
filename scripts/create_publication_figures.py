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


def get_fonts():
    try:
        return ImageFont.truetype("arial.ttf", 22), ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default(), ImageFont.load_default()


def draw_polygons(image, polygons, scale=1.0, color=(255, 230, 0), width=2):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for poly in polygons:
        pts = [(x * scale, y * scale) for x, y in poly["pts"]]
        if len(pts) > 2:
            draw.line(pts + [pts[0]], fill=color, width=width)
    return overlay


def save_overview(data):
    overview = load_image(data["overview"]["img"])
    draw = ImageDraw.Draw(overview)
    font, _small = get_fonts()

    for idx, key in enumerate(data["cores"], start=1):
        core = data["cores"][key]
        x, y, r = core["cx_overview"], core["cy_overview"], core["r_overview"]
        color = "#334155" if idx == 7 else core["color"]
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=5)
        draw.text((x - 35, max(8, y - r - 28)), f"Core {idx}", fill=color, font=font)

    overview.thumbnail((1600, 1100), Image.Resampling.LANCZOS)
    overview.save(FIG_DIR / "overview_1path_cores.png", quality=95)


def save_core_detail(data, core_key="core7"):
    core_num = int(core_key.replace("core", ""))
    core = data["cores"][core_key]
    image = load_image(core["img"])
    draw = ImageDraw.Draw(image)
    font, small = get_fonts()
    for cell in core["cells"]:
        r = max(2, min(6, np.sqrt(cell["area"]) / 4))
        x, y = cell["x"], cell["y"]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 230, 0))
    image.thumbnail((1400, 1000), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image.width, image.height + 54), "white")
    title = f"Viewer core detail: Core {core_num} ({core['total_cells']:,} detected nuclei)"
    d = ImageDraw.Draw(canvas)
    d.text((16, 14), title, fill=(23, 32, 51), font=font)
    canvas.paste(image, (0, 54))
    canvas.save(FIG_DIR / f"viewer_core_detail_{core_key}.png", quality=95)


def save_patch_grid(data, core_key="core1"):
    group = data["patches"][core_key]
    core_num = int(core_key.replace("core", ""))
    tile = 170
    label_h = 36
    font, small = get_fonts()
    canvas = Image.new(
        "RGB",
        (group["n_cols"] * tile, group["n_rows"] * (tile + label_h) + 54),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 14), f"Viewer patch grid: Core {core_num}", fill=(23, 32, 51), font=font)
    by_pos = {(p["row"], p["col"]): p for p in group["items"]}
    for row in range(group["n_rows"]):
        for col in range(group["n_cols"]):
            patch = by_pos.get((row, col))
            if patch is None:
                continue
            x = col * tile
            y = 54 + row * (tile + label_h)
            img = load_image(patch["img"]).resize((tile, tile), Image.Resampling.LANCZOS)
            img = draw_polygons(img, patch["polys_thumb"], scale=tile / 256, width=1)
            canvas.paste(img, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile + label_h - 1), outline=(203, 213, 225))
            draw.text((x + 6, y + tile + 7), f"r{row}, c{col}  {patch['n_cells']} nuclei", fill=(71, 85, 105), font=small)
    canvas.save(FIG_DIR / f"viewer_patch_grid_{core_key}.png", quality=92)


def save_selected_patch(data, core_key="core1", row=1, col=3):
    core_num = int(core_key.replace("core", ""))
    patch = next(p for p in data["patches"][core_key]["items"] if p["row"] == row and p["col"] == col)
    image = load_image(patch["img_hi"])
    scale = image.width / patch["hi_width"]
    image = draw_polygons(image, patch["polys_hi"], scale=scale, width=3)
    image.thumbnail((1500, 950), Image.Resampling.LANCZOS)
    font, _small = get_fonts()
    canvas = Image.new("RGB", (image.width, image.height + 58), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"Selected high-resolution patch: Core {core_num}, row {row}, col {col} ({patch['n_cells']} nuclei)"
    draw.text((16, 16), title, fill=(23, 32, 51), font=font)
    canvas.paste(image, (0, 58))
    canvas.save(FIG_DIR / f"viewer_selected_patch_core{core_num}_row{row}_col{col}.png", quality=95)


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


def save_area_histogram(data):
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    colors = [data["cores"][f"core{i}"]["color"] for i in range(1, 8)]
    grid = np.linspace(20, 500, 500)
    bandwidth = 18.0
    for i in range(1, 8):
        areas = np.array([cell["area"] for cell in data["cores"][f"core{i}"]["cells"]], dtype=float)
        if len(areas) == 0:
            continue
        density = np.exp(-0.5 * ((grid[:, None] - areas[None, :]) / bandwidth) ** 2).sum(axis=1)
        density /= len(areas) * bandwidth * np.sqrt(2 * np.pi)
        ax.plot(grid, density, linewidth=2.0, color=colors[i - 1], label=f"Core {i}")
    ax.set_title("Smoothed Nuclear Mask Area Distribution")
    ax.set_xlabel("Predicted nuclear mask area at model scale (px)")
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nuclear_area_histogram.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    save_overview(data)
    save_core_counts(data)
    save_density(data)
    save_core_detail(data, "core7")
    save_patch_grid(data, "core1")
    save_selected_patch(data, "core1", 1, 3)
    save_area_histogram(data)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
