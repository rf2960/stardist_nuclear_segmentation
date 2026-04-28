import base64
import csv
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
from csbdeep.utils import normalize
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from stardist.models import StarDist2D


ROOT = Path(r"C:\Users\ruoch\Desktop\CU\Research\StarDist")
SLIDE_PATH = ROOT / "TJ Cre Myc TMA2.svs"
OUT_DIR = ROOT / "results_tma2_clear"

PATCH_SIZE = 512
SEG_STEP = 256
GRID_STEP = 512
PROB_THRESH = {
    1: 0.05,
    2: 0.20,
    3: 0.10,
    4: 0.20,
    5: 0.20,
    6: 0.20,
    7: 0.10,
}
CORE_COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#a855f7", "#f8fafc"]


def encode_jpeg(image, quality=92):
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def enhance_patch(patch):
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def detect_cores(slide):
    thumbnail_img = slide.get_thumbnail((500, 500))
    thumb = np.array(thumbnail_img.convert("RGB"))
    scale_x = slide.level_dimensions[0][0] / thumb.shape[1]
    scale_y = slide.level_dimensions[0][1] / thumb.shape[0]

    gray = np.mean(thumb, axis=2)
    # TMA2 has a pale split in the lower-right core; 220 keeps the seven
    # expected cores connected without merging neighboring tissue spots.
    tissue_mask = gray < 220
    labeled_cores, num_cores = ndimage.label(tissue_mask)
    sizes = ndimage.sum(tissue_mask, labeled_cores, range(1, num_cores + 1))
    core_indices = [i + 1 for i, size in enumerate(sizes) if size > 1000]

    components = []
    for idx in core_indices:
        core_mask = labeled_cores == idx
        darkness_weight = (255 - gray) * core_mask
        total_weight = darkness_weight.sum()
        y_center = (darkness_weight * np.arange(thumb.shape[0])[:, None]).sum() / total_weight
        x_center = (darkness_weight * np.arange(thumb.shape[1])[None, :]).sum() / total_weight

        rows = np.where(core_mask.any(axis=1))[0]
        cols = np.where(core_mask.any(axis=0))[0]
        bbox_cy = (rows[0] + rows[-1]) / 2
        bbox_cx = (cols[0] + cols[-1]) / 2
        radius = ((rows[-1] - rows[0]) / 2 + (cols[-1] - cols[0]) / 2) / 2
        # Keep the center tied to the circular core footprint, while allowing
        # darker tissue to refine it slightly when staining is asymmetric.
        center_y = 0.7 * bbox_cy + 0.3 * y_center
        center_x = 0.7 * bbox_cx + 0.3 * x_center
        components.append({
            "x": float(center_x),
            "y": float(center_y),
            "radius": float(radius),
            "area": int(core_mask.sum()),
        })

    components = sorted(components, key=lambda item: item["area"], reverse=True)[:7]
    components = sorted(components, key=lambda item: (round(item["y"] / 80), item["x"]))
    centers = [(item["y"], item["x"]) for item in components]
    radii = [item["radius"] for item in components]

    if len(centers) != 7:
        raise RuntimeError(f"Expected 7 tissue cores, detected {len(centers)}")

    return thumb, scale_x, scale_y, tuple(centers), radii


def segment_cells(slide, model, scale_x, scale_y, centers, radii):
    all_cells = []
    patch_records_by_core = {f"core{i}": [] for i in range(1, 8)}

    for core_idx, center in enumerate(centers):
        core_num = core_idx + 1
        print(f"Processing core {core_num}/7", flush=True)
        full_cx = int(center[1] * scale_x)
        full_cy = int(center[0] * scale_y)
        core_radius_full = radii[core_idx] * scale_x
        core_size = max(int(core_radius_full * 2.2), 1024)
        prob_thresh = PROB_THRESH[core_num]
        raw_count = 0

        for row in range(0, core_size, SEG_STEP):
            for col in range(0, core_size, SEG_STEP):
                px = full_cx - core_size // 2 + col
                py = full_cy - core_size // 2 + row
                patch_cx = px + PATCH_SIZE // 2 - full_cx
                patch_cy = py + PATCH_SIZE // 2 - full_cy
                if np.hypot(patch_cx, patch_cy) > core_radius_full + PATCH_SIZE * 0.7:
                    continue

                patch = np.array(slide.read_region((px, py), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB"))
                if patch.mean() > 220:
                    continue

                yy, xx = np.mgrid[0:PATCH_SIZE, 0:PATCH_SIZE]
                pixel_dist = np.hypot(xx + px - full_cx, yy + py - full_cy)
                if (pixel_dist <= core_radius_full).mean() < 0.10:
                    continue

                labels, _details = model.predict_instances(
                    normalize(enhance_patch(patch), 1, 99.8),
                    prob_thresh=prob_thresh,
                    nms_thresh=0.3,
                )
                if labels.max() == 0:
                    continue

                for cell_id in range(1, labels.max() + 1):
                    cell_pixels = np.where(labels == cell_id)
                    cell_y = float(np.mean(cell_pixels[0]) + py)
                    cell_x = float(np.mean(cell_pixels[1]) + px)
                    area = int(len(cell_pixels[0]))
                    if np.hypot(cell_x - full_cx, cell_y - full_cy) > core_radius_full:
                        continue
                    all_cells.append({"core": core_num, "x": cell_x, "y": cell_y, "area_pixels": area})
                    raw_count += 1

        print(f"  raw detections: {raw_count}", flush=True)

    coords = np.array([[c["x"], c["y"]] for c in all_cells])
    areas = np.array([c["area_pixels"] for c in all_cells])
    to_remove = set()
    if len(coords):
        for i, j in cKDTree(coords).query_pairs(r=8.0):
            to_remove.add(j if areas[i] >= areas[j] else i)
    deduped = [c for idx, c in enumerate(all_cells) if idx not in to_remove]
    df = pd.DataFrame(deduped)
    df = df[(df["area_pixels"] >= 30) & (df["area_pixels"] <= 2000)].reset_index(drop=True)
    df.insert(1, "cell_id", df.groupby("core").cumcount() + 1)

    print(f"Final cells after NMS and size filtering: {len(df)}", flush=True)
    return df, patch_records_by_core


def export_viewer_data(slide, model, df, thumb, scale_x, scale_y, centers, radii):
    overview = slide.read_region((0, 0), 2, slide.level_dimensions[2]).convert("RGB")
    scale_lv2_x = slide.level_dimensions[0][0] / slide.level_dimensions[2][0]
    scale_lv2_y = slide.level_dimensions[0][1] / slide.level_dimensions[2][1]

    payload = {
        "slide": {
            "filename": SLIDE_PATH.name,
            "dimensions": list(slide.level_dimensions[0]),
            "mpp": slide.properties.get("aperio.MPP"),
            "appmag": slide.properties.get("aperio.AppMag"),
        },
        "overview": {
            "img": encode_jpeg(overview, 92),
            "width": overview.width,
            "height": overview.height,
            "scale_x": scale_lv2_x,
            "scale_y": scale_lv2_y,
        },
        "cores": {},
        "patches": {},
    }

    for core_idx, center in enumerate(centers):
        core_num = core_idx + 1
        full_cx = int(center[1] * scale_x)
        full_cy = int(center[0] * scale_y)
        core_radius_full = radii[core_idx] * scale_x
        lv1_scale = float(slide.level_downsamples[1])
        r1 = int(core_radius_full / lv1_scale)
        size1 = r1 * 2 + 40
        origin_x = int(full_cx - (size1 * lv1_scale) / 2)
        origin_y = int(full_cy - (size1 * lv1_scale) / 2)

        region = slide.read_region((origin_x, origin_y), 1, (size1, size1)).convert("RGB")
        core_df = df[df["core"] == core_num]
        cells = [
            {
                "x": round((row.x - origin_x) / lv1_scale, 1),
                "y": round((row.y - origin_y) / lv1_scale, 1),
                "area": int(row.area_pixels),
            }
            for row in core_df.itertuples(index=False)
        ]

        payload["cores"][f"core{core_num}"] = {
            "img": encode_jpeg(region, 94),
            "width": region.width,
            "height": region.height,
            "cells": cells,
            "total_cells": int(len(core_df)),
            "color": CORE_COLORS[core_idx],
            "cx_overview": round(full_cx / scale_lv2_x, 1),
            "cy_overview": round(full_cy / scale_lv2_y, 1),
            "r_overview": round(core_radius_full / scale_lv2_x, 1),
        }

        patches = []
        core_size = max(int(core_radius_full * 2.2), 1024)
        prob_thresh = PROB_THRESH[core_num]
        for row_i, row in enumerate(range(0, core_size, GRID_STEP)):
            for col_i, col in enumerate(range(0, core_size, GRID_STEP)):
                px = full_cx - core_size // 2 + col
                py = full_cy - core_size // 2 + row
                if np.hypot(px + PATCH_SIZE // 2 - full_cx, py + PATCH_SIZE // 2 - full_cy) > core_radius_full + PATCH_SIZE * 0.5:
                    continue

                patch_img = slide.read_region((px, py), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                patch_np = np.array(patch_img)
                if patch_np.mean() > 225:
                    continue

                labels, details = model.predict_instances(
                    normalize(enhance_patch(patch_np), 1, 99.8),
                    prob_thresh=prob_thresh,
                    nms_thresh=0.3,
                )

                polys = []
                for cell_id, coords in enumerate(details["coord"], start=1):
                    pts = np.array(coords).T
                    cy_abs = float(np.mean(pts[:, 0]) + py)
                    cx_abs = float(np.mean(pts[:, 1]) + px)
                    if np.hypot(cx_abs - full_cx, cy_abs - full_cy) > core_radius_full:
                        continue
                    pts_xy = [[round(float(p[1]), 1), round(float(p[0]), 1)] for p in pts]
                    polys.append({"pts": pts_xy, "area": int((labels == cell_id).sum())})

                patches.append({
                    "row": row_i,
                    "col": col_i,
                    "img": encode_jpeg(patch_img, 90),
                    "polys": polys,
                    "n_cells": len(polys),
                })
        payload["patches"][f"core{core_num}"] = {
            "items": patches,
            "n_rows": max((patch["row"] for patch in patches), default=-1) + 1,
            "n_cols": max((patch["col"] for patch in patches), default=-1) + 1,
        }
        print(f"Exported core {core_num}: {len(core_df)} cells, {len(patches)} patches", flush=True)

    return payload


def write_html(payload, path):
    payload_json = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TMA2 StarDist Nuclear Segmentation</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#667085; --line:#d8dee9; --panel:#ffffff; --bg:#f5f7fa; --accent:#0f766e; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }}
header {{ padding:18px 22px 12px; background:#ffffff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }}
h1 {{ margin:0; font-size:20px; }}
.meta {{ margin-top:6px; display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:13px; }}
main {{ display:grid; grid-template-columns:320px minmax(0,1fr); min-height:calc(100vh - 70px); }}
aside {{ padding:16px; border-right:1px solid var(--line); background:#fbfcfe; overflow:auto; }}
.core-list {{ display:grid; gap:8px; }}
button {{ font:inherit; }}
.core-btn {{ width:100%; display:flex; align-items:center; justify-content:space-between; gap:10px; border:1px solid var(--line); background:#fff; padding:10px 12px; border-radius:7px; cursor:pointer; }}
.core-btn.active {{ border-color:var(--accent); box-shadow:0 0 0 2px rgba(15,118,110,.12); }}
.dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; margin-right:8px; border:1px solid rgba(0,0,0,.25); }}
.stage {{ padding:16px; overflow:auto; }}
.tabs {{ display:flex; gap:8px; margin-bottom:12px; }}
.tab {{ border:1px solid var(--line); background:#fff; border-radius:7px; padding:8px 12px; cursor:pointer; }}
.tab.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.viewer {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px; }}
.image-wrap {{ position:relative; width:max-content; max-width:100%; margin:auto; }}
.image-wrap img {{ display:block; max-width:100%; height:auto; image-rendering:auto; }}
canvas.overlay {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
.patch-grid {{ display:grid; gap:10px; align-items:start; justify-content:start; overflow:auto; padding-bottom:8px; }}
.patch-card {{ background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; cursor:zoom-in; }}
.patch-card.empty {{ visibility:hidden; }}
.patch-card .image-wrap {{ width:100%; }}
.patch-card img {{ width:100%; }}
.patch-info {{ padding:8px 10px; font-size:13px; color:var(--muted); display:flex; justify-content:space-between; }}
.overview-grid {{ display:grid; grid-template-columns:minmax(0, 1fr); gap:14px; }}
.hint {{ margin:0 0 12px; color:var(--muted); font-size:13px; }}
.modal {{ position:fixed; inset:0; background:rgba(15,23,42,.72); display:none; align-items:center; justify-content:center; z-index:20; padding:24px; }}
.modal.open {{ display:flex; }}
.modal-inner {{ background:#fff; border-radius:8px; max-width:min(92vw, 980px); max-height:92vh; overflow:auto; padding:12px; box-shadow:0 20px 80px rgba(0,0,0,.35); }}
.modal-bar {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:8px; color:var(--muted); }}
.modal-close {{ border:1px solid var(--line); background:#fff; border-radius:7px; padding:6px 10px; cursor:pointer; }}
.modal .image-wrap {{ width:512px; max-width:100%; }}
</style>
</head>
<body>
<header>
  <h1>TMA2 StarDist Nuclear Segmentation</h1>
  <div class="meta">
    <span id="slideName"></span>
    <span id="slideDims"></span>
    <span id="slideMpp"></span>
    <span id="totalCells"></span>
  </div>
</header>
<main>
  <aside>
    <p class="hint">Select a core to inspect the half-resolution tissue image or full-resolution 512 px patches with StarDist polygon outlines.</p>
    <div class="core-list" id="coreList"></div>
  </aside>
  <section class="stage">
    <div class="tabs">
      <button class="tab active" data-view="overview">Overview</button>
      <button class="tab" data-view="core">Core detail</button>
      <button class="tab" data-view="patches">Patch grid</button>
    </div>
    <div class="viewer" id="content"></div>
  </section>
</main>
<div class="modal" id="modal" role="dialog" aria-modal="true">
  <div class="modal-inner">
    <div class="modal-bar"><span id="modalTitle"></span><button class="modal-close" id="modalClose">Close</button></div>
    <div id="modalContent"></div>
  </div>
</div>
<script>
const DATA = {payload_json};
let selected = "core1";
let view = "overview";
const coreKeys = Object.keys(DATA.cores);

function setMeta() {{
  const total = coreKeys.reduce((sum, key) => sum + DATA.cores[key].total_cells, 0);
  document.getElementById("slideName").textContent = DATA.slide.filename;
  document.getElementById("slideDims").textContent = `${{DATA.slide.dimensions[0]}} x ${{DATA.slide.dimensions[1]}} px`;
  document.getElementById("slideMpp").textContent = `${{DATA.slide.appmag}}x, ${{DATA.slide.mpp}} um/px`;
  document.getElementById("totalCells").textContent = `${{total.toLocaleString()}} nuclei`;
}}

function renderCoreList() {{
  const list = document.getElementById("coreList");
  list.innerHTML = "";
  coreKeys.forEach((key, i) => {{
    const core = DATA.cores[key];
    const btn = document.createElement("button");
    btn.className = "core-btn" + (key === selected ? " active" : "");
    btn.innerHTML = `<span><span class="dot" style="background:${{core.color}}"></span>Core ${{i + 1}}</span><strong>${{core.total_cells.toLocaleString()}}</strong>`;
    btn.onclick = () => {{ selected = key; if (view === "overview") view = "core"; syncTabs(); renderCoreList(); render(); }};
    list.appendChild(btn);
  }});
}}

function syncTabs() {{
  document.querySelectorAll(".tab").forEach(btn => btn.classList.toggle("active", btn.dataset.view === view));
}}

function canvasFor(img, draw) {{
  const wrap = img.closest(".image-wrap");
  const canvas = wrap.querySelector("canvas");
  const paint = () => {{
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    draw(canvas.getContext("2d"), canvas.width, canvas.height);
  }};
  if (img.complete) paint(); else img.onload = paint;
}}

function renderOverview() {{
  const overview = DATA.overview;
  document.getElementById("content").innerHTML = `<div class="overview-grid"><div class="image-wrap"><img id="overviewImg" src="data:image/jpeg;base64,${{overview.img}}" width="${{overview.width}}" height="${{overview.height}}"><canvas class="overlay"></canvas></div></div>`;
  canvasFor(document.getElementById("overviewImg"), ctx => {{
    ctx.lineWidth = 3;
    ctx.font = "bold 18px Arial";
    ctx.textAlign = "center";
    coreKeys.forEach((key, i) => {{
      const core = DATA.cores[key];
      ctx.strokeStyle = core.color;
      ctx.fillStyle = core.color;
      ctx.beginPath();
      ctx.arc(core.cx_overview, core.cy_overview, core.r_overview, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillText(`Core ${{i + 1}}`, core.cx_overview, Math.max(22, core.cy_overview - core.r_overview - 8));
    }});
  }});
}}

function renderCore() {{
  const core = DATA.cores[selected];
  document.getElementById("content").innerHTML = `<div class="image-wrap"><img id="coreImg" src="data:image/jpeg;base64,${{core.img}}" width="${{core.width}}" height="${{core.height}}"><canvas class="overlay"></canvas></div>`;
  canvasFor(document.getElementById("coreImg"), ctx => {{
    ctx.fillStyle = "rgba(255, 230, 0, .78)";
    core.cells.forEach(cell => {{
      const r = Math.max(1.8, Math.min(5, Math.sqrt(cell.area) / 4));
      ctx.beginPath();
      ctx.arc(cell.x, cell.y, r, 0, Math.PI * 2);
      ctx.fill();
    }});
  }});
}}

function renderPatches() {{
  const patchGroup = DATA.patches[selected];
  const patches = Array.isArray(patchGroup) ? patchGroup : patchGroup.items;
  const nCols = Array.isArray(patchGroup) ? Math.max(...patches.map(p => p.col)) + 1 : patchGroup.n_cols;
  const nRows = Array.isArray(patchGroup) ? Math.max(...patches.map(p => p.row)) + 1 : patchGroup.n_rows;
  const cells = [];
  const byPos = new Map(patches.map((p, i) => [`${{p.row}}:${{p.col}}`, {{p, i}}]));
  for (let row = 0; row < nRows; row++) {{
    for (let col = 0; col < nCols; col++) {{
      const hit = byPos.get(`${{row}}:${{col}}`);
      if (hit) {{
        const p = hit.p;
        cells.push(`<div class="patch-card" data-index="${{hit.i}}" style="grid-row:${{row + 1}};grid-column:${{col + 1}}"><div class="image-wrap"><img id="patch-${{hit.i}}" src="data:image/jpeg;base64,${{p.img}}" width="512" height="512"><canvas class="overlay"></canvas></div><div class="patch-info"><span>row ${{p.row}}, col ${{p.col}}</span><strong>${{p.n_cells}} nuclei</strong></div></div>`);
      }} else {{
        cells.push(`<div class="patch-card empty" style="grid-row:${{row + 1}};grid-column:${{col + 1}}"></div>`);
      }}
    }}
  }}
  document.getElementById("content").innerHTML = `<div class="patch-grid" style="grid-template-columns:repeat(${{nCols}}, minmax(180px, 220px));">${{cells.join("")}}</div>`;
  patches.forEach((patch, i) => {{
    canvasFor(document.getElementById(`patch-${{i}}`), ctx => {{
      ctx.strokeStyle = "rgba(255, 230, 0, .95)";
      ctx.lineWidth = 1.25;
      patch.polys.forEach(poly => {{
        if (!poly.pts.length) return;
        ctx.beginPath();
        ctx.moveTo(poly.pts[0][0], poly.pts[0][1]);
        poly.pts.slice(1).forEach(pt => ctx.lineTo(pt[0], pt[1]));
        ctx.closePath();
        ctx.stroke();
      }});
    }});
  }});
  document.querySelectorAll(".patch-card:not(.empty)").forEach(card => {{
    card.onclick = () => showPatch(patches[Number(card.dataset.index)]);
  }});
}}

function drawPatchOn(img, patch) {{
  canvasFor(img, ctx => {{
    ctx.strokeStyle = "rgba(255, 230, 0, .96)";
    ctx.lineWidth = 1.4;
    patch.polys.forEach(poly => {{
      if (!poly.pts.length) return;
      ctx.beginPath();
      ctx.moveTo(poly.pts[0][0], poly.pts[0][1]);
      poly.pts.slice(1).forEach(pt => ctx.lineTo(pt[0], pt[1]));
      ctx.closePath();
      ctx.stroke();
    }});
  }});
}}

function showPatch(patch) {{
  document.getElementById("modalTitle").textContent = `Core ${{coreKeys.indexOf(selected) + 1}} · row ${{patch.row}}, col ${{patch.col}} · ${{patch.n_cells}} nuclei`;
  document.getElementById("modalContent").innerHTML = `<div class="image-wrap"><img id="modalImg" src="data:image/jpeg;base64,${{patch.img}}" width="512" height="512"><canvas class="overlay"></canvas></div>`;
  document.getElementById("modal").classList.add("open");
  drawPatchOn(document.getElementById("modalImg"), patch);
}}

function render() {{
  if (view === "overview") renderOverview();
  if (view === "core") renderCore();
  if (view === "patches") renderPatches();
}}

document.querySelectorAll(".tab").forEach(btn => btn.onclick = () => {{ view = btn.dataset.view; syncTabs(); render(); }});
document.getElementById("modalClose").onclick = () => document.getElementById("modal").classList.remove("open");
document.getElementById("modal").onclick = event => {{ if (event.target.id === "modal") event.currentTarget.classList.remove("open"); }};
document.addEventListener("keydown", event => {{ if (event.key === "Escape") document.getElementById("modal").classList.remove("open"); }});
setMeta();
renderCoreList();
render();
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slide = openslide.OpenSlide(str(SLIDE_PATH))
    thumb, scale_x, scale_y, centers, radii = detect_cores(slide)
    print("Detected cores:", [(round(c[1], 1), round(c[0], 1)) for c in centers], flush=True)

    model = StarDist2D.from_pretrained("2D_versatile_he")
    df, _ = segment_cells(slide, model, scale_x, scale_y, centers, radii)
    df.to_csv(OUT_DIR / "detected_cells_tma2.csv", index=False)

    payload = export_viewer_data(slide, model, df, thumb, scale_x, scale_y, centers, radii)
    (OUT_DIR / "cell_data_tma2_clear.json").write_text(json.dumps(payload), encoding="utf-8")
    write_html(payload, OUT_DIR / "tma2_stardist_clear.html")
    print(f"Saved outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
