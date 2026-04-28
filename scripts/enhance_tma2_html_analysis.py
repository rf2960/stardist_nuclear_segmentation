import base64
import json
import math
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import openslide
from PIL import Image
from scipy import ndimage


ROOT = Path(r"C:\Users\ruoch\Desktop\CU\Research\StarDist")
OUT_DIR = ROOT / "results_tma2_clear"
JSON_PATH = OUT_DIR / "cell_data_tma2_clear.json"
HTML_PATH = OUT_DIR / "tma2_stardist_clear.html"
PATH_SLIDE = ROOT / "1-pathology core stained.svs"
COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#a855f7", "#f8fafc"]


def encode_jpeg(image, quality=92):
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def detect_cores(slide, threshold=230):
    thumb = np.array(slide.get_thumbnail((500, 500)).convert("RGB"))
    scale_x = slide.level_dimensions[0][0] / thumb.shape[1]
    scale_y = slide.level_dimensions[0][1] / thumb.shape[0]
    gray = np.mean(thumb, axis=2)
    labeled, n = ndimage.label(gray < threshold)
    sizes = ndimage.sum(gray < threshold, labeled, range(1, n + 1))
    components = []
    for idx, size in enumerate(sizes, start=1):
        if size <= 1000:
            continue
        mask = labeled == idx
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        darkness = (255 - gray) * mask
        total = darkness.sum()
        wy = (darkness * np.arange(thumb.shape[0])[:, None]).sum() / total
        wx = (darkness * np.arange(thumb.shape[1])[None, :]).sum() / total
        bbox_y = (rows[0] + rows[-1]) / 2
        bbox_x = (cols[0] + cols[-1]) / 2
        radius = ((rows[-1] - rows[0]) / 2 + (cols[-1] - cols[0]) / 2) / 2
        components.append({
            "x": float(0.7 * bbox_x + 0.3 * wx),
            "y": float(0.7 * bbox_y + 0.3 * wy),
            "radius": float(radius),
            "area": int(mask.sum()),
        })
    components = sorted(components, key=lambda item: item["area"], reverse=True)[:7]
    components = sorted(components, key=lambda item: (round(item["y"] / 80), item["x"]))
    if len(components) != 7:
        raise RuntimeError(f"Expected 7 pathology cores, detected {len(components)}")
    return scale_x, scale_y, components


def build_pathology_reference():
    slide = openslide.OpenSlide(str(PATH_SLIDE))
    scale_x, scale_y, cores = detect_cores(slide)
    overview = slide.read_region((0, 0), 2, slide.level_dimensions[2]).convert("RGB")
    scale_lv2_x = slide.level_dimensions[0][0] / slide.level_dimensions[2][0]
    scale_lv2_y = slide.level_dimensions[0][1] / slide.level_dimensions[2][1]

    payload = {
        "filename": PATH_SLIDE.name,
        "dimensions": list(slide.level_dimensions[0]),
        "mpp": slide.properties.get("aperio.MPP") or slide.properties.get("openslide.mpp-x"),
        "appmag": slide.properties.get("aperio.AppMag") or slide.properties.get("openslide.objective-power"),
        "overview": {
            "img": encode_jpeg(overview, 90),
            "width": overview.width,
            "height": overview.height,
        },
        "cores": {},
    }

    for i, core in enumerate(cores, start=1):
        full_cx = int(core["x"] * scale_x)
        full_cy = int(core["y"] * scale_y)
        full_r = core["radius"] * scale_x
        payload["cores"][f"core{i}"] = {
            "cx_overview": round(full_cx / scale_lv2_x, 1),
            "cy_overview": round(full_cy / scale_lv2_y, 1),
            "r_overview": round(full_r / scale_lv2_x, 1),
            "color": COLORS[i - 1],
        }

        lv1_scale = float(slide.level_downsamples[1])
        r1 = int(full_r / lv1_scale)
        size = r1 * 2 + 40
        origin_x = int(full_cx - (size * lv1_scale) / 2)
        origin_y = int(full_cy - (size * lv1_scale) / 2)
        region = slide.read_region((origin_x, origin_y), 1, (size, size)).convert("RGB")
        region.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        payload["cores"][f"core{i}"].update({
            "img": encode_jpeg(region, 92),
            "width": region.width,
            "height": region.height,
        })
    return payload


def add_analysis(data):
    mpp = float(data["slide"]["mpp"])
    summary = []
    all_areas = []
    for idx, key in enumerate(data["cores"], start=1):
        core = data["cores"][key]
        areas = np.array([cell["area"] for cell in core["cells"]], dtype=float)
        all_areas.extend(areas.tolist())
        full_radius_px = core["r_overview"] * data["overview"]["scale_x"]
        area_mm2 = math.pi * (full_radius_px * mpp / 1000) ** 2
        summary.append({
            "core": idx,
            "cells": int(core["total_cells"]),
            "area_mm2": round(area_mm2, 3),
            "density_per_mm2": round(core["total_cells"] / area_mm2, 1) if area_mm2 else 0,
            "mean_area": round(float(np.mean(areas)), 1) if len(areas) else 0,
            "median_area": round(float(np.median(areas)), 1) if len(areas) else 0,
            "p90_area": round(float(np.percentile(areas, 90)), 1) if len(areas) else 0,
        })

    patch_stats = {}
    for key, group in data["patches"].items():
        items = group["items"] if isinstance(group, dict) else group
        patch_stats[key] = sorted(
            [{"row": p["row"], "col": p["col"], "n_cells": p["n_cells"]} for p in items],
            key=lambda item: item["n_cells"],
            reverse=True,
        )

    data["analysis"] = {
        "summary": summary,
        "overall": {
            "total_cells": int(sum(row["cells"] for row in summary)),
            "mean_area": round(float(np.mean(all_areas)), 1) if all_areas else 0,
            "median_area": round(float(np.median(all_areas)), 1) if all_areas else 0,
        },
        "patch_stats": patch_stats,
    }
    data["pathology_reference"] = build_pathology_reference()
    return data


def write_html(data):
    payload_json = json.dumps(data, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TMA2 StarDist Nuclear Segmentation</title>
<style>
:root {{ color-scheme:light; --ink:#172033; --muted:#667085; --line:#d8dee9; --panel:#fff; --bg:#f5f7fa; --accent:#0f766e; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }}
header {{ padding:18px 22px 12px; background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }}
h1 {{ margin:0; font-size:20px; }}
.meta {{ margin-top:6px; display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:13px; }}
main {{ display:grid; grid-template-columns:320px minmax(0,1fr); min-height:calc(100vh - 70px); }}
aside {{ padding:16px; border-right:1px solid var(--line); background:#fbfcfe; overflow:auto; }}
.stage {{ padding:16px; overflow:auto; }}
.core-list {{ display:grid; gap:8px; }}
button {{ font:inherit; }}
.core-btn {{ width:100%; display:flex; align-items:center; justify-content:space-between; gap:10px; border:1px solid var(--line); background:#fff; padding:10px 12px; border-radius:7px; cursor:pointer; }}
.core-btn.active {{ border-color:var(--accent); box-shadow:0 0 0 2px rgba(15,118,110,.12); }}
.dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; margin-right:8px; border:1px solid rgba(0,0,0,.25); }}
.tabs {{ display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; }}
.tab,.tool-btn {{ border:1px solid var(--line); background:#fff; border-radius:7px; padding:8px 12px; cursor:pointer; }}
.tab.active,.tool-btn.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
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
.hint {{ margin:0 0 12px; color:var(--muted); font-size:13px; }}
.split {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; align-items:start; }}
.panel-title {{ margin:0 0 8px; font-size:14px; color:var(--muted); }}
.analysis-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; margin-bottom:14px; }}
.metric {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; }}
.metric strong {{ display:block; font-size:22px; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; background:#fff; }}
th,td {{ border-bottom:1px solid var(--line); padding:9px 8px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-weight:600; }}
.bar {{ height:7px; background:#e5e7eb; border-radius:999px; overflow:hidden; min-width:80px; }}
.bar span {{ display:block; height:100%; background:var(--accent); }}
.modal {{ position:fixed; inset:0; background:rgba(15,23,42,.72); display:none; align-items:center; justify-content:center; z-index:20; padding:24px; }}
.modal.open {{ display:flex; }}
.modal-inner {{ background:#fff; border-radius:8px; max-width:min(92vw, 1100px); max-height:92vh; overflow:auto; padding:12px; box-shadow:0 20px 80px rgba(0,0,0,.35); }}
.modal-bar {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:8px; color:var(--muted); }}
.modal-close {{ border:1px solid var(--line); background:#fff; border-radius:7px; padding:6px 10px; cursor:pointer; }}
.modal .image-wrap {{ width:512px; max-width:100%; }}
@media (max-width: 900px) {{ main {{ grid-template-columns:1fr; }} aside {{ border-right:0; border-bottom:1px solid var(--line); }} .split {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>TMA2 StarDist Nuclear Segmentation</h1>
  <div class="meta"><span id="slideName"></span><span id="slideDims"></span><span id="slideMpp"></span><span id="pathMeta"></span><span id="totalCells"></span></div>
</header>
<main>
  <aside>
    <p class="hint">Core order is row-major. Patch grid keeps spatial position; click any patch for a larger polygon overlay.</p>
    <div class="core-list" id="coreList"></div>
  </aside>
  <section class="stage">
    <div class="tabs">
      <button class="tab active" data-view="overview">Overview</button>
      <button class="tab" data-view="core">Core detail</button>
      <button class="tab" data-view="reference">1-path reference</button>
      <button class="tab" data-view="patches">Patch grid</button>
      <button class="tab" data-view="analysis">Analysis</button>
    </div>
    <div class="viewer" id="content"></div>
  </section>
</main>
<div class="modal" id="modal" role="dialog" aria-modal="true"><div class="modal-inner"><div class="modal-bar"><span id="modalTitle"></span><button class="modal-close" id="modalClose">Close</button></div><div id="modalContent"></div></div></div>
<script>
const DATA = {payload_json};
let selected = "core1";
let view = "overview";
let overviewMode = "tma2";
const coreKeys = Object.keys(DATA.cores);

function setMeta() {{
  const total = DATA.analysis.overall.total_cells;
  document.getElementById("slideName").textContent = DATA.slide.filename;
  document.getElementById("slideDims").textContent = `${{DATA.slide.dimensions[0]}} x ${{DATA.slide.dimensions[1]}} px`;
  document.getElementById("slideMpp").textContent = `${{DATA.slide.appmag}}x, ${{DATA.slide.mpp}} um/px`;
  document.getElementById("pathMeta").textContent = `reference: ${{DATA.pathology_reference.appmag}}x, ${{DATA.pathology_reference.mpp}} um/px`;
  document.getElementById("totalCells").textContent = `${{total.toLocaleString()}} nuclei`;
}}
function renderCoreList() {{
  const list = document.getElementById("coreList"); list.innerHTML = "";
  coreKeys.forEach((key, i) => {{
    const core = DATA.cores[key];
    const btn = document.createElement("button");
    btn.className = "core-btn" + (key === selected ? " active" : "");
    btn.innerHTML = `<span><span class="dot" style="background:${{core.color}}"></span>Core ${{i + 1}}</span><strong>${{core.total_cells.toLocaleString()}}</strong>`;
    btn.onclick = () => {{ selected = key; if (view === "overview") view = "core"; syncTabs(); renderCoreList(); render(); }};
    list.appendChild(btn);
  }});
}}
function syncTabs() {{ document.querySelectorAll(".tab").forEach(btn => btn.classList.toggle("active", btn.dataset.view === view)); }}
function canvasFor(img, draw) {{
  const canvas = img.closest(".image-wrap").querySelector("canvas");
  const paint = () => {{ canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; draw(canvas.getContext("2d"), canvas.width, canvas.height); }};
  if (img.complete) paint(); else img.onload = paint;
}}
function drawCores(img, source) {{
  canvasFor(img, ctx => {{
    ctx.lineWidth = 3; ctx.font = "bold 18px Arial"; ctx.textAlign = "center";
    coreKeys.forEach((key, i) => {{
      const core = source.cores ? source.cores[key] : DATA.cores[key];
      ctx.strokeStyle = core.color; ctx.fillStyle = core.color;
      ctx.beginPath(); ctx.arc(core.cx_overview, core.cy_overview, core.r_overview, 0, Math.PI * 2); ctx.stroke();
      ctx.fillText(`Core ${{i + 1}}`, core.cx_overview, Math.max(22, core.cy_overview - core.r_overview - 8));
    }});
  }});
}}
function renderOverview() {{
  const src = overviewMode === "path" ? DATA.pathology_reference : DATA;
  const ov = overviewMode === "path" ? DATA.pathology_reference.overview : DATA.overview;
  document.getElementById("content").innerHTML = `<div class="tabs"><button class="tool-btn ${{overviewMode === "tma2" ? "active" : ""}}" id="tma2Btn">TMA2 segmentation slide</button><button class="tool-btn ${{overviewMode === "path" ? "active" : ""}}" id="pathBtn">1-path stained reference</button></div><div class="image-wrap"><img id="overviewImg" src="data:image/jpeg;base64,${{ov.img}}" width="${{ov.width}}" height="${{ov.height}}"><canvas class="overlay"></canvas></div>`;
  document.getElementById("tma2Btn").onclick = () => {{ overviewMode = "tma2"; renderOverview(); }};
  document.getElementById("pathBtn").onclick = () => {{ overviewMode = "path"; renderOverview(); }};
  drawCores(document.getElementById("overviewImg"), src);
}}
function renderCore() {{
  const core = DATA.cores[selected];
  document.getElementById("content").innerHTML = `<div class="image-wrap"><img id="coreImg" src="data:image/jpeg;base64,${{core.img}}" width="${{core.width}}" height="${{core.height}}"><canvas class="overlay"></canvas></div>`;
  canvasFor(document.getElementById("coreImg"), ctx => {{
    ctx.fillStyle = "rgba(255, 230, 0, .78)";
    core.cells.forEach(cell => {{ const r = Math.max(1.8, Math.min(5, Math.sqrt(cell.area) / 4)); ctx.beginPath(); ctx.arc(cell.x, cell.y, r, 0, Math.PI * 2); ctx.fill(); }});
  }});
}}
function renderReference() {{
  const tma = DATA.cores[selected], ref = DATA.pathology_reference.cores[selected];
  document.getElementById("content").innerHTML = `<div class="split"><div><p class="panel-title">TMA2 segmentation core</p><div class="image-wrap"><img id="refTma" src="data:image/jpeg;base64,${{tma.img}}" width="${{tma.width}}" height="${{tma.height}}"><canvas class="overlay"></canvas></div></div><div><p class="panel-title">1-path stained reference core</p><div class="image-wrap"><img src="data:image/jpeg;base64,${{ref.img}}" width="${{ref.width}}" height="${{ref.height}}"></div></div></div>`;
  canvasFor(document.getElementById("refTma"), ctx => {{ ctx.fillStyle = "rgba(255,230,0,.7)"; tma.cells.forEach(cell => {{ const r=Math.max(1.8,Math.min(5,Math.sqrt(cell.area)/4)); ctx.beginPath(); ctx.arc(cell.x,cell.y,r,0,Math.PI*2); ctx.fill(); }}); }});
}}
function renderPatches() {{
  const group = DATA.patches[selected], patches = group.items, nCols = group.n_cols, nRows = group.n_rows;
  const byPos = new Map(patches.map((p, i) => [`${{p.row}}:${{p.col}}`, {{p, i}}]));
  const cells = [];
  for (let row=0; row<nRows; row++) for (let col=0; col<nCols; col++) {{
    const hit = byPos.get(`${{row}}:${{col}}`);
    cells.push(hit ? `<div class="patch-card" data-index="${{hit.i}}" style="grid-row:${{row+1}};grid-column:${{col+1}}"><div class="image-wrap"><img id="patch-${{hit.i}}" src="data:image/jpeg;base64,${{hit.p.img}}" width="512" height="512"><canvas class="overlay"></canvas></div><div class="patch-info"><span>row ${{hit.p.row}}, col ${{hit.p.col}}</span><strong>${{hit.p.n_cells}} nuclei</strong></div></div>` : `<div class="patch-card empty" style="grid-row:${{row+1}};grid-column:${{col+1}}"></div>`);
  }}
  document.getElementById("content").innerHTML = `<div class="patch-grid" style="grid-template-columns:repeat(${{nCols}}, minmax(180px,220px));">${{cells.join("")}}</div>`;
  patches.forEach((patch, i) => drawPatchOn(document.getElementById(`patch-${{i}}`), patch));
  document.querySelectorAll(".patch-card:not(.empty)").forEach(card => card.onclick = () => showPatch(patches[Number(card.dataset.index)]));
}}
function drawPatchOn(img, patch) {{
  canvasFor(img, ctx => {{ ctx.strokeStyle = "rgba(255,230,0,.96)"; ctx.lineWidth = 1.25; patch.polys.forEach(poly => {{ if (!poly.pts.length) return; ctx.beginPath(); ctx.moveTo(poly.pts[0][0], poly.pts[0][1]); poly.pts.slice(1).forEach(pt => ctx.lineTo(pt[0], pt[1])); ctx.closePath(); ctx.stroke(); }}); }});
}}
function showPatch(patch) {{
  document.getElementById("modalTitle").textContent = `Core ${{coreKeys.indexOf(selected)+1}} · row ${{patch.row}}, col ${{patch.col}} · ${{patch.n_cells}} nuclei`;
  document.getElementById("modalContent").innerHTML = `<div class="image-wrap"><img id="modalImg" src="data:image/jpeg;base64,${{patch.img}}" width="512" height="512"><canvas class="overlay"></canvas></div>`;
  document.getElementById("modal").classList.add("open"); drawPatchOn(document.getElementById("modalImg"), patch);
}}
function renderAnalysis() {{
  const maxDensity = Math.max(...DATA.analysis.summary.map(r => r.density_per_mm2));
  const maxPatch = Math.max(...DATA.analysis.patch_stats[selected].map(p => p.n_cells));
  document.getElementById("content").innerHTML = `<div class="analysis-grid"><div class="metric">Total nuclei<strong>${{DATA.analysis.overall.total_cells.toLocaleString()}}</strong></div><div class="metric">Mean nucleus area<strong>${{DATA.analysis.overall.mean_area}} px</strong></div><div class="metric">Median nucleus area<strong>${{DATA.analysis.overall.median_area}} px</strong></div></div><table><thead><tr><th>Core</th><th>Cells</th><th>Area mm2</th><th>Density / mm2</th><th>Mean area</th><th>Median</th><th>P90</th><th>Density</th></tr></thead><tbody>${{DATA.analysis.summary.map(r => `<tr><td>Core ${{r.core}}</td><td>${{r.cells.toLocaleString()}}</td><td>${{r.area_mm2}}</td><td>${{r.density_per_mm2.toLocaleString()}}</td><td>${{r.mean_area}}</td><td>${{r.median_area}}</td><td>${{r.p90_area}}</td><td><div class="bar"><span style="width:${{100*r.density_per_mm2/maxDensity}}%"></span></div></td></tr>`).join("")}}</tbody></table><p class="panel-title" style="margin-top:18px">Densest patches in selected core</p><table><thead><tr><th>Patch</th><th>Nuclei</th><th>Relative density</th></tr></thead><tbody>${{DATA.analysis.patch_stats[selected].slice(0,12).map(p => `<tr><td>row ${{p.row}}, col ${{p.col}}</td><td>${{p.n_cells}}</td><td><div class="bar"><span style="width:${{100*p.n_cells/maxPatch}}%"></span></div></td></tr>`).join("")}}</tbody></table>`;
}}
function render() {{ if(view==="overview") renderOverview(); if(view==="core") renderCore(); if(view==="reference") renderReference(); if(view==="patches") renderPatches(); if(view==="analysis") renderAnalysis(); }}
document.querySelectorAll(".tab").forEach(btn => btn.onclick = () => {{ view = btn.dataset.view; syncTabs(); render(); }});
document.getElementById("modalClose").onclick = () => document.getElementById("modal").classList.remove("open");
document.getElementById("modal").onclick = event => {{ if (event.target.id === "modal") event.currentTarget.classList.remove("open"); }};
document.addEventListener("keydown", event => {{ if (event.key === "Escape") document.getElementById("modal").classList.remove("open"); }});
setMeta(); renderCoreList(); render();
</script>
</body>
</html>"""
    HTML_PATH.write_text(html, encoding="utf-8")
    JSON_PATH.write_text(json.dumps(data), encoding="utf-8")


def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    data = add_analysis(data)
    write_html(data)
    print(f"Updated {HTML_PATH}")


if __name__ == "__main__":
    main()
