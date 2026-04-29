import base64
import csv
import hashlib
import json
import math
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
SLIDE_PATH = ROOT / "1-pathology core stained.svs"
OUT_DIR = ROOT / "results_1path_analysis"
ARCHIVE_DIR = ROOT / "local_archive_ignored" / "superseded_pre_1path"

REFERENCE_SLIDES = [
    [ROOT / "TJ Cre Myc TMA1.svs", ARCHIVE_DIR / "TJ Cre Myc TMA1.svs"],
    [ROOT / "TJ Cre Myc TMA2.svs", ARCHIVE_DIR / "TJ Cre Myc TMA2.svs"],
    [ROOT / "1-pathology core stained.svs"],
]

PATCH_SIZE = 512
STEP = 256
GRID_STEP = 512
TARGET_MPP = 0.5027
PROB_THRESH = {
    1: 0.08,
    2: 0.08,
    3: 0.08,
    4: 0.10,
    5: 0.10,
    6: 0.10,
    7: 0.08,
}
CORE_COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#a855f7", "#111827"]
BOUNDARY_MARGIN_MODEL = {
    # Invisible, stain-gated boundary margin. The overview circle stays tight;
    # these margins only allow real tissue/cell signal just outside the circle.
    2: 96,
    6: 112,
    7: 160,
}
MIN_PATCH_TISSUE_FRACTION = 0.010
MIN_PATCH_DARK_FRACTION = 0.0015
CELL_AREA_MIN = 20
CELL_AREA_MAX = 900


def encode_jpeg(image, quality=92, max_size=None):
    if max_size:
        image = image.copy()
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), image.width, image.height


def enhance_patch(patch):
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def tissue_signal_masks(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    tissue = ((gray < 235) | ((sat > 18) & (gray < 248))).astype("uint8")
    dark = ((gray < 178) & (sat > 8)).astype("uint8")
    return tissue, dark, gray, sat


def clean_tissue_mask(rgb, close_size=7, open_size=3, min_size=200):
    tissue, _dark, _gray, _sat = tissue_signal_masks(rgb)
    mask = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.zeros_like(mask, dtype=bool)
    for idx, size in enumerate(sizes, start=1):
        if size >= min_size:
            keep |= labeled == idx
    return keep


def slide_quality(slide, path):
    level = slide.level_count - 1
    img = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tissue = img.mean(axis=2) < 235
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    ten = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) ** 2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) ** 2
    return {
        "file": path.name,
        "sha256_prefix": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "width": slide.level_dimensions[0][0],
        "height": slide.level_dimensions[0][1],
        "levels": slide.level_count,
        "mpp": slide.properties.get("aperio.MPP") or slide.properties.get("openslide.mpp-x"),
        "objective_power": slide.properties.get("aperio.AppMag") or slide.properties.get("openslide.objective-power"),
        "image_id": slide.properties.get("aperio.ImageID"),
        "aperio_filename": slide.properties.get("aperio.Filename"),
        "tissue_fraction_lowest": round(float(tissue.mean()), 4),
        "laplacian_var_tissue_lowest": round(float(lap[tissue].var()), 2) if tissue.any() else 0,
        "tenengrad_mean_tissue_lowest": round(float(ten[tissue].mean()), 2) if tissue.any() else 0,
    }


def write_slide_comparison():
    rows = []
    for candidates in REFERENCE_SLIDES:
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            continue
        if not path.exists():
            continue
        slide = openslide.OpenSlide(str(path))
        rows.append(slide_quality(slide, path))
    with (OUT_DIR / "slide_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def ordered_cores(components):
    components = sorted(components, key=lambda item: item["area"], reverse=True)[:7]
    components = sorted(components, key=lambda item: item["y_thumb"])
    rows = [components[:3], components[3:5], components[5:]]
    rows[0] = sorted(rows[0], key=lambda item: item["x_thumb"])
    rows[1] = sorted(rows[1], key=lambda item: item["x_thumb"])
    # Preserve the established viewer convention: lower-right is Core 6 and
    # the large lower-left tissue core is Core 7.
    rows[2] = sorted(rows[2], key=lambda item: item["x_thumb"], reverse=True)
    return rows[0] + rows[1] + rows[2]


def detect_display_cores(slide):
    thumbnail_img = slide.get_thumbnail((500, 500))
    thumb = np.array(thumbnail_img.convert("RGB"))
    scale_x = slide.level_dimensions[0][0] / thumb.shape[1]
    scale_y = slide.level_dimensions[0][1] / thumb.shape[0]
    gray = np.mean(thumb, axis=2)
    mask = (gray < 230).astype("uint8")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    components = []
    for idx, size in enumerate(sizes, start=1):
        if size <= 120:
            continue
        core_mask = labeled == idx
        rows = np.where(core_mask.any(axis=1))[0]
        cols = np.where(core_mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        darkness = (255 - gray) * core_mask
        if darkness.sum() > 0:
            yy, xx = np.indices(gray.shape)
            weighted_x = float((xx * darkness).sum() / darkness.sum())
            weighted_y = float((yy * darkness).sum() / darkness.sum())
        else:
            weighted_x = float((cols[0] + cols[-1]) / 2)
            weighted_y = float((rows[0] + rows[-1]) / 2)
        box_x = float((cols[0] + cols[-1]) / 2)
        box_y = float((rows[0] + rows[-1]) / 2)
        x = 0.65 * weighted_x + 0.35 * box_x
        y = 0.65 * weighted_y + 0.35 * box_y
        radius = max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1) * 0.52
        components.append({
            "x_thumb": x,
            "y_thumb": y,
            "radius_thumb": float(radius),
            "area": int(core_mask.sum()),
        })

    components = ordered_cores(components)
    if len(components) != 7:
        raise RuntimeError(f"Expected 7 cores, detected {len(components)}")

    for i, core in enumerate(components, start=1):
        core["core"] = i
        core["x_full"] = core["x_thumb"] * scale_x
        core["y_full"] = core["y_thumb"] * scale_y
        core["radius_full"] = core["radius_thumb"] * scale_x
    return thumb, scale_x, scale_y, components


def add_inclusion_masks(slide, cores):
    level = 2
    overview = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
    lv2 = np.array(overview)
    scale_x = slide.level_dimensions[0][0] / lv2.shape[1]
    scale_y = slide.level_dimensions[0][1] / lv2.shape[0]
    mask = clean_tissue_mask(lv2, close_size=9, open_size=3, min_size=350)
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    lv2_components = []
    for idx, size in enumerate(sizes, start=1):
        if size < 900:
            continue
        ys, xs = np.where(labeled == idx)
        lv2_components.append({
            "idx": idx,
            "area": int(size),
            "x_full": float(xs.mean() * scale_x),
            "y_full": float(ys.mean() * scale_y),
        })

    used = set()
    for core in cores:
        nearest = None
        nearest_dist = float("inf")
        for comp in lv2_components:
            if comp["idx"] in used:
                continue
            dist = math.hypot(comp["x_full"] - core["x_full"], comp["y_full"] - core["y_full"])
            if dist < nearest_dist:
                nearest = comp
                nearest_dist = dist
        if nearest is None:
            include_mask = np.zeros_like(mask, dtype=bool)
        else:
            used.add(nearest["idx"])
            include_mask = labeled == nearest["idx"]
            include_mask = ndimage.binary_dilation(include_mask, iterations=10)
        core["include_mask_lv2"] = include_mask
        core["lv2_scale_x"] = scale_x
        core["lv2_scale_y"] = scale_y
    return cores


def detect_cores(slide):
    thumb, scale_x, scale_y, components = detect_display_cores(slide)
    components = add_inclusion_masks(slide, components)
    return thumb, scale_x, scale_y, components


def read_model_patch(slide, x_full, y_full, read_size):
    patch = slide.read_region((int(round(x_full)), int(round(y_full))), 0, (read_size, read_size)).convert("RGB")
    patch_model = patch.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    return patch, np.array(patch_model)


def patch_signal(rgb):
    tissue, dark, _gray, _sat = tissue_signal_masks(rgb)
    return float(tissue.mean()), float(dark.mean())


def point_in_include_mask(core, x_full, y_full):
    x = int(round(x_full / core["lv2_scale_x"]))
    y = int(round(y_full / core["lv2_scale_y"]))
    mask = core["include_mask_lv2"]
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def cell_has_nuclear_signal(labels, cell_id, patch_model):
    pix = labels == cell_id
    if not pix.any():
        return False
    area = int(pix.sum())
    if area < CELL_AREA_MIN or area > CELL_AREA_MAX:
        return False
    _tissue, _dark, gray, sat = tissue_signal_masks(patch_model)
    vals = gray[pix]
    sats = sat[pix]
    # Blank background can still produce low-probability StarDist polygons,
    # especially around dust. Require real hematoxylin/stain signal inside the
    # predicted object before accepting a boundary nucleus.
    return (
        float(vals.mean()) < 218
        or float(np.percentile(vals, 15)) < 176
        or (float(sats.mean()) > 24 and float(vals.mean()) < 232)
    )


def keep_cell_for_core(core, cell_x_model, cell_y_model, patch_model, labels, cell_id, model_scale):
    core_num = core["core"]
    radius_model = core["radius_full"] / model_scale
    dist = math.hypot(cell_x_model, cell_y_model)
    if dist <= radius_model:
        return cell_has_nuclear_signal(labels, cell_id, patch_model)

    margin = BOUNDARY_MARGIN_MODEL.get(core_num, 64)
    if dist > radius_model + margin:
        return False

    x_full = core["x_full"] + cell_x_model * model_scale
    y_full = core["y_full"] + cell_y_model * model_scale
    if not point_in_include_mask(core, x_full, y_full):
        return False
    return cell_has_nuclear_signal(labels, cell_id, patch_model)


def patch_allowed_for_core(core, patch_model, model_left, model_top, model_scale):
    radius_model = core["radius_full"] / model_scale
    yy, xx = np.mgrid[0:PATCH_SIZE, 0:PATCH_SIZE]
    dist = np.hypot(xx + model_left, yy + model_top)
    circle_overlap = float((dist <= radius_model).mean())
    tissue_frac, dark_frac = patch_signal(patch_model)
    has_signal = tissue_frac >= MIN_PATCH_TISSUE_FRACTION or dark_frac >= MIN_PATCH_DARK_FRACTION
    if not has_signal:
        return False, circle_overlap, tissue_frac, dark_frac
    if circle_overlap >= 0.01:
        return True, circle_overlap, tissue_frac, dark_frac

    margin = BOUNDARY_MARGIN_MODEL.get(core["core"], 64)
    if float((dist <= radius_model + margin).mean()) < 0.01:
        return False, circle_overlap, tissue_frac, dark_frac

    sample_y, sample_x = np.where(tissue_signal_masks(patch_model)[0] > 0)
    if len(sample_x) == 0:
        return False, circle_overlap, tissue_frac, dark_frac
    stride = max(1, len(sample_x) // 1200)
    xs_full = core["x_full"] + (sample_x[::stride] + model_left) * model_scale
    ys_full = core["y_full"] + (sample_y[::stride] + model_top) * model_scale
    hits = sum(point_in_include_mask(core, x, y) for x, y in zip(xs_full, ys_full))
    return hits > 0, circle_overlap, tissue_frac, dark_frac


def segment_slide(slide, model, cores, source_mpp):
    model_scale = TARGET_MPP / source_mpp
    read_size = int(round(PATCH_SIZE * model_scale))
    all_cells = []
    patch_data = {}

    for core in cores:
        core_num = core["core"]
        print(f"Processing 1-path core {core_num}/7", flush=True)
        cx = core["x_full"]
        cy = core["y_full"]
        radius_model = core["radius_full"] / model_scale
        margin = BOUNDARY_MARGIN_MODEL.get(core_num, 64)
        core_size_model = max(int((radius_model + margin) * 2 + PATCH_SIZE), 1024)
        prob_thresh = PROB_THRESH[core_num]
        raw_count = 0

        for row in range(0, core_size_model, STEP):
            for col in range(0, core_size_model, STEP):
                model_left = -core_size_model / 2 + col
                model_top = -core_size_model / 2 + row
                patch_center_x = model_left + PATCH_SIZE / 2
                patch_center_y = model_top + PATCH_SIZE / 2
                if math.hypot(patch_center_x, patch_center_y) > radius_model + margin + PATCH_SIZE * 0.7:
                    continue

                x_full = cx + model_left * model_scale
                y_full = cy + model_top * model_scale
                _patch_hi, patch = read_model_patch(slide, x_full, y_full, read_size)
                allowed, _circle_overlap, _tissue_frac, _dark_frac = patch_allowed_for_core(
                    core, patch, model_left, model_top, model_scale
                )
                if not allowed:
                    continue

                labels, _ = model.predict_instances(
                    normalize(enhance_patch(patch), 1, 99.8),
                    prob_thresh=prob_thresh,
                    nms_thresh=0.3,
                )
                if labels.max() == 0:
                    continue

                for cell_id in range(1, labels.max() + 1):
                    pix = np.where(labels == cell_id)
                    cell_x_model = float(np.mean(pix[1]) + model_left)
                    cell_y_model = float(np.mean(pix[0]) + model_top)
                    if not keep_cell_for_core(core, cell_x_model, cell_y_model, patch, labels, cell_id, model_scale):
                        continue
                    all_cells.append({
                        "core": core_num,
                        "x_full": cx + cell_x_model * model_scale,
                        "y_full": cy + cell_y_model * model_scale,
                        "x_model": cell_x_model,
                        "y_model": cell_y_model,
                        "area_model_px": int(len(pix[0])),
                    })
                    raw_count += 1
        print(f"  raw detections: {raw_count}", flush=True)

    coords = np.array([[c["x_model"] + cores[c["core"] - 1]["core"] * 100000, c["y_model"]] for c in all_cells])
    areas = np.array([c["area_model_px"] for c in all_cells])
    to_remove = set()
    if len(coords):
        for i, j in cKDTree(coords).query_pairs(r=8.0):
            to_remove.add(j if areas[i] >= areas[j] else i)
    deduped = [c for i, c in enumerate(all_cells) if i not in to_remove]
    df = pd.DataFrame(deduped)
    df = df[(df["area_model_px"] >= CELL_AREA_MIN) & (df["area_model_px"] <= CELL_AREA_MAX)].reset_index(drop=True)
    df.insert(1, "cell_id", df.groupby("core").cumcount() + 1)
    print(f"Final 1-path nuclei: {len(df)}", flush=True)
    return df, model_scale, read_size


def export_payload(slide, model, df, cores, source_mpp, model_scale, read_size, slide_comparison):
    overview = slide.read_region((0, 0), 2, slide.level_dimensions[2]).convert("RGB")
    overview_b64, overview_w, overview_h = encode_jpeg(overview, 90)
    scale_lv2_x = slide.level_dimensions[0][0] / slide.level_dimensions[2][0]
    scale_lv2_y = slide.level_dimensions[0][1] / slide.level_dimensions[2][1]

    payload = {
        "slide": {
            "filename": SLIDE_PATH.name,
            "dimensions": list(slide.level_dimensions[0]),
            "mpp": source_mpp,
            "appmag": slide.properties.get("aperio.AppMag") or slide.properties.get("openslide.objective-power"),
            "target_mpp_for_model": TARGET_MPP,
            "model_scale_source_px_per_model_px": round(model_scale, 4),
        },
        "slide_comparison": slide_comparison,
        "overview": {"img": overview_b64, "width": overview_w, "height": overview_h},
        "cores": {},
        "patches": {},
    }

    for core in cores:
        core_num = core["core"]
        key = f"core{core_num}"
        core_df = df[df["core"] == core_num]
        color = CORE_COLORS[core_num - 1]
        lv1_scale = float(slide.level_downsamples[1])
        full_r = core["radius_full"]
        margin_full = BOUNDARY_MARGIN_MODEL.get(core_num, 64) * model_scale
        r1 = int((full_r + margin_full) / lv1_scale)
        size1 = r1 * 2 + 40
        origin_x = int(core["x_full"] - (size1 * lv1_scale) / 2)
        origin_y = int(core["y_full"] - (size1 * lv1_scale) / 2)
        region = slide.read_region((origin_x, origin_y), 1, (size1, size1)).convert("RGB")
        region_b64, region_w, region_h = encode_jpeg(region, 92, max_size=1200)
        resize_factor = region_w / size1

        cells = []
        for row in core_df.itertuples(index=False):
            cells.append({
                "x": round(((row.x_full - origin_x) / lv1_scale) * resize_factor, 1),
                "y": round(((row.y_full - origin_y) / lv1_scale) * resize_factor, 1),
                "area": int(row.area_model_px),
            })

        payload["cores"][key] = {
            "img": region_b64,
            "width": region_w,
            "height": region_h,
            "cells": cells,
            "total_cells": int(len(core_df)),
            "color": color,
            "cx_overview": round(core["x_full"] / scale_lv2_x, 1),
            "cy_overview": round(core["y_full"] / scale_lv2_y, 1),
            "r_overview": round(full_r / scale_lv2_x, 1),
            "area_mm2": round(math.pi * (full_r * source_mpp / 1000) ** 2, 4),
        }

        patches = []
        radius_model = full_r / model_scale
        margin = BOUNDARY_MARGIN_MODEL.get(core_num, 64)
        core_size_model = max(int((radius_model + margin) * 2 + PATCH_SIZE), 1024)
        for row_i, row in enumerate(range(0, core_size_model, GRID_STEP)):
            for col_i, col in enumerate(range(0, core_size_model, GRID_STEP)):
                model_left = -core_size_model / 2 + col
                model_top = -core_size_model / 2 + row
                x_full = core["x_full"] + model_left * model_scale
                y_full = core["y_full"] + model_top * model_scale
                patch_hi, patch_model = read_model_patch(slide, x_full, y_full, read_size)
                allowed, circle_overlap, tissue_frac, dark_frac = patch_allowed_for_core(
                    core, patch_model, model_left, model_top, model_scale
                )
                if not allowed:
                    continue
                polys_model = []
                polys_hi = []
                labels, details = model.predict_instances(
                    normalize(enhance_patch(patch_model), 1, 99.8),
                    prob_thresh=PROB_THRESH[core_num],
                    nms_thresh=0.3,
                )
                for cell_id, coords in enumerate(details["coord"], start=1):
                    pts = np.array(coords).T
                    center_y = float(np.mean(pts[:, 0]) + model_top)
                    center_x = float(np.mean(pts[:, 1]) + model_left)
                    if not keep_cell_for_core(core, center_x, center_y, patch_model, labels, cell_id, model_scale):
                        continue
                    pts_model = [[round(float(p[1]), 1), round(float(p[0]), 1)] for p in pts]
                    pts_hi = [[round(float(p[1] * model_scale), 1), round(float(p[0] * model_scale), 1)] for p in pts]
                    area = int((labels == cell_id).sum())
                    polys_model.append({"pts": pts_model, "area": area})
                    polys_hi.append({"pts": pts_hi, "area": area})

                thumb = Image.fromarray(patch_model).resize((256, 256), Image.Resampling.LANCZOS)
                thumb_b64, _, _ = encode_jpeg(thumb, 86)
                hi_b64, hi_w, hi_h = encode_jpeg(patch_hi, 90, max_size=1100)
                scale_hi = hi_w / read_size
                if abs(scale_hi - 1) > 1e-6:
                    for poly in polys_hi:
                        poly["pts"] = [[round(x * scale_hi, 1), round(y * scale_hi, 1)] for x, y in poly["pts"]]

                patches.append({
                    "row": row_i,
                    "col": col_i,
                    "img": thumb_b64,
                    "img_hi": hi_b64,
                    "hi_width": hi_w,
                    "hi_height": hi_h,
                    "polys_thumb": [
                        {"pts": [[round(x / 2, 1), round(y / 2, 1)] for x, y in poly["pts"]], "area": poly["area"]}
                        for poly in polys_model
                    ],
                    "polys_model": polys_model,
                    "polys_hi": polys_hi,
                    "n_cells": len(polys_model),
                    "circle_overlap": round(float(circle_overlap), 3),
                    "tissue_fraction": round(float(tissue_frac), 4),
                    "dark_fraction": round(float(dark_frac), 4),
                })

        payload["patches"][key] = {
            "items": patches,
            "n_rows": max((p["row"] for p in patches), default=-1) + 1,
            "n_cols": max((p["col"] for p in patches), default=-1) + 1,
        }
        print(f"Exported core {core_num}: {len(core_df)} cells, {len(patches)} patches", flush=True)

    add_analysis(payload)
    return payload


def add_analysis(payload):
    summary = []
    all_areas = []
    for i, key in enumerate(payload["cores"], start=1):
        core = payload["cores"][key]
        areas = np.array([cell["area"] for cell in core["cells"]], dtype=float)
        all_areas.extend(areas.tolist())
        density = core["total_cells"] / core["area_mm2"] if core["area_mm2"] else 0
        summary.append({
            "core": i,
            "cells": int(core["total_cells"]),
            "area_mm2": core["area_mm2"],
            "density_per_mm2": round(density, 1),
            "mean_area": round(float(np.mean(areas)), 1) if len(areas) else 0,
            "median_area": round(float(np.median(areas)), 1) if len(areas) else 0,
            "p90_area": round(float(np.percentile(areas, 90)), 1) if len(areas) else 0,
        })
    patch_stats = {}
    for key, group in payload["patches"].items():
        patch_stats[key] = sorted(
            [{"row": p["row"], "col": p["col"], "n_cells": p["n_cells"]} for p in group["items"]],
            key=lambda item: item["n_cells"],
            reverse=True,
        )
    payload["analysis"] = {
        "summary": summary,
        "overall": {
            "total_cells": int(sum(r["cells"] for r in summary)),
            "mean_area": round(float(np.mean(all_areas)), 1) if all_areas else 0,
            "median_area": round(float(np.median(all_areas)), 1) if all_areas else 0,
        },
        "patch_stats": patch_stats,
    }


def write_summary_tables(payload):
    summary_df = pd.DataFrame(payload["analysis"]["summary"])
    summary_df.to_csv(OUT_DIR / "analysis_summary_1path.csv", index=False)
    summary_df[["core", "cells"]].to_csv(OUT_DIR / "core_counts_1path.csv", index=False)


def write_html(payload, path):
    payload_json = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>1-path StarDist Nuclear Viewer</title>
<style>
:root {{ color-scheme:light; --ink:#172033; --muted:#667085; --line:#d8dee9; --bg:#f5f7fa; --accent:#0f766e; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }}
header {{ padding:18px 22px 12px; background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }}
h1 {{ margin:0; font-size:20px; }}
.meta {{ margin-top:6px; display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:13px; }}
main {{ display:grid; grid-template-columns:320px minmax(0,1fr); min-height:calc(100vh - 70px); }}
aside {{ padding:16px; border-right:1px solid var(--line); background:#fbfcfe; overflow:auto; }}
.stage {{ padding:16px; overflow:auto; }}
.hint {{ margin:0 0 12px; color:var(--muted); font-size:13px; }}
.core-list {{ display:grid; gap:8px; }}
button {{ font:inherit; }}
.core-btn {{ width:100%; display:flex; align-items:center; justify-content:space-between; gap:10px; border:1px solid var(--line); background:#fff; padding:10px 12px; border-radius:7px; cursor:pointer; }}
.core-btn.active {{ border-color:var(--accent); box-shadow:0 0 0 2px rgba(15,118,110,.12); }}
.dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; margin-right:8px; border:1px solid rgba(0,0,0,.25); }}
.tabs,.tools {{ display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; }}
.tab,.tool {{ border:1px solid var(--line); background:#fff; border-radius:7px; padding:8px 12px; cursor:pointer; }}
.tab.active,.tool.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.viewer {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px; }}
.image-wrap {{ position:relative; width:max-content; max-width:100%; margin:auto; }}
.image-wrap img {{ display:block; max-width:100%; height:auto; }}
canvas.overlay {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
.patch-grid {{ display:grid; gap:10px; align-items:start; justify-content:start; overflow:auto; padding-bottom:8px; }}
.patch-card {{ background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; cursor:zoom-in; }}
.patch-card.empty {{ visibility:hidden; }}
.patch-card img {{ width:100%; display:block; }}
.patch-info {{ padding:8px 10px; font-size:13px; color:var(--muted); display:flex; justify-content:space-between; }}
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
.modal-inner {{ background:#fff; border-radius:8px; max-width:min(94vw, 1200px); max-height:94vh; overflow:auto; padding:12px; box-shadow:0 20px 80px rgba(0,0,0,.35); }}
.modal-bar {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:8px; color:var(--muted); }}
.modal-close {{ border:1px solid var(--line); background:#fff; border-radius:7px; padding:6px 10px; cursor:pointer; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} aside {{ border-right:0; border-bottom:1px solid var(--line); }} }}
</style>
</head>
<body>
<header>
  <h1>1-path StarDist Nuclear Viewer</h1>
  <div class="meta"><span id="slideName"></span><span id="slideDims"></span><span id="slideMpp"></span><span id="scaleInfo"></span><span id="totalCells"></span></div>
</header>
<main>
  <aside><p class="hint">Segmentation was run from the 40x slide after scale-normalizing patches to ~20x for StarDist. Patch cards keep spatial order and open high-resolution 40x crops.</p><div class="core-list" id="coreList"></div></aside>
  <section class="stage"><div class="tabs"><button class="tab active" data-view="overview">Overview</button><button class="tab" data-view="core">Core detail</button><button class="tab" data-view="patches">Patch grid</button></div><div class="viewer" id="content"></div></section>
</main>
<div class="modal" id="modal"><div class="modal-inner"><div class="modal-bar"><span id="modalTitle"></span><button class="modal-close" id="modalClose">Close</button></div><div id="modalContent"></div></div></div>
<script>
const DATA = {payload_json};
let selected = "core1";
let view = "overview";
let minCells = 0;
let showSegmentation = true;
const coreKeys = Object.keys(DATA.cores);
function setMeta() {{
 document.getElementById("slideName").textContent = DATA.slide.filename;
 document.getElementById("slideDims").textContent = `${{DATA.slide.dimensions[0]}} x ${{DATA.slide.dimensions[1]}} px`;
 document.getElementById("slideMpp").textContent = `${{DATA.slide.appmag}}x, ${{DATA.slide.mpp}} um/px`;
 document.getElementById("scaleInfo").textContent = `model scale: ${{DATA.slide.target_mpp_for_model}} um/px`;
 document.getElementById("totalCells").textContent = `${{coreKeys.reduce((sum,key)=>sum+DATA.cores[key].total_cells,0).toLocaleString()}} nuclei`;
}}
function renderCoreList() {{
 const list=document.getElementById("coreList"); list.innerHTML="";
 coreKeys.forEach((key,i)=>{{ const c=DATA.cores[key]; const b=document.createElement("button"); b.className="core-btn"+(key===selected?" active":""); b.innerHTML=`<span><span class="dot" style="background:${{c.color}}"></span>Core ${{i+1}}</span><strong>${{c.total_cells.toLocaleString()}}</strong>`; b.onclick=()=>{{selected=key;if(view==="overview")view="core";syncTabs();renderCoreList();render();}}; list.appendChild(b); }});
}}
function syncTabs() {{ document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("active",b.dataset.view===view)); }}
function canvasFor(img, draw) {{ const canvas=img.closest(".image-wrap").querySelector("canvas"); const paint=()=>{{canvas.width=img.naturalWidth; canvas.height=img.naturalHeight; draw(canvas.getContext("2d"));}}; if(img.complete) paint(); else img.onload=paint; }}
function drawOverview(img) {{ canvasFor(img,ctx=>{{ctx.lineWidth=3; ctx.font="bold 18px Arial"; ctx.textAlign="center"; coreKeys.forEach((key,i)=>{{const c=DATA.cores[key]; ctx.strokeStyle=c.color; ctx.fillStyle=c.color; ctx.beginPath(); ctx.arc(c.cx_overview,c.cy_overview,c.r_overview,0,Math.PI*2); ctx.stroke(); ctx.fillText(`Core ${{i+1}}`,c.cx_overview,Math.max(22,c.cy_overview-c.r_overview-8));}});}}); }}
function renderOverview() {{ const ov=DATA.overview; document.getElementById("content").innerHTML=`<div class="image-wrap"><img id="overviewImg" src="data:image/jpeg;base64,${{ov.img}}" width="${{ov.width}}" height="${{ov.height}}"><canvas class="overlay"></canvas></div>`; drawOverview(document.getElementById("overviewImg")); }}
function renderCore() {{ const c=DATA.cores[selected]; document.getElementById("content").innerHTML=`<div class="image-wrap"><img id="coreImg" src="data:image/jpeg;base64,${{c.img}}" width="${{c.width}}" height="${{c.height}}"><canvas class="overlay"></canvas></div>`; canvasFor(document.getElementById("coreImg"),ctx=>{{ctx.fillStyle="rgba(255,230,0,.72)"; c.cells.forEach(cell=>{{const r=Math.max(1.8,Math.min(5,Math.sqrt(cell.area)/4)); ctx.beginPath(); ctx.arc(cell.x,cell.y,r,0,Math.PI*2); ctx.fill();}});}}); }}
function drawPatch(img, patch, high=false) {{ const polys=high?patch.polys_hi:patch.polys_thumb; canvasFor(img,ctx=>{{if(!showSegmentation)return; ctx.strokeStyle="rgba(255,230,0,.96)"; ctx.lineWidth=high?1.8:1.1; polys.forEach(poly=>{{if(!poly.pts.length)return; ctx.beginPath(); ctx.moveTo(poly.pts[0][0],poly.pts[0][1]); poly.pts.slice(1).forEach(pt=>ctx.lineTo(pt[0],pt[1])); ctx.closePath(); ctx.stroke();}});}}); }}
function renderPatches() {{ const group=DATA.patches[selected], patches=group.items, nCols=group.n_cols, nRows=group.n_rows; const byPos=new Map(patches.map((p,i)=>[`${{p.row}}:${{p.col}}`,{{p,i}}])); let cells=[]; for(let r=0;r<nRows;r++)for(let c=0;c<nCols;c++){{const hit=byPos.get(`${{r}}:${{c}}`); if(hit&&hit.p.n_cells>=minCells)cells.push(`<div class="patch-card" data-index="${{hit.i}}" style="grid-row:${{r+1}};grid-column:${{c+1}}"><div class="image-wrap"><img id="patch-${{hit.i}}" src="data:image/jpeg;base64,${{hit.p.img}}" width="256" height="256"><canvas class="overlay"></canvas></div><div class="patch-info"><span>row ${{hit.p.row}}, col ${{hit.p.col}}</span><strong>${{hit.p.n_cells}} nuclei</strong></div></div>`); else cells.push(`<div class="patch-card empty" style="grid-row:${{r+1}};grid-column:${{c+1}}"></div>`);}} document.getElementById("content").innerHTML=`<div class="tools"><label>Min nuclei <input id="minCells" type="range" min="0" max="250" value="${{minCells}}" step="10"></label><span>${{minCells}}</span><label><input id="segToggle" type="checkbox" ${{showSegmentation?"checked":""}}> Segmentation overlay</label></div><div class="patch-grid" style="grid-template-columns:repeat(${{nCols}}, minmax(160px,210px));">${{cells.join("")}}</div>`; document.getElementById("minCells").oninput=e=>{{minCells=Number(e.target.value);renderPatches();}}; document.getElementById("segToggle").onchange=e=>{{showSegmentation=e.target.checked;renderPatches();}}; patches.forEach((p,i)=>{{const img=document.getElementById(`patch-${{i}}`); if(img) drawPatch(img,p,false);}}); document.querySelectorAll(".patch-card:not(.empty)").forEach(card=>card.onclick=()=>showPatch(patches[Number(card.dataset.index)])); }}
function showPatch(p) {{ document.getElementById("modalTitle").textContent=`Core ${{coreKeys.indexOf(selected)+1}} · row ${{p.row}}, col ${{p.col}} · ${{p.n_cells}} nuclei`; document.getElementById("modalContent").innerHTML=`<div class="image-wrap"><img id="modalImg" src="data:image/jpeg;base64,${{p.img_hi}}" width="${{p.hi_width}}" height="${{p.hi_height}}"><canvas class="overlay"></canvas></div>`; document.getElementById("modal").classList.add("open"); drawPatch(document.getElementById("modalImg"),p,true); }}
function renderAnalysis() {{ const maxD=Math.max(...DATA.analysis.summary.map(r=>r.density_per_mm2)); const maxP=Math.max(...DATA.analysis.patch_stats[selected].map(p=>p.n_cells)); document.getElementById("content").innerHTML=`<div class="analysis-grid"><div class="metric">Total nuclei<strong>${{DATA.analysis.overall.total_cells.toLocaleString()}}</strong></div><div class="metric">Mean area<strong>${{DATA.analysis.overall.mean_area}} px</strong></div><div class="metric">Median area<strong>${{DATA.analysis.overall.median_area}} px</strong></div></div><table><thead><tr><th>Core</th><th>Cells</th><th>Area mm2</th><th>Density/mm2</th><th>Mean</th><th>Median</th><th>P90</th><th>Density</th></tr></thead><tbody>${{DATA.analysis.summary.map(r=>`<tr><td>Core ${{r.core}}</td><td>${{r.cells.toLocaleString()}}</td><td>${{r.area_mm2}}</td><td>${{r.density_per_mm2.toLocaleString()}}</td><td>${{r.mean_area}}</td><td>${{r.median_area}}</td><td>${{r.p90_area}}</td><td><div class="bar"><span style="width:${{100*r.density_per_mm2/maxD}}%"></span></div></td></tr>`).join("")}}</tbody></table><h3>Densest patches in selected core</h3><table><thead><tr><th>Patch</th><th>Nuclei</th><th>Relative</th></tr></thead><tbody>${{DATA.analysis.patch_stats[selected].slice(0,15).map(p=>`<tr><td>row ${{p.row}}, col ${{p.col}}</td><td>${{p.n_cells}}</td><td><div class="bar"><span style="width:${{100*p.n_cells/maxP}}%"></span></div></td></tr>`).join("")}}</tbody></table>`; }}
function renderCompare() {{ document.getElementById("content").innerHTML=`<table><thead><tr><th>File</th><th>Power</th><th>MPP</th><th>Dimensions</th><th>Size MB</th><th>Laplacian</th><th>Tenengrad</th></tr></thead><tbody>${{DATA.slide_comparison.map(s=>`<tr><td>${{s.file}}</td><td>${{s.objective_power}}x</td><td>${{s.mpp}}</td><td>${{s.width}} x ${{s.height}}</td><td>${{s.size_mb}}</td><td>${{s.laplacian_var_tissue_lowest}}</td><td>${{s.tenengrad_mean_tissue_lowest}}</td></tr>`).join("")}}</tbody></table>`; }}
function render() {{ if(view==="overview")renderOverview(); if(view==="core")renderCore(); if(view==="patches")renderPatches(); }}
document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{{view=b.dataset.view;syncTabs();render();}});
document.getElementById("modalClose").onclick=()=>document.getElementById("modal").classList.remove("open");
document.getElementById("modal").onclick=e=>{{if(e.target.id==="modal")e.currentTarget.classList.remove("open");}};
document.addEventListener("keydown",e=>{{if(e.key==="Escape")document.getElementById("modal").classList.remove("open");}});
setMeta(); renderCoreList(); render();
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slide_comparison = write_slide_comparison()
    slide = openslide.OpenSlide(str(SLIDE_PATH))
    source_mpp = float(slide.properties.get("aperio.MPP") or slide.properties.get("openslide.mpp-x"))
    _thumb, _scale_x, _scale_y, cores = detect_cores(slide)
    print("Detected 1-path cores:", [(round(c["x_thumb"], 1), round(c["y_thumb"], 1)) for c in cores], flush=True)

    model = StarDist2D.from_pretrained("2D_versatile_he")
    model_scale = TARGET_MPP / source_mpp
    read_size = int(round(PATCH_SIZE * model_scale))
    csv_path = OUT_DIR / "detected_cells_1path.csv"
    if csv_path.exists():
        print(f"Reusing existing segmentation CSV: {csv_path}", flush=True)
        df = pd.read_csv(csv_path)
    else:
        df, model_scale, read_size = segment_slide(slide, model, cores, source_mpp)
        df.to_csv(csv_path, index=False)

    payload = export_payload(slide, model, df, cores, source_mpp, model_scale, read_size, slide_comparison)
    (OUT_DIR / "analysis_data_1path.json").write_text(json.dumps(payload), encoding="utf-8")
    write_summary_tables(payload)
    write_html(payload, OUT_DIR / "1path_stardist_analysis.html")
    print(f"Saved outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
