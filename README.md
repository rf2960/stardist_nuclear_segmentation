# StarDist Nuclear Segmentation

Scale-aware StarDist nuclear segmentation and interactive analysis for TMA whole-slide images.

## Final Analysis

The final interactive HTML analysis was generated from:

`1-pathology core stained.svs`

This file is the highest-resolution slide in the local set:

- `40x`
- `0.2522 um/px`
- `25896 x 25028 px`

For StarDist, 40x patches were scale-normalized to approximately the model-friendly 20x resolution (`0.5027 um/px`) before inference. The HTML viewer still uses high-resolution 40x crops for inspection.

## Compared Slides

| Slide | Role |
| --- | --- |
| `TJ Cre Myc TMA1.svs` | Original 20x slide |
| `TJ Cre Myc TMA2.svs` | 20x slide used for the earlier clear TMA2 viewer |
| `1-pathology core stained.svs` | Final 40x slide used for the richer analysis HTML |

`TJ Cre Myc TMA1 2.svs` was removed locally because it was byte-identical to `TJ Cre Myc TMA1.svs`.

## Repository Layout

```text
scripts/
  run_1path_analysis_pipeline.py   # final pipeline used for the 40x 1-path slide
  run_tma2_stardist_clear.py       # earlier TMA2 pipeline
  enhance_tma2_html_analysis.py    # interim TMA2 HTML enhancer
results_1path_analysis/
  analysis_summary_1path.csv       # lightweight per-core analysis summary
  core_counts_1path.csv            # lightweight counts by core
  slide_comparison.csv             # slide metadata and sharpness comparison
```

Large raw slides and generated self-contained HTML/JSON outputs are intentionally ignored by Git because they are too large for normal GitHub storage. Keep those files locally, or use Git LFS / external storage if they need to be shared.

## Final Output Files Produced Locally

These are not committed because of size, but they are the main local artifacts:

- `results_1path_analysis/1path_stardist_analysis.html`
- `results_1path_analysis/analysis_data_1path.json`
- `results_1path_analysis/detected_cells_1path.csv`

## HTML Features

- Overview with fitted core circles
- Core-level high-resolution images with nuclei overlay
- Spatial patch grid preserving the tissue layout
- Clickable high-resolution patch popups
- Per-core counts, density, area statistics, and densest-patch ranking
- Slide comparison table for TMA1, TMA2, and 1-path stained

## Reproducing

Create/use an environment with:

- Python
- OpenSlide
- StarDist
- TensorFlow
- csbdeep
- NumPy
- pandas
- OpenCV
- Pillow
- SciPy

Then run:

```powershell
python scripts/run_1path_analysis_pipeline.py
```

The script expects the raw slides to live in the project root with the filenames listed above.
