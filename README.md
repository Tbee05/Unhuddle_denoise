
### UNHUDDLE
**Uncovering Neighborhood Heterogeneity Using Deterministic Normalization and Local Equilibrium**

UNHUDDLE is an algorithm designed to resolve multiplexed signal in densely packed tissue regions — or "cell huddles" — where traditional segmentation and quantification blur the phenotypic signal.

By identifying stable, broadly expressed "sensor markers" and performing per-cell normalization, UNHUDDLE enables accurate within-cell-type comparison of functional markers (e.g., checkpoint proteins), even in spatially crowded microenvironments.


## 🗂️ Input Requirements

The base input directory should contain one folder per FOV:

```
base_path/
├── FOV1/
│   ├── CD3.ome.tiff
│   ├── CD20.ome.tiff
│   └── ...
├── FOV2/
│   ├── CD3.ome.tiff
│   └── ...
```

Each FOV folder should contain:
- Denoised marker images (`{marker}.ome.tiff`, shape: `H x W`, dtype: `float32/64`)
- Optionally: a segmentation mask (`*_mask_0.tiff`, shape:   `H x W` or `Z x H x W`, dtype: `uint16`)  
  If a mask is not provided, one can be generated using `--create_deepcell_mask`.

---

## ⚙️ Pipeline Overview

For each FOV folder, the following stages are run:

### 1. **Mask Processing & Interaction Computation**
- Segment cells, nuclei, and membranes.
- Compute:
  - Border interactions (from membrane adjacency)
  - Background interactions (from empty space)
- Merge into a typed interaction dictionary.

### 2. **Feature Extraction**
- Extract per-cell morphological features (area, eccentricity, etc.)
- Exported to CSV.

### 3. **Protein Intensity Extraction**
- Compute marker intensities per segmented cell (excluding nuclear markers).
- Exported to CSV.

### 4. **Object-Intensity Reallocation**
- Reorganize per-pixel interactions into per-cell structure.
- Allocate intensities based on interaction-weighted mean signal.
- *(Graphical schematic to be added here 🔜)*

### 5. **Intensity Settlement**
- Merge morphology and intensity features.
- Apply corrections and output per-measure-type summaries (keeps original available):
  - `/unhuddle_sum/{fov}.csv`, `/original_sum/{fov}.csv`


### 6. **Normalized Intensity Calculation**
- Normalize per-cell values by a weighted mean of top **sensor markers** (e.g., CD45, CD3, Vimentin).
- Sensor markers are broadly expressed (typically membrane or cytoplasmic) and support robust within-cell-type comparisons of functional markers like checkpoint inhibitors.
- Values are then robustly scaled per-marker using [0.1, 99.9] percentiles.
- Output: normalized per-cell intensity matrix (values in [0, 1]).
  - `/unhuddle_normalized/{fov}.csv`, `/original_normalized/{fov}.csv`

### 7. **Optional DeepCell Mask Creation**
- Generate RGB overlays from selected markers.
- Use Selenium to upload to DeepCell.org, download results, and integrate mask into downstream analysis.

---

## 🧪 Example CLI Usage

```bash
python run_pipeline.py \
  --base_path /data/my_experiment \
  --output_base_path /results/my_experiment \
  --markers_for_normalisation CD45 CD3 Vimentin \
  --create_nuclear_mask \
  --create_deepcell_mask \
  --max_workers 4
```
```bash
🧾 Argument Reference
Argument	Description
--base_path	Path to folder containing FOV subfolders
--output_base_path	Where to write processed outputs
--max_workers	Number of parallel processes (default: 1)
--create_nuclear_mask	Generate nuclear mask (enables N/C ratio)
--create_deepcell_mask	Generate RGB overlay & segment with DeepCell
--red-markers	Markers for red channel (nuclear)
--green-markers	Markers for green channel (membrane/cytoplasm)
--blue-markers	Optional markers for blue channel
--geckodriver_path	Path to geckodriver for Selenium
--deepcell_url	URL of DeepCell website
--fovs	List of FOV folder names to process (optional)
--mask_pattern	Glob pattern(s) for mask files (eg *_mask_0.tiff)
--list_available_markers	List markers and exit
--log-level	One of: DEBUG, INFO, WARNING, ERROR
--check_output_exist	Skip FOVs if output already exists
--markers_for_normalisation	REQUIRED unless using --list_available_markers
```

##📎 Notes
Requires Python 3.8+

Dependencies: numpy, pandas, scikit-image, Selenium, and DeepCell-compatible image stack.

Segmentation masks should ideally match the {fov}_{label} convention if supplied.

##📊 Coming Soon
Visualization of sensor marker coverage and normalization effects

Graphical breakdown of reallocation logic

Support for additional normalization schemes (e.g. quantile)

##📣 Citation & License
This tool is part of an ongoing research pipeline for high-dimensional tissue profiling.
Please cite appropriately once a manuscript is available. Open-source license to be defined.


