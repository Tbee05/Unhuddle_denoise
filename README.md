
## UNHUDDLE
**Uncovering Neighborhood Heterogeneity Using Deterministic Normalization and Local Equilibrium**

UNHUDDLE is an algorithm designed to resolve multiplexed signal in densely packed tissue regions — or "cell huddles" — in multiplex spatial proteomics, where traditional segmentation and quantification blur the phenotypic signal.

On the cell to cell borderpixels, shared signal is observed due to 1 resolution issues, 2 lateral bleed and 3 z-projection. Unhuddle knows the cell's neighbors, measures their mean intensity and reallocates the bordersignal to the rightful owner. 

By identifying stable, broadly expressed "sensor markers" and performing per-cell normalization, UNHUDDLE enables more accurate within-cell-type comparison of functional markers (e.g., checkpoint proteins), even in spatially crowded microenvironments.

## 🚀 Getting Started with UNHUDDLE

### 📥 1. Clone the Repository
via https
```bash
git clone https://github.com/tbee05/unhuddle.git
cd unhuddle
```
via ssh
```bash
git clone git@github.com:tbee05/unhuddle.git
cd unhuddle
```
### 📦 2. Set Up a Virtual Environment
Using venv:
```bash
python -m venv unhuddle
source unhuddle/bin/activate      # On Windows: unhuddle\Scripts\activate
```
or conda:
```bash
conda create -n unhuddle -y
conda activate unhuddle
conda install pip
```
### 🛠️ 3. Install UNHUDDLE in Editable Mode
```bash
pip install -e .
```
This installs unhuddle as a CLI tool available from anywhere in your terminal.

### ✅ 4. Verify Installation
```bash
unhuddle --help
```
Should print a list of CLI arguments and options.

### 🗂️ 5. Check Input Requirements

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


### 🧪 6. Run the Pipeline
```bash
unhuddle \
  --base_path /path/to/input_fovs \
  --output_base_path /path/to/output_folder \
  --markers_for_normalisation CD45 CD3 Vimentin \
  --create_nuclear_mask \
  --create_deepcell_mask \
  --max_workers 1
```
🚀 Try It Out on Included Demo Data
You can run UNHUDDLE directly on the provided Tonsil demo data:

linux:
```bash
unhuddle \
  --base_path demodata \
  --output_base_path results/unhuddle_output \
  --markers_for_normalisation CD20 CD68 CD11b CD11c CD8a CD3 CD7 CD45RA CD45RO CD15 CD163 Vimentin CD31 CD14 CD15 \
  --create_nuclear_mask \
  --max_workers 1
```
windows powershell
```powershell
unhuddle `
  --base_path demodata `
  --output_base_path results\unhuddle_output `
  --markers_for_normalisation CD20 CD68 CD11b CD11c CD8a CD3 CD7 CD45RA CD45RO CD15 CD163 Vimentin CD31 CD14 CD15 `
  --create_nuclear_mask `
  --max_workers 1
```
## 🌐 DeepCell Integration Setup (Selenium + Firefox)
UNHUDDLE can upload overlays to DeepCell.org using Selenium and Firefox — no GUI required.
This enables automated segmentation of images through the DeepCell web interface.

### ✅ 1. Install Firefox (Portable)
If Firefox isn’t available on your system:
Linux (portable install):
```bash
mkdir -p $HOME/tools
cd $HOME/tools
wget https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US -O firefox.tar.bz2
tar -xjf firefox.tar.bz2
```
Firefox will now be at:
```bash
$HOME/tools/firefox/firefox
```
✅ 2. Install GeckoDriver (No sudo)
Download:
```bash
cd $HOME/tools
wget https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-linux64.tar.gz
tar -xvzf geckodriver-*.tar.gz
chmod +x geckodriver
```
Now GeckoDriver is located at:
```bash
$HOME/tools/geckodriver
```

## ⚙️ Pipeline Overview

For each FOV (field of view) folder, the following stages are run:

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

## 📎 Notes
✅ Python Compatibility:
Requires Python 3.8 or higher

🧰 Core Dependencies:

numpy, pandas — numerical and tabular processing

scikit-image, scipy, tifffile — image and mask operations

selenium, requests — automation for DeepCell.org

tqdm — progress bar visualization

🧠 Image Input Expectations:

Marker files should follow the pattern: {marker}.ome.tiff

Data shape: H x W (2D)

Data type: float32 or float64 (from denoising like PENGUIN)

🧩 Segmentation Mask Guidelines:

Auto-detected via --mask_pattern (default: *_0.tiff)

Must be 2D (H x W) or stack format (Z x H x W, Z is squeezed)

If no mask is provided, DeepCell can generate one using --create_deepcell_mask

🔖 Label Convention:

Each cell should have a unique integer label

These labels form the link between segmentation, intensity, and feature CSVs

Optional: match {fov}_{label} for external compatibility

🚀 Parallel Processing:

Use --max_workers to process multiple FOVs concurrently

--check_output_exist skips already-processed FOVs

💡 Sensor Markers:

Markers like CD3, CD45, Vimentin used for intra-cell normalization

Must be broadly expressed and phenotype-informative

Crucial for deconvolution of functional marker signal

## 📊 Coming Soon
Graphical breakdown of reallocation logic
Support for additional normalization schemes (e.g. quantile)

UNHUDDLE will be installable via `pip install unhuddle`

## 📣 Citation & License
This tool is part of an ongoing research pipeline for high-dimensional tissue profiling originating from the Alizadeh laboratory at Stanford School of Medicine.
Please cite appropriately once a manuscript is available. Open-source license to be defined.


