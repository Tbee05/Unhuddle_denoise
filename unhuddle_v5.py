#!/usr/bin/env python3
"""
Final Integrated Pipeline Documentation
=======================================

Overview:
---------
This pipeline processes multiplexed imaging data across multiple Fields of View (FOVs),
extracts morphological and protein expression features, and computes spatial interaction
metrics at the single-cell level.

Input Requirements:
-------------------
The input should be structured as follows:

1. Base Directory:
   - Contains multiple folders, each corresponding to a single FOV.
   - Folder name format: {fov-name}

2. Within each FOV folder:
   - Denoised marker images, for example penguin denoiser https://github.com/deMirandaLab/PENGUIN:
     - Format: {marker}.ome.tiff
     - Shape: (H x W)
     - Dtype: float32 or float64

   - Optional: Segmentation mask:
     - Filename: any (provided via argument or convention)
     - Shape: (H x W), or (Z x H x W) — in which case the Z-axis will be squeezed
     - Dtype: uint16, uint32
     - If not provided, a segmentation mask can be auto-generated using the `--create_deepcell_mask` flag.

Command-Line Arguments Summary:
-------------------------------

Required:
  --base_path                Path to the base directory containing FOV folders.
  --output_base_path         Output directory where results will be stored.
  --markers_for_normalisation  
                            List of sensor markers used for normalization (CD45 CD3 Vimentin ...).
                            These should represent broadly expressed phenotype markers.
                            When unsure on available markers, use --list_available_markers.

Optional Flags:
  --create_nuclear_mask      If set, generates nuclear masks to enable morphology features and nuclear/cytoplasmic ratio.
  --create_deepcell_mask     If set, creates a DeepCell RGB overlay and runs web-based segmentation via Selenium.
  --mask_pattern             Wildcard pattern(s) to identify input segmentation masks (default: "*_0.tiff").

Processing Control:
  --fovs                     Specific FOV folder names to process. If omitted, all subfolders in base_path are used.
  --check_output_exist       If set, FOVs with existing outputs will be skipped.
  --max_workers              Number of FOVs to process in parallel (default: 1).

Utility:
  --list_available_markers   Print all detected marker names and exit.
  --log-level                Logging verbosity (default: WARNING). Options: DEBUG, INFO, WARNING, ERROR.

DeepCell Overlay Settings:
  --red-markers              Markers for the red channel (typically nuclear, default: DNA1, DNA2, HistoneH3).
  --green-markers            Markers for the green channel (typically membrane/cytoplasm).
  --blue-markers             Markers for the blue channel (optional, currently ignored in processing).

DeepCell Automation:
  --geckodriver_path         Path to the GeckoDriver for automating the DeepCell site (default path provided).
  --deepcell_url             URL of the DeepCell website (default: http://www.deepcell.org).


Pipeline Stages:
----------------

1. **Mask Processing and Interaction Computation**
   - Compute segmentation masks for cells, nuclei, and membranes.
   - Derive two types of pixel-level interactions:
     a. Border interactions (based on membrane pixel adjacency).
     b. Background interactions (based on isolated background pixels).
   - Merge all interactions into a structured dictionary with type annotations (`"type": "border"` or `"background"`).

2. **Feature Extraction**
   - Extract per-cell morphological features using region properties (e.g., area, eccentricity).
   - Save the resulting features as a CSV file per FOV.

3. **Protein Intensity Extraction**
   - Compute per-cell intensity values for each marker, excluding DNA/histone markers.
   - Output a CSV summarizing intensity per cell per marker.

4. **Object-Intensity Analysis**
   - Reorganize coordinate-level interaction data into a per-object (cell-level) structure.
   - Reallocate intensities based on each object's mean intensity profile.

5. **Intensity Settlement**
   - Load the morphological and protein intensity CSVs.
   - Update them with intensity corrections from the object-interaction dictionary.
   - Output corrected CSV files per measure type (e.g., `corrected_mean.csv`, `corrected_sum.csv`).

6. **Normalized Intensity Calculation**
   - Normalize per-cell intensities by a weighted average of high-confidence "sensor markers" (typically membrane or cytoplasmic proteins used for coarse cell phenotyping).
   - This enables within-cell-type comparison of functional markers (e.g., checkpoint inhibitors).
   - Up to 4 top sensor markers per cell are used, with intensity thresholds to ensure stability.
   - After per-cell normalization, apply robust per-marker scaling (e.g., 0.1–99.9 percentiles) to bring values into a consistent [0, 1] range.
   - Outputs a CSV of normalized intensities per marker, per cell, suitable for phenotyping and feature ranking.

7. **DeepCell Overlay Mask Processing (Optional)**
   - If `--create_deepcell_mask` is enabled:
     - Generate a visualization overlay from denoised marker TIFFs.
     - Composite the overlay into RGB format using pre-defined marker-channel assignments.
     - Use Selenium to upload this overlay to DeepCell.
     - Download and unpack the resulting segmentation into the FOV folder.

"""


import os

# Limit the number of threads used by various libraries
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # Intel Math Kernel Library
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Numexpr
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # macOS Accelerate framework (if applicable)
# os.environ["BLIS_NUM_THREADS"] = "1"         # BLIS (if used)

import glob
import signal
import random
import time
import logging
import argparse
import json
import functools
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from skimage import io, segmentation, measure, morphology
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_fill_holes
from scipy import ndimage
from tifffile import imread
from tqdm import tqdm

# ---------------------- Logging & Shutdown ---------------------- #
logger = logging.getLogger(__name__) 

def setup_logging(log_level):
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def shutdown_handler(signum, frame):
    logging.info("Graceful shutdown initiated.")
    exit(0)


signal.signal(signal.SIGINT, shutdown_handler)


def log_resource_usage(message):
    logging.debug(f"RESOURCE USAGE: {message}")


# ---------------------- Debug Decorator ---------------------- #
def debug_log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"Entering {func.__name__} with args={args} kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.debug(f"Exiting {func.__name__}")
        return result

    return wrapper


# !/usr/bin/env python3
import os
import re
import glob
import logging
import argparse
import tempfile
import time
import zipfile
import shutil
import requests
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import tifffile

# ----- Selenium Imports for DeepCell Processing ----- #
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile


# ----- Helper Function: Create DeepCell Overlay ----- #
def create_deepcell_mask_overlay(fov_path,
                                 red_markers,
                                 green_markers,
                                 blue_markers=None):
    """
    Create a DeepCell overlay mask from single-slice TIFFs in fov_path.
    Expected file pattern: {marker}.ome.tiff
    The overlay is saved in the fov_path as overlay_<FOV_ID>.ome.tiff.
    """
    if blue_markers is None:
        blue_markers = []  # default: no blue channel

    pattern = os.path.join(fov_path, "*.ome.tiff")
    file_list = glob.glob(pattern)
    if not file_list:
        logger.warning(f"No marker files found in {fov_path} for DeepCell mask creation.")
        return None

    regex = re.compile(r"^(?P<marker>[^.]+)\.ome\.tiff$", re.IGNORECASE)
    marker_dict = {}
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        m = regex.match(file_name)
        if m:
            marker = m.group("marker")
            marker_dict.setdefault(marker, []).append(file_path)
        else:
            logger.info(f"File '{file_name}' does not match expected pattern. Skipping.")

    if not marker_dict:
        logger.warning(f"No valid marker files found in {fov_path} for DeepCell mask creation.")
        return None

    red_channel = None
    green_channel = None
    blue_channel = None if not blue_markers else None

    def clip_0_255(arr):
        return np.clip(arr, 0, 255).astype(np.uint8)

    for marker, files in marker_dict.items():
        marker_img = None
        for file_path in files:
            try:
                img = tifffile.imread(file_path).astype(np.float32)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue
            marker_img = img if marker_img is None else marker_img + img

        if marker_img is None:
            continue

        if red_channel is None:
            H, W = marker_img.shape
            red_channel = np.zeros((H, W), dtype=np.float32)
            green_channel = np.zeros((H, W), dtype=np.float32)
            if blue_markers:
                blue_channel = np.zeros((H, W), dtype=np.float32)

        if marker in red_markers:
            red_channel += marker_img
        elif marker in green_markers:
            green_channel += marker_img
        elif blue_markers and marker in blue_markers:
            blue_channel += marker_img
        else:
            logger.debug(f"Marker '{marker}' not assigned to any channel for overlay. Skipping.")

    if red_channel is None or green_channel is None:
        logger.warning(f"Required markers for red and green channels not found in {fov_path}.")
        return None

    red_8 = clip_0_255(red_channel)
    green_8 = clip_0_255(green_channel)
    blue_8 = clip_0_255(blue_channel) if blue_channel is not None else None

    if blue_8 is not None:
        overlay = np.stack((red_8, green_8, blue_8), axis=-1)
    else:
        overlay = np.stack((red_8, green_8), axis=0)

    fov_id = os.path.basename(os.path.normpath(fov_path))
    overlay_file = os.path.join(fov_path, f"overlay_{fov_id}.tiff")
    try:
        if blue_8 is not None:
            tifffile.imwrite(overlay_file, overlay, photometric="rgb")
        else:
            tifffile.imwrite(overlay_file, overlay)
        logger.info(f"DeepCell overlay mask saved to {overlay_file}")
    except Exception as e:
        logger.error(f"Error saving DeepCell overlay mask for {fov_path}: {e}")
        return None

    return overlay_file


# ----- Helper Function: safe_get ----- #
def safe_get(driver, url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            driver.get(url)
            return
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to load {url}: {e}")
            time.sleep(delay)
    raise Exception(f"Unable to load {url} after {retries} attempts.")


# ----- Helper Function: Process DeepCell Overlay ----- #
def process_deepcell_overlay(overlay_file, output_directory, deepcell_url, geckodriver_path, max_total_wait=300):
    download_directory = tempfile.mkdtemp()
    logger.info(f"Temporary download directory: {download_directory}")

    profile = FirefoxProfile()
    profile.set_preference("pdfjs.disabled", True)

    options = Options()
    options.profile = profile
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = Firefox(service=Service(geckodriver_path), options=options)

    try:
        logger.info("Navigating to DeepCell URL...")
        safe_get(driver, deepcell_url)

        logger.info("Clicking Predict button...")
        WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".MuiButton-containedSecondary"))
        ).click()

        logger.info("Selecting resolution...")
        WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.ID, "input-resolution-select"))
        ).click()
        WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, "//li[contains(.,'10x (1 μm/pixel)')]"))
        ).click()

        logger.info(f"Uploading overlay file: {overlay_file}")
        upload_input = driver.find_element(By.XPATH, "//input[@type='file']")
        upload_input.send_keys(overlay_file)

        logger.info("Submitting overlay for processing...")
        WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.ID, "submitButton"))
        ).click()

        logger.info("Waiting for DeepCell processing to finish...")
        download_button = WebDriverWait(driver, max_total_wait).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(),'Download Results')]"))
        )
        download_url = download_button.get_attribute('href')
        logger.info(f"Retrieved download URL: {download_url}")

        zip_resp = requests.get(download_url, timeout=180)
        if zip_resp.status_code != 200:
            raise Exception(f"Failed to download file, status code: {zip_resp.status_code}")

        zip_path = os.path.join(download_directory, "result.zip")
        with open(zip_path, 'wb') as zf:
            zf.write(zip_resp.content)
        logger.info(f"Downloaded ZIP file to: {zip_path}")

        if not zipfile.is_zipfile(zip_path):
            raise Exception(f"Downloaded file is not a valid ZIP: {zip_path}")

        base_name = os.path.basename(overlay_file).replace("overlay_", "deepcell_mask_").replace(".ome.tiff", "")
        new_zip_path = os.path.join(output_directory, base_name + ".zip")
        shutil.move(zip_path, new_zip_path)
        logger.info(f"Moved ZIP file to final destination: {new_zip_path}")

        extract_dir = os.path.join(output_directory, base_name + "_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(new_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Extracted ZIP file to: {extract_dir}")

        tif_files = sorted(f for f in os.listdir(extract_dir) if f.lower().endswith(".tif"))
        logger.info(f"Extracted TIFF files: {tif_files}")

        if len(tif_files) != 2:
            raise Exception(f"Unexpected number of TIFF files extracted: {len(tif_files)}")

        final_tif_0 = os.path.join(output_directory, base_name + "_0.tiff")
        final_tif_1 = os.path.join(output_directory, base_name + "_1.tiff")
        shutil.move(os.path.join(extract_dir, tif_files[0]), final_tif_0)
        shutil.move(os.path.join(extract_dir, tif_files[1]), final_tif_1)
        logger.info(f"Renamed and moved TIFF files to {final_tif_0} and {final_tif_1}")

    except Exception as e:
        logger.error(f"Error processing overlay {overlay_file}: {e}")
        raise
    finally:
        driver.quit()
        shutil.rmtree(download_directory, ignore_errors=True)
        if os.path.exists(new_zip_path):
            os.remove(new_zip_path)
        shutil.rmtree(extract_dir, ignore_errors=True)
        logger.info("Cleaned up temporary files and closed browser.")


# ---------------------- Stage 1: Mask Processing & Interaction Computation ---------------------- #
# @debug_log
def generate_pseudocolor_mask(mask):
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    unique_labels = unique_labels.tolist()
    random.shuffle(unique_labels)
    pseudocolor_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in unique_labels:
        color_value = np.random.randint(0, 255, 3)
        pseudocolor_mask[mask == label] = color_value
    return pseudocolor_mask


# @debug_log
def load_fov_files(fov_folder, mask_patterns=["*_0.tiff"]):
    """
    Loads file paths for a given FOV folder using specified patterns.

    Args:
      fov_folder (str): Path to the FOV folder.
      mask_patterns (list[str]): List of wildcard patterns for the mask file.

    Returns:
      dict: A dictionary containing lists of file paths for each key.

    Raises:
      ValueError: If the total number of mask files matching the patterns is not exactly one.
    """
    # Combine results from all provided mask patterns
    mask_files = []
    for pattern in mask_patterns:
        found = glob.glob(os.path.join(fov_folder, pattern))
        mask_files.extend(found)

    # Enforce that exactly one mask file is found.
    if len(mask_files) != 1:
        raise ValueError(
            f"Expected exactly one mask file in {fov_folder} matching patterns {mask_patterns}, "
            f"but found {len(mask_files)}: {mask_files}"
        )

    files = {
        "mask": mask_files,  # This will be a list with one file.
        "dna2": glob.glob(os.path.join(fov_folder, '*DNA2.ome.tiff')),
        "dna1": glob.glob(os.path.join(fov_folder, '*DNA1.ome.tiff')),
        "histoneh3": glob.glob(os.path.join(fov_folder, '*HistoneH3.ome.tiff'))
    }
    return files


# @debug_log
def save_image(path, image, description=""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        io.imsave(path, image)

    logging.info(f"Saved {description} at: {path}")



# @debug_log
def process_cell_mask(fov_folder, mask_files):
    cell_mask = io.imread(mask_files[0])
    logging.info(f"Original cell mask shape: {cell_mask.shape}, dtype: {cell_mask.dtype}")
    if len(cell_mask.shape) > 2:
        logging.info("Squeezing cell mask dimensions...")
        cell_mask = np.squeeze(cell_mask)
    cell_mask = cell_mask.astype(np.uint16)
    cellmask_output_path = os.path.join(fov_folder, 'deepcel_mask.tiff')
    save_image(cellmask_output_path, cell_mask, "restructured cell mask")
    pseudocolor = generate_pseudocolor_mask(cell_mask)
    cellmask_pseudocolor_path = os.path.join(fov_folder, 'deepcel_mask_pseudocolor.png')
    save_image(cellmask_pseudocolor_path, pseudocolor, "pseudocolor cell mask")
    return cell_mask


# @debug_log
def process_nuclear_mask(fov_folder, cell_mask, files):
    dna2_signal = io.imread(files["dna2"][0])
    dna1_signal = io.imread(files["dna1"][0])
    histoneh3_signal = io.imread(files["histoneh3"][0])
    logging.info("Loaded nuclear signals.")

    nuclear_signal = dna2_signal + dna1_signal + histoneh3_signal
    logging.info("Computed combined nuclear signal.")

    nuclear_mask = np.zeros_like(cell_mask, dtype=np.uint16)
    labels = np.unique(cell_mask)
    count = 0
    for label in labels:
        if label == 0:
            continue
        cell_region = (cell_mask == label)
        dna_region = nuclear_signal * cell_region
        nucleus_region = dna_region > 0
        filled_nucleus = binary_fill_holes(nucleus_region)
        filled_nucleus = morphology.remove_small_holes(filled_nucleus, area_threshold=64)
        labeled_nuclei = measure.label(filled_nucleus)
        properties = measure.regionprops(labeled_nuclei, intensity_image=dna_region)
        if len(properties) > 1:
            largest_nucleus = max(properties, key=lambda prop: prop.area)
            filled_nucleus = labeled_nuclei == largest_nucleus.label
        nuclear_mask[filled_nucleus] = label
        count += 1
        if count % 50 == 0:
            logging.debug(f"Processed {count} labels out of {len(labels) - 1}")
    logging.info("Nuclear mask processing completed.")
    out_path = os.path.join(fov_folder, 'filled_nucmask.tiff')
    save_image(out_path, nuclear_mask.astype(np.uint16), "filled nuclear mask")
    pseudocolor = generate_pseudocolor_mask(nuclear_mask)
    out_pc = os.path.join(fov_folder, 'filled_nucmask_pseudocolor.png')
    save_image(out_pc, pseudocolor, "pseudocolor nuclear mask")
    return nuclear_mask


@debug_log
def process_membrane_masks(fov_folder, cell_mask):
    """
    Computes:
    1) Membrane mask using inner boundaries.
    2) Membrane exclusion mask where membrane pixels are removed.
    3) Ensures that if a label is completely removed, all its pixels are restored.

    Avoids unnecessary copying of cell_mask and allows missing labels.
    """

    # Compute the membrane mask
    membrane_mask = find_boundaries(cell_mask, mode='inner').astype(np.uint16)

    # Assign each membrane pixel the label of the underlying cell
    membrane_labeled = np.where(membrane_mask > 0, cell_mask, 0).astype(np.uint16)

    # Save the membrane mask
    memmask_output_path = os.path.join(fov_folder, 'membrane_mask.tiff')
    save_image(memmask_output_path, membrane_labeled, "membrane mask")

    # Generate and save pseudocolor mask for visualization
    pseudocolor = generate_pseudocolor_mask(membrane_labeled)
    memmask_pseudocolor_path = os.path.join(fov_folder, 'membrane_mask_pseudocolor.png')
    save_image(memmask_pseudocolor_path, pseudocolor, "pseudocolor membrane mask")

    # Create exclusion mask (cell_mask with membrane pixels removed)
    exclusion_mask = np.where(membrane_mask > 0, 0, cell_mask)

    # Ensure labels that would disappear retain all their pixels
    unique_cell_labels = np.unique(cell_mask[cell_mask > 0])
    unique_excl_labels = np.unique(exclusion_mask[exclusion_mask > 0])

    missing_labels = set(unique_cell_labels) - set(unique_excl_labels)
    if missing_labels:
        logging.info(f"Restoring {len(missing_labels)} missing labels in exclusion mask (too small), fallback-->exclusionmask=cellmask.")

        for label in missing_labels:
            # Restore **all pixels** from `cell_mask` for this missing label
            exclusion_mask[cell_mask == label] = label

    # Save exclusion mask
    exclusion_output_path = os.path.join(fov_folder, 'membrane_exclusion_mask.tiff')
    save_image(exclusion_output_path, exclusion_mask.astype(np.uint16), "membrane exclusion mask")

    # Generate and save pseudocolor exclusion mask
    exclusion_pseudocolor = generate_pseudocolor_mask(exclusion_mask)
    exclusion_pseudocolor_path = os.path.join(fov_folder, 'membrane_exclusion_mask_pseudocolor.png')
    save_image(exclusion_pseudocolor_path, exclusion_pseudocolor, "pseudocolor membrane exclusion mask")

    return membrane_labeled, exclusion_mask


# ---------------------- Stage 2 & 4: Feature and Protein Extraction ---------------------- #
# @debug_log
def extract_morphology_features(fov_folder, morph_features_dir, cell_mask, nuclear_mask=None):
    """
    Extract features from a FOV. If cell_mask and/or nuclear_mask are provided,
    they will be used; otherwise, the function will load them from disk.
    Returns a tuple (features_list, area_lookup) where area_lookup is a dictionary mapping
    each cell label to its Area.
    """
    features_list = []
    try:
        fov_name = os.path.basename(fov_folder)
        logging.info(f"Extracting features for FOV: {fov_name}")

        # Load nuclear mask if not provided.
        if nuclear_mask is None:
            logging.info("No nuclear mask provided; nuclear features will be set to NaN.")

        # Load images for intensity (assumes they exist).
        dna1_paths = glob.glob(os.path.join(fov_folder, '*DNA1.ome.tiff'))
        dna2_paths = glob.glob(os.path.join(fov_folder, '*DNA2.ome.tiff'))
        histoneh3_paths = glob.glob(os.path.join(fov_folder, '*HistoneH3.ome.tiff'))
        if not (dna1_paths and dna2_paths and histoneh3_paths):
            logging.error(f"One or more intensity image files are missing in {fov_folder}")
            return (features_list, {})
        dna1_image = io.imread(dna1_paths[0])
        dna2_image = io.imread(dna2_paths[0])
        histoneh3_image = io.imread(histoneh3_paths[0])

        cell_regions = measure.regionprops(cell_mask)
        if nuclear_mask is not None:
            nuclear_regions = {region.label: region for region in
                               measure.regionprops(nuclear_mask, intensity_image=dna1_image)}
        else:
            nuclear_regions = {}
        if not cell_regions:
            logging.error(f"No cell regions found for FOV {fov_name}. Skipping.")
            return (features_list, {})

        for region in cell_regions:
            label = region.label
            area = region.area
            perimeter = region.perimeter
            convex_area = region.convex_area
            solidity = area / convex_area if convex_area > 0 else np.nan
            bbox_area = region.bbox_area
            extent = area / bbox_area if bbox_area > 0 else np.nan
            orientation = region.orientation
            eccentricity = region.eccentricity
            equivalent_diameter = region.equivalent_diameter
            major_axis_length = region.major_axis_length
            minor_axis_length = region.minor_axis_length
            major_minor_axis_ratio = major_axis_length / minor_axis_length if minor_axis_length > 0 else np.nan
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else np.nan
            form_factor = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else np.nan
            euler_number = region.euler_number
            centroid_row, centroid_col = region.centroid

            # If nuclear mask available and region has a corresponding nucleus, compute stats; otherwise set NaN.
            if nuclear_mask is not None and label in nuclear_regions:
                nucleus_region = nuclear_regions[label]
                nucleus_area = nucleus_region.area
                nucleus_eccentricity = nucleus_region.eccentricity
                nc_area_ratio = nucleus_area / area if area > 0 else np.nan
                nucleus_centroid_row, nucleus_centroid_col = nucleus_region.centroid
                centroid_deviation = np.sqrt((nucleus_centroid_row - centroid_row) ** 2 +
                                             (nucleus_centroid_col - centroid_col) ** 2)
                mean_dna1_intensity = nucleus_region.mean_intensity
                integrated_dna1_intensity = mean_dna1_intensity * nucleus_area
                # For DNA2 and HistoneH3, assume similar logic.
                mean_dna2_intensity = nucleus_region.mean_intensity if dna2_image is not None else np.nan
                integrated_dna2_intensity = mean_dna2_intensity * nucleus_area if dna2_image is not None else np.nan
                mean_histoneh3_intensity = nucleus_region.mean_intensity if histoneh3_image is not None else np.nan
                integrated_histoneh3_intensity = mean_histoneh3_intensity * nucleus_area if histoneh3_image is not None else np.nan
            else:
                nucleus_area = nucleus_eccentricity = nc_area_ratio = np.nan
                nucleus_centroid_row = nucleus_centroid_col = centroid_deviation = np.nan
                mean_dna1_intensity = integrated_dna1_intensity = np.nan
                mean_dna2_intensity = integrated_dna2_intensity = np.nan
                mean_histoneh3_intensity = integrated_histoneh3_intensity = np.nan

            features_list.append({
                'FOV': fov_name,
                'Label': label,
                'Area': area,
                'Perimeter': perimeter,
                'Convex_Area': convex_area,
                'Solidity': solidity,
                'BoundingBox_Area': bbox_area,
                'Extent': extent,
                'Orientation': orientation,
                'Eccentricity': eccentricity,
                'Equivalent_Diameter': equivalent_diameter,
                'Major_Axis_Length': major_axis_length,
                'Minor_Axis_Length': minor_axis_length,
                'Major_Minor_Axis_Ratio': major_minor_axis_ratio,
                'Circularity': circularity,
                'Form_Factor': form_factor,
                'Euler_Number': euler_number,
                'Centroid_Row': centroid_row,
                'Centroid_Col': centroid_col,
                'Nucleus_Area': nucleus_area,
                'Nucleus_Eccentricity': nucleus_eccentricity,
                'NC_Area_Ratio': nc_area_ratio,
                'Nucleus_Centroid_Row': nucleus_centroid_row,
                'Nucleus_Centroid_Col': nucleus_centroid_col,
                'Centroid_Deviation': centroid_deviation,
                'Mean_DNA1_Intensity': mean_dna1_intensity,
                'Integrated_DNA1_Intensity': integrated_dna1_intensity,
                'Mean_DNA2_Intensity': mean_dna2_intensity,
                'Integrated_DNA2_Intensity': integrated_dna2_intensity,
                'Mean_HistoneH3_Intensity': mean_histoneh3_intensity,
                'Integrated_HistoneH3_Intensity': integrated_histoneh3_intensity
            })
        morph_features = pd.DataFrame(features_list)
        csv_path = os.path.join(morph_features_dir, f"{fov_name}.csv")
        morph_features.to_csv(csv_path, index=False)
        logging.info(f"Saved features for FOV {fov_name} to {csv_path}")
    except Exception as e:
        logging.error(f"Error extracting features for FOV {fov_folder}: {e}")
    return (morph_features)


@debug_log
def extract_protein_intensity(fov_folder, protein_features_dir, morph_features, cell_mask,
                              membrane_mask, memexcl_mask):
    try:
        fov_name = os.path.basename(fov_folder)
        logging.info(f"Processing protein intensities for FOV: {fov_name}")

        # Extract Labels and Areas directly from morph_features
        if not {"Label", "Area"}.issubset(morph_features.columns):
            logging.error(f"Missing 'Label' or 'Area' columns in morph_features for {fov_name}, skipping.")
            return None

        labels = morph_features["Label"].values
        areas = morph_features["Area"].values

        # Get unique labels from the cell mask (excluding background)
        unique_labels = np.unique(cell_mask)
        unique_labels = unique_labels[unique_labels != 0]
        n_labels = len(unique_labels)

        logging.debug(f"Number of labels in cell mask: {n_labels}")
        if len(areas) != n_labels:
            logging.warning(f"Mismatch: {len(areas)} areas vs. {n_labels} labels for {fov_name}")

        # Initialize results dictionary
        results = {"FOV": [fov_name] * n_labels, "Label": unique_labels.tolist()}

        # Identify and process OME-TIFF marker images (excluding DNA/Histone)
        ome_files = [f for f in glob.glob(os.path.join(fov_folder, "*.ome.tiff"))
                     if "DNA" not in f and "Histone" not in f]

        for ome_file in ome_files:
            marker_name = os.path.basename(ome_file).replace(".ome.tiff", "")
            marker_image = io.imread(ome_file)

            # Compute summed intensities per region
            cell_sum = np.atleast_1d(ndimage.sum(marker_image, labels=cell_mask, index=unique_labels))
            mem_sum = np.atleast_1d(ndimage.sum(marker_image, labels=membrane_mask, index=unique_labels))
            memexcl_sum = np.atleast_1d(ndimage.sum(marker_image, labels=memexcl_mask, index=unique_labels))

            logging.debug(f"Marker {marker_name}: cell_sum: {len(cell_sum)}, "
                          f"mem_sum: {len(mem_sum)}, memexcl_sum: {len(memexcl_sum)}")

            # Handle cases where membrane exclusion sum is zero but area is nonzero
            memexcl_sum = np.where((memexcl_sum == 0) & (areas > 0), cell_sum, memexcl_sum)

            # Compute mean intensities directly using NumPy division (avoid loops)
            valid_areas = np.where(areas > 0, areas, 1)  # Avoid division by zero
            cell_mean = cell_sum / valid_areas
            mem_mean = mem_sum / valid_areas
            memexcl_mean = memexcl_sum / valid_areas

            # Store computed values in results
            results[f"{marker_name}_Cell_Mean_Intensity"] = cell_mean.tolist()
            results[f"{marker_name}_Cell_Sum_Intensity"] = cell_sum.tolist()
            results[f"{marker_name}_Membrane_Mean_Intensity"] = mem_mean.tolist()
            results[f"{marker_name}_Membrane_Sum_Intensity"] = mem_sum.tolist()
            results[f"{marker_name}_ExclusionMembrane_Mean_Intensity"] = memexcl_mean.tolist()
            results[f"{marker_name}_ExclusionMembrane_Sum_Intensity"] = memexcl_sum.tolist()

        # Convert to DataFrame
        protein_features = pd.DataFrame(results)

        # Save to CSV
        output_csv_path = os.path.join(protein_features_dir, f"{fov_name}.csv")
        protein_features.to_csv(output_csv_path, index=False)
        logging.info(f"Saved protein intensity summary for FOV: {fov_folder}")

        return protein_features  # Return the DataFrame

    except Exception as e:
        logging.error(f"Error in protein intensity extraction for FOV {fov_folder}: {e}")
        return None  # Return None in case of failure


# @debug_log
def compute_border_interactions(cell_mask, membrane_mask):
    interactions = defaultdict(dict)
    membrane_indices = np.argwhere(membrane_mask > 0)[:, :2]
    logging.info(f"Border interactions: found {len(membrane_indices)} membrane pixels.")
    for y, x in membrane_indices:
        current_label = cell_mask[y, x]
        neighbor_labels = set()
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1), (0, 1),
                       (1, -1), (1, 0), (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < membrane_mask.shape[0] and 0 <= nx < membrane_mask.shape[1]:
                neighbor_label = cell_mask[ny, nx]
                if neighbor_label != 0 and neighbor_label != current_label:
                    neighbor_labels.add(neighbor_label)
        if neighbor_labels:
            interactions[f"{y}_{x}"] = {
                "current": int(current_label),
                "neighbors": sorted([int(label) for label in neighbor_labels]),
                "type": "border"
            }
    return interactions


# @debug_log
def compute_background_interactions(cell_mask):
    logging.info("Processing background interactions")
    background_indices = np.argwhere(cell_mask == 0)[:, :2]
    background_interactions = {}
    neighbor_count = Counter()
    for y, x in background_indices:
        object_labels = set()
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1), (0, 1),
                       (1, -1), (1, 0), (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < cell_mask.shape[0] and 0 <= nx < cell_mask.shape[1]:
                neighbor_label = cell_mask[ny, nx]
                if neighbor_label != 0:
                    object_labels.add(neighbor_label)
        if object_labels:
            background_interactions[f"{y}_{x}"] = {
                "current": "background",
                "interacts_with": sorted([int(label) for label in object_labels]),
                "type": "background"
            }
            neighbor_count[len(object_labels)] += 1
    return background_interactions, neighbor_count


# @debug_log
def merge_interactions(border_int, background_int):
    logging.info("Processing merge")
    merged = {}
    merged.update(border_int)
    merged.update(background_int)
    return merged


# @debug_log
from skimage import io
import os
import glob
import logging


def integrate_intensities_for_interactions(fov_folder, interactions):
    """
    Integrate pixel intensities from OME-TIFF files into the interactions dictionary.

    For each OME-TIFF file found in the fov_folder, this function:
      - Extracts a marker name from the filename.
      - Reads the image using skimage.io.imread.
      - For every coordinate key in the interactions dictionary (formatted as "y_x"),
        it retrieves the pixel value from the image.
          - If the pixel is multi-channel (i.e. an array with more than one element),
            it takes the first channel's value. (You can modify this to compute an average,
            maximum, etc.)
          - Otherwise, it converts the pixel value directly to a float.
      - This intensity value is stored under the 'intensities' sub-dictionary of the
        interaction, keyed by the marker name.

    Extensive logging is provided for each major step.

    Args:
      fov_folder (str): The folder path for the current Field-Of-View.
      interactions (dict): The interactions dictionary with coordinate keys.

    Returns:
      dict: The updated interactions dictionary with integrated intensity values.
    """
    logging.info("Starting intensity integration for interactions.")

    # Find all OME-TIFF files in the fov_folder.
    ome_tiff_files = glob.glob(os.path.join(fov_folder, '*.ome.tiff'))
    logging.info(f"Found {len(ome_tiff_files)} OME-TIFF file(s) in {fov_folder}.")

    if not ome_tiff_files:
        logging.info(f"No OME-TIFF files found in {fov_folder}. Returning original interactions.")
        return interactions

    # Process each OME-TIFF file individually.
    for ome_file in ome_tiff_files:
        # Derive the marker name by removing the '.ome.tiff' suffix from the filename.
        marker_name = os.path.basename(ome_file).replace('.ome.tiff', '')
        logging.debug(f"Processing file: {ome_file} (marker: '{marker_name}').")

        # Read the OME-TIFF image from disk.
        try:
            ome_image = io.imread(ome_file)
            logging.debug(
                f"Successfully loaded image from {ome_file} with shape {ome_image.shape} and dtype {ome_image.dtype}.")
        except Exception as e:
            logging.error(f"Error reading file {ome_file}: {e}")
            continue  # Skip this file if it cannot be read

        # Loop over each coordinate in the interactions dictionary.
        for coord_str, data in interactions.items():
            try:
                # Expecting coordinate keys in the format "y_x"
                y, x = map(int, coord_str.split('_'))
            except Exception as e:
                logging.error(f"Error parsing coordinate '{coord_str}': {e}")
                continue

            # Check if the coordinate is within the image bounds.
            if 0 <= y < ome_image.shape[0] and 0 <= x < ome_image.shape[1]:
                pix = ome_image[y, x]
                # logging.debug(f"At coordinate ({y}, {x}), raw pixel value: {pix}")

                # If the pixel value is an array (multi-channel), handle accordingly.
                if hasattr(pix, "ndim") and pix.ndim > 0:
                    # If it is a one-element array, extract the scalar.
                    if pix.size == 1:
                        intensity_value = float(pix.item())
                        logging.debug(
                            f"Pixel at ({y}, {x}) is a single-element array; converted to scalar: {intensity_value}")
                    else:
                        # For multi-channel, take the first channel.
                        intensity_value = float(pix[0])
                        logging.debug(
                            f"Pixel at ({y}, {x}) is multi-channel; using first channel value: {intensity_value}")
                else:
                    # Otherwise, it is assumed to be already a scalar.
                    intensity_value = float(pix)
                    logging.debug(f"Pixel at ({y}, {x}) is scalar; value: {intensity_value}")
            else:
                intensity_value = None
                logging.warning(
                    f"Coordinate ({y}, {x}) is out-of-bounds for image with shape {ome_image.shape}. Setting intensity to None.")

            # Add the intensity value to the 'intensities' dictionary in the interactions entry.
            if 'intensities' not in data:
                data['intensities'] = {}
            data['intensities'][marker_name] = intensity_value
            # logging.info(f"Set intensity for marker '{marker_name}' at coordinate '{coord_str}' to {intensity_value}.")

    logging.info("Completed intensity integration for interactions.")
    return interactions


# @debug_log

import pandas as pd
from collections import defaultdict

import logging
import numpy as np
import pandas as pd
from collections import defaultdict


def compute_reallocation_with_checks(interactions, protein_features, tol=1e-6):
    logging.info("start calculating reallocation")
    """
    Group the combined per-coordinate interactions (from border and background)
    by object (cell label) and compute a per-object intensity dictionary.

    - 'Markers' with 'DNA' or 'Histone' in their name are skipped.
    - Border intensities are always "taken" by the primary cell (the 'current'),
      then reallocated among current + neighbors based on each cell's
      ExclusionMembrane mean intensity.
    - Background intensities are reallocated among all interacting cells based
      on each cell's ExclusionMembrane mean intensity.
    - If, for border, the sum of means is 0, the 'current' cell keeps
      (reallocated) all intensity. For background, if all means are 0, that
      intensity is effectively lost.

    Returns:
      A dict mapping each cell label to another dict:
        {
          'taken_intensity':      { marker: value, ... },
          'reallocated_intensity': { marker: value, ... }
        }
    """

    """
    Similar to the original compute_reallocation, but with additional checks:
      1) For border interactions, checks that sum(taken) == sum(reallocated).
      2) Ensures that no border reallocation > total intensity.
      3) For background, 'lost' intensity is not assigned to any cell. 
         We do not raise an error if there's leftover unclaimed background,
         but we verify that the total allocated is <= total_intensity.
    """

    # 1) Precompute mean_exclusion_membrane_intensity for each (cell, marker)
    mean_intensities = {}
    for cell_idx in protein_features.index:
        for col in protein_features.columns:
            if col.endswith("_ExclusionMembrane_Mean_Intensity"):
                marker_name = col.replace("_ExclusionMembrane_Mean_Intensity", "")
                val = protein_features.at[cell_idx, col]
                if pd.isna(val):
                    val = 0
                else:
                    # Ensure the value is a scalar
                    if isinstance(val, np.ndarray):
                        try:
                            val = val.item()
                        except Exception as e:
                            logging.error(
                                f"Error converting protein_features value at cell {cell_idx}, column {col}: {val} (type: {type(val)})")
                            raise
                mean_intensities[(cell_idx, marker_name)] = val

    reallocation = defaultdict(lambda: {
        "taken_intensity": defaultdict(float),
        "reallocated_intensity": defaultdict(float)
    })

    # 2) GROUP & PROCESS BORDER INTERACTIONS
    border_groups = defaultdict(list)
    for coord, data in interactions.items():
        if data.get("type") == "border" and "intensities" in data:
            key = (data["current"], tuple(sorted(data.get("neighbors", []))))
            border_groups[key].append(coord)

    for (current, neighbors), coords in border_groups.items():
        # For each group, we also track sums to verify after distribution.
        # `marker_total_taken` for all border coords in this group
        # `marker_total_reallocated` for sum across all cells
        marker_group_taken = defaultdict(float)
        marker_group_reallocated = defaultdict(float)

        # Build union of all markers
        marker_set = set()
        for coord in coords:
            marker_set.update(interactions[coord]["intensities"].keys())

        for marker in marker_set:
            if "DNA" in marker or "Histone" in marker:
                continue

            # Sum intensities for these coords
            total_intensity = 0.0
            for coord in coords:
                total_intensity += interactions[coord]["intensities"].get(marker, 0)

            # Mark that the current cell "takes" it
            reallocation[current]["taken_intensity"][marker] += total_intensity

            # Bookkeeping for checks
            marker_group_taken[marker] += total_intensity

            # Distribute
            current_mean = mean_intensities.get((current, marker), 0)
            neighbor_means = [mean_intensities.get(n, 0)
                              if isinstance(n, tuple) else mean_intensities.get((n, marker), 0)
                              for n in neighbors]
            # (Note: in case `neighbors` are ints, we do get((n, marker), 0).
            #  If your code sometimes yields weird tuples, adapt accordingly.)

            total_mean = current_mean + sum(neighbor_means)

            if total_mean == 0:
                # Fallback: reallocate all intensity to the current cell
                reallocation[current]["reallocated_intensity"][marker] += total_intensity
                # Also track reallocated sum (for checks)
                marker_group_reallocated[marker] += total_intensity
            else:
                # Collect cells with nonzero means
                labels_nonzero = []
                intensities_nonzero = []
                if current_mean > 0:
                    labels_nonzero.append(current)
                    intensities_nonzero.append(current_mean)
                for nb, mval in zip(neighbors, neighbor_means):
                    if mval > 0:
                        labels_nonzero.append(nb)
                        intensities_nonzero.append(mval)

                sum_nonzero = sum(intensities_nonzero)
                if sum_nonzero == 0:
                    # Should only happen if all means were <= 0
                    reallocation[current]["reallocated_intensity"][marker] += total_intensity
                    marker_group_reallocated[marker] += total_intensity
                else:
                    for lab, mval in zip(labels_nonzero, intensities_nonzero):
                        factor = mval / sum_nonzero
                        allocated = factor * total_intensity
                        reallocation[lab]["reallocated_intensity"][marker] += allocated
                        marker_group_reallocated[marker] += allocated

        # Now check sums for all markers in this group
        for marker in marker_group_taken.keys():
            taken_val = marker_group_taken[marker]
            reallocated_val = marker_group_reallocated[marker]
            # Make sure we didn't create or destroy intensity
            if not (abs(taken_val - reallocated_val) <= tol):
                msg = (f"Border group mismatch for (cell={current}, neighbors={neighbors}), "
                       f"marker={marker}. Taken={taken_val}, Reallocated={reallocated_val}.")
                logging.error(msg)
                # Optionally: raise ValueError(msg)

    # 3) GROUP & PROCESS BACKGROUND INTERACTIONS
    background_groups = defaultdict(list)
    for coord, data in interactions.items():
        if data.get("type") == "background" and "intensities" in data:
            key = tuple(sorted(data.get("interacts_with", [])))
            background_groups[key].append(coord)

    for cells, coords in background_groups.items():
        # For background, it's normal that some portion may remain unclaimed
        # (if no cell has a nonzero mean).
        # So we only check that sum(allocated) <= total_intensity.
        marker_allocated = defaultdict(float)

        marker_set = set()
        for coord in coords:
            marker_set.update(interactions[coord]["intensities"].keys())

        for marker in marker_set:
            if "DNA" in marker or "Histone" in marker:
                continue

            total_intensity = 0.0
            for coord in coords:
                total_intensity += interactions[coord]["intensities"].get(marker, 0)

            if len(cells) == 1:
                cell = cells[0]
                cell_mean = mean_intensities.get((cell, marker), 0)
                if cell_mean > 0:
                    reallocation[cell]["reallocated_intensity"][marker] += total_intensity
                    marker_allocated[marker] += total_intensity
            else:
                nonzero_labels = []
                nonzero_means = []
                for cell_id in cells:
                    val = mean_intensities.get((cell_id, marker), 0)
                    if val > 0:
                        nonzero_labels.append(cell_id)
                        nonzero_means.append(val)

                if nonzero_labels:
                    sum_nonzero = sum(nonzero_means)
                    if sum_nonzero == 0:
                        # No actual claims
                        # Then no intensity is allocated
                        pass
                    else:
                        for lab, mval in zip(nonzero_labels, nonzero_means):
                            factor = mval / sum_nonzero
                            allocated_val = factor * total_intensity
                            reallocation[lab]["reallocated_intensity"][marker] += allocated_val
                            marker_allocated[marker] += allocated_val

            # Safety check: ensure allocated <= total
            if marker_allocated[marker] - total_intensity > tol:
                msg = (f"Background over-allocation for marker={marker}, "
                       f"allocated={marker_allocated[marker]:.6f} vs total={total_intensity:.6f}.")
                logging.error(msg)
                # Optionally: raise ValueError(msg)

    logging.info("Finished reallocation with safety checks.")
    return dict(reallocation)


@debug_log
def settle_debts_intensity(fov_folder, reallocation, protein_features, original_sum_dir, unhuddle_sum_dir):
    """
    Applies intensity reallocation to protein_features, and saves both original and corrected sum intensities.

    Args:
        fov_folder (str): Path to the FOV folder (used to extract FOV name).
        reallocation (dict): Dictionary of taken/reallocated intensities per label.
        protein_features (pd.DataFrame): Indexed by (FOV, Label) with _Cell_Sum_Intensity columns.
        original_sum_dir (str): Directory to save original sum CSV.
        unhuddle_sum_dir (str): Directory to save corrected sum CSV.

    Returns:
        pd.DataFrame: Original sum intensities.
        pd.DataFrame: Corrected sum intensities.
    """
    import os
    import pandas as pd

    fov_name = os.path.basename(fov_folder)
    logging.info(f"[{fov_name}] Starting intensity settlement...")

    os.makedirs(original_sum_dir, exist_ok=True)
    os.makedirs(unhuddle_sum_dir, exist_ok=True)

    # Ensure index is (FOV, Label)
    if not isinstance(protein_features.index, pd.MultiIndex) or protein_features.index.names != ["FOV", "Label"]:
        logging.info(f"[{fov_name}] Setting protein_features index to (FOV, Label).")
        protein_features = protein_features.set_index(["FOV", "Label"])

    # Identify markers
    sum_cols = [col for col in protein_features.columns if col.endswith("_Cell_Sum_Intensity")]
    marker_names = sorted([col.replace("_Cell_Sum_Intensity", "") for col in sum_cols])
    if not marker_names:
        logging.error(f"[{fov_name}] No valid _Cell_Sum_Intensity markers found.")
        return None

    # --- Save original sums before changes ---
    orig_sum_df = protein_features.reset_index().drop(columns=["FOV"], errors="ignore")[
        ["Label"] + sum_cols]
    orig_sum_df = orig_sum_df.rename(columns={f"{m}_Cell_Sum_Intensity": m for m in marker_names})
    orig_path = os.path.join(original_sum_dir, f"{fov_name}.csv")
    orig_sum_df.to_csv(orig_path, index=False)
    logging.info(f"[{fov_name}] Saved original sum intensities to {orig_path}")

    # --- Apply reallocation ---
    for label, data in reallocation.items():
        parsed_label = int(label) if isinstance(label, str) and label.isdigit() else label
        key = (fov_name, parsed_label)
        if key not in protein_features.index:
            logging.warning(f"[{fov_name}] Skipping missing cell: {key}")
            continue
        for marker, taken_val in data.get("taken_intensity", {}).items():
            col = f"{marker}_Cell_Sum_Intensity"
            if col in protein_features.columns:
                protein_features.at[key, col] -= taken_val
        for marker, realloc_val in data.get("reallocated_intensity", {}).items():
            col = f"{marker}_Cell_Sum_Intensity"
            if col in protein_features.columns:
                protein_features.at[key, col] += realloc_val

    # --- Save corrected sums ---
    corrected_sum_df = protein_features.reset_index().drop(columns=["FOV"], errors="ignore")[
        ["Label"] + sum_cols]
    corrected_sum_df = corrected_sum_df.rename(columns={f"{m}_Cell_Sum_Intensity": m for m in marker_names})
    corr_path = os.path.join(unhuddle_sum_dir, f"{fov_name}.csv")
    corrected_sum_df.to_csv(corr_path, index=False)
    logging.info(f"[{fov_name}] Saved corrected sum intensities to {corr_path}")

    return orig_sum_df, corrected_sum_df


# ---------------------- Stage 6: Normalized Intensity Calculation ---------------------- #
def fully_weighted_normalize_and_scale(matrix, var_names, sensor_markers, max_n=4, min_relative_weight=0.01,
                                       lower=0.1, upper=99.9):
    """
    Performs per-cell normalization using top sensor markers (weighted), followed by robust per-marker scaling.

    Parameters:
        matrix (np.ndarray): Raw intensity matrix (cells x markers)
        var_names (list): List of marker names (length = matrix.shape[1])
        sensor_markers (list): Markers used for per-cell normalization
        max_n (int): Max number of sensor markers to use per cell
        min_relative_weight (float): Minimum contribution threshold for a marker to be used
        lower (float): Lower percentile for robust scaling
        upper (float): Upper percentile for robust scaling

    Returns:
        scaled_matrix (np.ndarray): Matrix after per-cell and per-marker normalization
        marker_counts (np.ndarray): Number of sensor markers used per cell
    """
    marker_idx = [i for i, name in enumerate(var_names) if name in sensor_markers]
    norm_matrix = np.zeros_like(matrix)
    marker_counts = np.zeros(matrix.shape[0], dtype=int)

    # --- Per-cell normalization using sensor markers ---
    for i in range(matrix.shape[0]):
        cell_values = matrix[i, marker_idx]
        sorted_idx = np.argsort(cell_values)[::-1]
        sorted_values = cell_values[sorted_idx]

        top_value = sorted_values[0] if len(sorted_values) > 0 else 0
        if top_value <= 0:
            continue

        relative_values = sorted_values / top_value
        weights_mask = relative_values >= min_relative_weight
        selected_raw_values = sorted_values[weights_mask][:max_n]
        selected_relative_values = relative_values[weights_mask][:max_n]

        marker_counts[i] = len(selected_raw_values)

        if len(selected_raw_values) > 0:
            weights = selected_relative_values
            scale = np.average(selected_raw_values, weights=weights)
            if scale > 0:
                norm_matrix[i, :] = matrix[i, :] / scale

    # --- Per-marker robust scaling ---
    scaled_matrix = np.zeros_like(norm_matrix)
    for j in range(norm_matrix.shape[1]):
        col = norm_matrix[:, j]
        nonzero_mask = col != 0
        nonzero_vals = col[nonzero_mask]

        if len(nonzero_vals) > 0:
            p1, p99 = np.percentile(nonzero_vals, [lower, upper])
            if p99 != p1:
                scaled_col = (col - p1) / (p99 - p1)
                scaled_col = np.clip(scaled_col, 0, 1)
                scaled_col[~nonzero_mask] = 0
                scaled_matrix[:, j] = scaled_col
            else:
                scaled_matrix[:, j] = 0.5  # fallback for flat marker
        else:
            scaled_matrix[:, j] = 0

    return scaled_matrix, marker_counts

@debug_log
def compute_normalized_intensities_for_fov(fov_folder, corrected_sum_df, sensor_markers):
    """
    Computes per-cell and per-marker normalized intensities using sensor markers.

    Args:
        fov_folder (str): Path to the FOV folder (used for naming only).
        corrected_sum_df (pd.DataFrame): DataFrame with 'Label' and marker columns.
        sensor_markers (list): List of markers used for per-cell normalization.

    Returns:
        fov_name (str): Extracted FOV name
        normalized_df (pd.DataFrame): DataFrame with 'Label', scaled marker values
    """
    import numpy as np
    import pandas as pd
    import os

    fov_name = os.path.basename(fov_folder)
    logging.info(f"Normalizing intensities for {fov_name}")

    if "Label" not in corrected_sum_df.columns:
        logging.error(f"{fov_name} is missing required 'Label' column.")
        return fov_name, None

    # Extract marker columns
    marker_columns = sorted([col for col in corrected_sum_df.columns if col != 'Label'])
    raw_matrix = corrected_sum_df[marker_columns].values
    logging.debug(f"Using marker columns: {marker_columns}")

    # Normalize and scale
    scaled_matrix, marker_counts = fully_weighted_normalize_and_scale(
        matrix=raw_matrix,
        var_names=marker_columns,
        sensor_markers=sensor_markers,
        max_n=4,
        min_relative_weight=0.01,
        lower=0.1,
        upper=99.9
    )

    # Construct output DataFrame
    normalized_df = pd.DataFrame(scaled_matrix, columns=marker_columns)
    normalized_df.insert(0, 'Label', corrected_sum_df["Label"].values)

    return fov_name, normalized_df


# ---------------------- Combined FOV Pipeline ---------------------- #
@debug_log
def process_fov_pipeline(fov_path, morph_features_dir, protein_features_dir,
                         original_sum_dir, original_norm_dir,
                         unhuddle_sum_dir, unhuddle_norm_dir, create_nuclear_mask,
                         create_deepcell_mask, geckodriver_path, deepcell_url, mask_pattern, markers_for_normalisation, red_markers, green_markers, blue_markers):
    result = {"fov": fov_path}
    try:
        if create_deepcell_mask:
            overlay_file = create_deepcell_mask_overlay(
                fov_path,
                red_markers=red_markers,
                green_markers=green_markers,
                blue_markers=blue_markers
            )

            result["deepcell_overlay_file"] = overlay_file
            if overlay_file:
                try:
                    process_deepcell_overlay(overlay_file, fov_path, deepcell_url, geckodriver_path)
                    result["deepcell_processing"] = "Success"
                except Exception as e:
                    logger.error(f"DeepCell processing failed for {fov_path}: {e}")
                    result["deepcell_processing"] = f"Error: {e}"
            else:
                result["deepcell_processing"] = "Overlay file not created"

        files = load_fov_files(fov_path, mask_pattern)
        cell_mask = process_cell_mask(fov_path, files["mask"])
        if create_nuclear_mask:
            nuclear_mask = process_nuclear_mask(fov_path, cell_mask, files)
        else:
            nuclear_mask = None
            logger.info("Skipping nuclear mask creation as per flag.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during mask creation for {fov_path}: {e}, skipping this FOV")
        print(f"\n[!] Skipped FOV: {fov_path} often due to artefactual image, failing deepcell QC.\n"
              f"    ➤ TIP: You can 1. retry with 1 worker or 2. add a custom mask file '*_0.tiff' and run the code again with the --fovs flag.\n"
              f"    ➤ You can override the mask pattern using --mask-pattern if needed.\n")
        result["critical_error"] = f"Mask creation failed: {e}"
        return result  # Stop further processing, all downstream steps depend on this

    try:
        morph_features = extract_morphology_features(fov_path, morph_features_dir, cell_mask, nuclear_mask)
    except Exception as e:
        logger.error(f"Error extracting features for {fov_path}: {e}")
        result["feature_extraction_error"] = str(e)

    try:
        membrane_mask, memexcl_mask = process_membrane_masks(fov_path, cell_mask)
    except Exception as e:
        logger.error(f"Error creating membrane-related masks for {fov_path}: {e}")
        result["membrane_mask_error"] = str(e)

    try:
        protein_features = extract_protein_intensity(fov_path, protein_features_dir,
                                                     morph_features, cell_mask, membrane_mask, memexcl_mask)
        result["protein_intensity_extracted"] = True
    except Exception as e:
        logger.error(f"Error in protein intensity extraction for {fov_path}: {e}")
        result["protein_intensity_error"] = str(e)

    try:
        border_int = compute_border_interactions(cell_mask, membrane_mask)
        background_int, _ = compute_background_interactions(cell_mask)
        merged = merge_interactions(border_int, background_int)
        all_interactions = integrate_intensities_for_interactions(fov_path, merged)
        reallocation = compute_reallocation_with_checks(all_interactions, protein_features, 1e-6)
    except Exception as e:
        logger.error(f"Error in object intensity analysis for {fov_path}: {e}")
        result["object_intensity_error"] = str(e)

    try:
        original_sum_df, corrected_sum_df = settle_debts_intensity(fov_path, reallocation, protein_features,
                                                                   original_sum_dir, unhuddle_sum_dir)
        result["intensity_settled"] = True
    except Exception as e:
        logger.error(f"Error settling intensities for {fov_path}: {e}")
        result["intensity_settlement_error"] = str(e)

    try:
        fov_name, norm_corrected = compute_normalized_intensities_for_fov(
            fov_folder=fov_path,
            corrected_sum_df=corrected_sum_df,
            sensor_markers=markers_for_normalisation
        )

        if norm_corrected is not None:
            output_file = os.path.join(unhuddle_norm_dir, f"{fov_name}.csv")
            norm_corrected.to_csv(output_file, index=False)
            logger.info(f"Saved normalized intensities for {fov_name} to {output_file}")
            result["normalized_intensity"] = True
        else:
            raise ValueError("Normalization returned None")
        _, norm_original = compute_normalized_intensities_for_fov(
            fov_folder=fov_path,
            corrected_sum_df=original_sum_df,
            sensor_markers=markers_for_normalisation
        )

        if norm_original is not None:
            output_file = os.path.join(original_norm_dir, f"{fov_name}.csv")
            norm_original.to_csv(output_file, index=False)
            logger.info(f"Saved normalized original intensities for {fov_name} to {output_file}")
            result["normalized_original_intensity"] = True
        else:
            raise ValueError("Normalization returned None")
    except Exception as e:
        logger.error(f"Error computing normalized intensities for {fov_path}: {e}")
        result["normalized_intensity_error"] = str(e)
        return result


# ----- Main Entry Point ----- #
def main():
    parser = argparse.ArgumentParser(
        description="Integrated pipeline to process FOV folders. Essential stages are run for each FOV."
    )
    parser.add_argument("--base_path", type=str, required=True, help="Base path containing FOV folders")
    parser.add_argument("--output_base_path", type=str, required=True, help="Base path for output")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--create_nuclear_mask", action="store_true", default=False,
                        help="Flag to create the nuclear mask, will ensure morphology features and N/C ratio")
    parser.add_argument("--create_deepcell_mask", action="store_true", default=False,
                        help="Flag to create and process the DeepCell overlay mask")
    parser.add_argument(
    "--red-markers", nargs="+",
    default=["DNA1", "DNA2", "HistoneH3"],
    help="List of red channel markers (nuclear)"
    )
    parser.add_argument(
    "--green-markers", nargs="+",
    default=["CD7", "CD3", "CD15", "CD11c", "CD68", "CD45RO", "CD45RA", "CD20", "Vimentin"],
    help="List of green channel markers (membrane/cytoplasm)"
    )
    parser.add_argument(
    "--blue-markers", nargs="+",
    default=[],
    help="List of blue channel markers (optional, will currently be ignored for mask creation)"
    )
    parser.add_argument("--geckodriver_path", type=str,
                        default="/drive3/tnoorden/tools/geckodriver-v0.35.0-linux64/geckodriver",
                        help="Path to the geckodriver executable")
    parser.add_argument("--deepcell_url", type=str,
                        default="http://www.deepcell.org",
                        help="URL for the DeepCell website.")
    parser.add_argument("--fovs", nargs="*", default=None,
                        help="List of FOV folder names to process. If not specified, all FOV folders in base_path will be processed.")
    parser.add_argument("--mask_pattern", type=str, nargs='+', default=["*_0.tiff"],
                        help="Wildcard pattern(s) for the mask file (default: '*_0.tiff').")
    parser.add_argument("--list_available_markers", action="store_true",
                        help="Print list of available markers and exit.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING")
    parser.add_argument("--check_output_exist", action="store_true", default=False,
                        help="If set, the script will check if output already exists in normalized_output_dir and skip that FOV.")
    parser.add_argument("--markers_for_normalisation", nargs="*", default=None,
                        help="List of markers to use for normalization (e.g. CD45 CD3 Vimentin). Required unless --list_available_markers is used.")

    args = parser.parse_args()
    # Conditional requirement for markers
    if not args.list_available_markers and not args.markers_for_normalisation:
        parser.error("--markers_for_normalisation is required unless --list_available_markers is set.")

    if args.list_available_markers:
        import glob
        import sys

        # Find FOV folders
        fov_folders = [
            os.path.join(args.base_path, fov)
            for fov in os.listdir(args.base_path)
            if os.path.isdir(os.path.join(args.base_path, fov))
        ]

        if not fov_folders:
            print("No FOV folders found under base_path. Cannot list markers.")
            sys.exit(1)

        # Use first FOV
        first_fov_path = fov_folders[0]
        fov_name = os.path.basename(first_fov_path)

        # Find *.ome.tiff files
        ome_tiffs = sorted(glob.glob(os.path.join(first_fov_path, "*.ome.tiff")))
        if not ome_tiffs:
            print(f"No .ome.tiff files found in {first_fov_path}. Cannot list markers.")
            sys.exit(1)

        # Extract marker names
        markers = sorted([
            os.path.basename(path).replace(".ome.tiff", "")
            for path in ome_tiffs
        ])

        print(f"\nAvailable markers in FOV '{fov_name}':\n")
        for m in markers:
            print(f"  {m}")
        print(f"\nTotal: {len(markers)} marker files found.")
        sys.exit(0)
    
    #set logger granularity based on flag 
    setup_logging(args.log_level)
    logger.info("Starting pipeline...")

    output_base_path = args.output_base_path
    morph_features_dir = os.path.join(output_base_path, "morphology_features")
    protein_features_dir = os.path.join(output_base_path, "protein_features")
    original_sum_dir = os.path.join(output_base_path, "original_sum")
    original_norm_dir = os.path.join(output_base_path, "original_normalized")
    unhuddle_sum_dir = os.path.join(output_base_path, "unhuddle_sum")
    unhuddle_norm_dir = os.path.join(output_base_path, "unhuddle_normalized")

    # Create all required directories
    for d in [output_base_path, morph_features_dir, protein_features_dir, original_sum_dir,
              original_norm_dir, unhuddle_sum_dir, unhuddle_norm_dir]:
        os.makedirs(d, exist_ok=True)

    # Gather FOV folders from the base path.
    all_fov_folders = [
        os.path.join(args.base_path, fov)
        for fov in os.listdir(args.base_path)
        if os.path.isdir(os.path.join(args.base_path, fov))
    ]

    # If --fovs is provided, filter the FOV folders.
    if args.fovs:
        fov_folders = [fov for fov in all_fov_folders if os.path.basename(fov) in args.fovs]
    else:
        fov_folders = all_fov_folders

    # If the check_output_exist flag is set, filter out FOVs that already have output.
    if args.check_output_exist:
        import glob
        filtered_fov_folders = []
        for fov in fov_folders:
            fov_basename = os.path.basename(fov)
            # Look for any file starting with the fov_basename in unhuddle_norm_dir.
            pattern = os.path.join(unhuddle_norm_dir, f"{fov_basename}*")
            matching_outputs = glob.glob(pattern)
            if matching_outputs:
                logger.info(f"Skipping FOV '{fov}' as output already exists: {matching_outputs}")
            else:
                filtered_fov_folders.append(fov)
        fov_folders = filtered_fov_folders

    if not fov_folders:
        logger.error("No FOV folders to process. Please check the --fovs flag, base_path, or existing outputs.")
        return

    results = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_fov = {
            executor.submit(
                process_fov_pipeline,
                fov,
                morph_features_dir,
                protein_features_dir,
                original_sum_dir,
                original_norm_dir,
                unhuddle_sum_dir,
                unhuddle_norm_dir,
                args.create_nuclear_mask,
                args.create_deepcell_mask,
                args.geckodriver_path,
                args.deepcell_url,
                args.mask_pattern,
                args.markers_for_normalisation,
                args.red_markers,
                args.green_markers,
                args.blue_markers
            ): fov for fov in fov_folders
        }
        for future in tqdm(as_completed(future_to_fov), total=len(fov_folders), desc="Processing FOVs"):
            fov = future_to_fov[future]
            try:
                res = future.result()
                results[fov] = res
            except Exception as e:
                logger.error(f"Error processing FOV {fov}: {e}")
                results[fov] = {"fov": fov, "error": str(e)}
            # ----- Summarize FOVs with Errors -----
    errored_fovs = []

    for fov, result in results.items():
        if result is None:
            errored_fovs.append(fov)
            continue

        # FOV failed if it had a critical error or never reached normalization
        if (
            "critical_error" in result
            or result.get("normalized_intensity") is not True
        ):
            errored_fovs.append(fov)

    if errored_fovs:
        print("\n⚠️ FOVs with errors during processing:")
        for fov in errored_fovs:
            print(f"- {fov}")
    else:
        print("\n✅ All FOVs processed successfully.")

	


if __name__ == "__main__":
    main()
