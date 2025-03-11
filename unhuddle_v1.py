#!/usr/bin/env python3
"""
Final Integrated Pipeline:
  For each FOV folder under the provided base path, the pipeline performs these essential stages:

    1. Mask Processing and Interaction Computation:
         - Compute cell, nuclear, and membrane masks.
         - Compute border interactions (from membrane pixels) and background interactions (from background pixels).
         - Merge these interactions into a single dictionary (annotated with "type": "border" or "background").

    2. Feature Extraction:
         - Extract per-cell features (region properties) and save a CSV.

    3. Protein Intensity Extraction:
         - Compute protein intensity features (from non-DNA/Histone markers) and save a CSV.

    4. Object-Intensity Analysis:
         - Group the per-coordinate interactions into a per-object (cell) dictionary and reallocate intensities based on mean intensity data.

    5. Intensity Settlement:
         - Load morphological and protein intensity summary CSVs, update intensities using the per-object dictionary,
           and write out measure-type CSV files (e.g. corrected_mean, corrected_sum).

    6. Normalized Intensity Calculation:
         - Compute "area blowup" on the fly from the cell mask and normalize the corrected-sum intensities
           (by dividing by the blowup area), then output a CSV.

Usage:
    ./integrated_pipeline.py --base_path /path/to/FOVs --output_base_path /path/to/output --max_workers 16
"""

import os
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




# Create subfolders for measure types.
MEASURE_TYPES = ["corrected_mean", "corrected_sum", "original_mean", "original_sum"]


# ---------------------- Logging & Shutdown ---------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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
def load_fov_files(fov_folder):
    files = {
        "mask": glob.glob(os.path.join(fov_folder, '*mask_0.tiff')),
        "dna2": glob.glob(os.path.join(fov_folder, '*DNA2.ome.tiff')),
        "dna1": glob.glob(os.path.join(fov_folder, '*DNA1.ome.tiff')),
        "histoneh3": glob.glob(os.path.join(fov_folder, '*HistoneH3.ome.tiff'))
    }
    return files


# @debug_log
def save_image(path, image, description=""):
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
        logging.warning(f"Restoring {len(missing_labels)} missing labels in exclusion mask.")

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
def compute_background_interactions(fov_folder):
    cellmask_output_path = os.path.join(fov_folder, 'deepcel_mask.tiff')
    if not os.path.exists(cellmask_output_path):
        logging.error(f"Missing cell mask in {fov_folder}.")
        return {}, Counter()
    cell_mask = io.imread(cellmask_output_path)
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
    merged = {}
    merged.update(border_int)
    merged.update(background_int)
    return merged


# @debug_log
def integrate_intensities_for_interactions(fov_folder, interactions):
    ome_tiff_files = glob.glob(os.path.join(fov_folder, '*.ome.tiff'))
    if not ome_tiff_files:
        logging.info(f"No OME-TIFF files found in {fov_folder}.")
        return interactions
    for ome_file in ome_tiff_files:
        marker_name = os.path.basename(ome_file).replace('.ome.tiff', '')
        ome_image = io.imread(ome_file)
        for coord_str, data in interactions.items():
            y, x = map(int, coord_str.split('_'))
            if 0 <= y < ome_image.shape[0] and 0 <= x < ome_image.shape[1]:
                intensity_value = float(ome_image[y, x])
            else:
                intensity_value = None
            data.setdefault('intensities', {})[marker_name] = intensity_value
    return interactions

# @debug_log

import pandas as pd
from collections import defaultdict

def compute_reallocation(interactions, protein_features):
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
    # Precompute a dictionary of (cell, marker) → mean_exclusion_membrane_intensity
    # to speed up repeated lookups.
    mean_intensities = {}
    for cell_idx in protein_features.index:
        for col in protein_features.columns:
            if col.endswith("_ExclusionMembrane_Mean_Intensity"):
                # Derive the marker name by stripping the known suffix
                # e.g. marker -> col.replace("_ExclusionMembrane_Mean_Intensity", "")
                marker_name = col.replace("_ExclusionMembrane_Mean_Intensity", "")
                val = protein_features.at[cell_idx, col]
                if pd.isna(val):
                    val = 0
                mean_intensities[(cell_idx, marker_name)] = val

    # Our final data structure, storing taken and reallocated intensities
    reallocation = defaultdict(lambda: {
        "taken_intensity": defaultdict(float),
        "reallocated_intensity": defaultdict(float)
    })

    # ----------------------------------
    # 1) GROUP & PROCESS BORDER INTERACTIONS
    # ----------------------------------
    border_groups = defaultdict(list)
    for coord, data in interactions.items():
        if data.get("type") == "border" and "intensities" in data:
            key = (data["current"], tuple(sorted(data.get("neighbors", []))))
            border_groups[key].append(coord)

    for (current, neighbors), coords in border_groups.items():
        # Build the union of all marker sets across the coords in this group
        marker_set = set()
        for coord in coords:
            marker_set.update(interactions[coord]["intensities"].keys())

        for marker in marker_set:
            # Skip if marker name contains 'DNA' or 'Histone'
            if "DNA" in marker or "Histone" in marker:
                continue

            # Sum total intensities for this marker in all coords of this group
            total_intensity = 0.0
            for coord in coords:
                total_intensity += interactions[coord]["intensities"].get(marker, 0)

            # 1) Tally up "taken_intensity" for the CURRENT cell
            reallocation[current]["taken_intensity"][marker] += total_intensity

            # 2) Redistribute that same total based on the mean intensities
            current_mean = mean_intensities.get((current, marker), 0)
            neighbor_means = [mean_intensities.get((n, marker), 0) for n in neighbors]

            total_mean = current_mean + sum(neighbor_means)
            if total_mean == 0:
                # Fallback: everything is reallocated to the current cell if
                # nobody has a non-zero mean
                reallocation[current]["reallocated_intensity"][marker] += total_intensity
            else:
                # Distribute only to cells that have > 0 mean
                labels_nonzero = []
                intensities_nonzero = []
                if current_mean > 0:
                    labels_nonzero.append(current)
                    intensities_nonzero.append(current_mean)
                for n, mval in zip(neighbors, neighbor_means):
                    if mval > 0:
                        labels_nonzero.append(n)
                        intensities_nonzero.append(mval)

                # If no cell has a positive mean, (which can't happen here because total_mean != 0),
                # you'd skip. Otherwise:
                sum_nonzero = sum(intensities_nonzero)
                for lab, mval in zip(labels_nonzero, intensities_nonzero):
                    factor = mval / sum_nonzero
                    reallocation[lab]["reallocated_intensity"][marker] += factor * total_intensity

    # ----------------------------------
    # 2) GROUP & PROCESS BACKGROUND INTERACTIONS
    # ----------------------------------
    background_groups = defaultdict(list)
    for coord, data in interactions.items():
        if data.get("type") == "background" and "intensities" in data:
            key = tuple(sorted(data.get("interacts_with", [])))
            background_groups[key].append(coord)

    for cells, coords in background_groups.items():
        # Build the union of marker sets for these coords
        marker_set = set()
        for coord in coords:
            marker_set.update(interactions[coord]["intensities"].keys())

        for marker in marker_set:
            if "DNA" in marker or "Histone" in marker:
                continue

            total_intensity = 0.0
            for coord in coords:
                total_intensity += interactions[coord]["intensities"].get(marker, 0)

            # Distribute among the cells that appear in `cells`
            if len(cells) == 1:
                # Only one cell claims these background coords
                cell = cells[0]
                cell_mean = mean_intensities.get((cell, marker), 0)
                if cell_mean > 0:
                    reallocation[cell]["reallocated_intensity"][marker] += total_intensity
                # If cell_mean == 0, that intensity effectively goes nowhere
            else:
                # Multiple cells compete for the background coords
                nonzero_labels = []
                nonzero_means = []
                for cell in cells:
                    val = mean_intensities.get((cell, marker), 0)
                    if val > 0:
                        nonzero_labels.append(cell)
                        nonzero_means.append(val)

                if nonzero_labels:
                    sum_nonzero = sum(nonzero_means)
                    for lab, mval in zip(nonzero_labels, nonzero_means):
                        factor = mval / sum_nonzero
                        reallocation[lab]["reallocated_intensity"][marker] += factor * total_intensity
                # If no cell has a positive mean, nothing is allocated -> intensities vanish

    # Convert top-level defaultdict to a normal dict for final return
    return dict(reallocation)



@debug_log
def settle_debts_intensity(fov_folder, reallocation, morph_features, protein_features,
                           original_sum_dir, original_mean_dir, corrected_sum_dir, corrected_mean_dir):
    """
    Applies intensity adjustments from 'reallocation' to 'protein_features',
    saves original and corrected sum/mean intensities, and returns `corrected_sum_df`.

    The function:
       1) Saves original sum/mean intensities before settlement.
       2) Subtracts 'taken_intensity' from the original sum.
       3) Adds 'reallocated_intensity' from redistribution.
       4) Computes corrected means = corrected_sum / cell_area.
       5) Saves both original and corrected values in structured CSVs.
       6) Returns `corrected_sum_df`.

    Assumes:
       - `protein_features` is indexed by (FOV, Label).
       - `morph_features` contains "Label" and "Area".
       - `reallocation` contains 'taken_intensity' and 'reallocated_intensity'.
    """
    fov_name = os.path.basename(fov_folder)

    # Ensure output directories exist
    os.makedirs(original_sum_dir, exist_ok=True)
    os.makedirs(original_mean_dir, exist_ok=True)
    os.makedirs(corrected_sum_dir, exist_ok=True)
    os.makedirs(corrected_mean_dir, exist_ok=True)

    # Ensure protein_features is indexed by (FOV, Label)
    if not isinstance(protein_features.index, pd.MultiIndex) or protein_features.index.names != ["FOV", "Label"]:
        logging.info(f"[{fov_name}] Resetting index on protein_features to ensure (FOV, Label) multi-index.")
        protein_features = protein_features.set_index(["FOV", "Label"])

    # Ensure morph_features has required columns
    if not {"Label", "Area"}.issubset(morph_features.columns):
        logging.error(f"[{fov_name}] Morphology features missing required columns. Skipping intensity settlement.")
        return None

    # Build area lookup dictionary
    area_lookup = dict(zip(morph_features["Label"], morph_features["Area"]))

    if not area_lookup:
        logging.error(f"[{fov_name}] Area lookup is empty. Skipping intensity settlement.")
        return None

    logging.info(f"[{fov_name}] Starting intensity settlement.")

    # Debugging: Print available columns before renaming
    logging.debug(f"[{fov_name}] Columns before renaming: {list(protein_features.columns)}")

    # Correct mapping: from full column name to marker (e.g. "CD11b_Cell_Sum_Intensity" -> "CD11b")
    original_sums = {col: col.replace("_Cell_Sum_Intensity", "") 
                     for col in protein_features.columns if col.endswith("_Cell_Sum_Intensity")}
    original_means = {col: col.replace("_Cell_Mean_Intensity", "") 
                      for col in protein_features.columns if col.endswith("_Cell_Mean_Intensity")}

    # Get only the markers that are present (the new marker names)
    available_markers = sorted(set(original_sums.values()) & set(original_means.values()))

    if not available_markers:
        logging.error(f"[{fov_name}] No valid markers found in protein_features! Available columns: {list(protein_features.columns)}")
        return None

    logging.debug(f"[{fov_name}] Available markers after filtering: {available_markers}")

    # Select original sum and mean intensities before modification
    try:
        original_sum_df = protein_features.reset_index().drop(columns=["FOV"], errors="ignore")[
            ["Label"] + list(original_sums.keys())]
        original_mean_df = protein_features.reset_index().drop(columns=["FOV"], errors="ignore")[
            ["Label"] + list(original_means.keys())]
    except KeyError as e:
        logging.error(f"[{fov_name}] Error selecting columns: {e}")
        logging.error(f"[{fov_name}] Available columns: {list(protein_features.columns)}")
        return None

    # Rename columns: mapping from full column name to marker name
    original_sum_df = original_sum_df.rename(columns=original_sums)
    original_mean_df = original_mean_df.rename(columns=original_means)

    # Ensure consistency in marker ordering (Label followed by markers in alphabetical order)
    original_sum_df = original_sum_df[['Label'] + available_markers]
    original_mean_df = original_mean_df[['Label'] + available_markers]

    # Save original CSVs
    original_sum_df.to_csv(os.path.join(original_sum_dir, f"{fov_name}.csv"), index=False)
    original_mean_df.to_csv(os.path.join(original_mean_dir, f"{fov_name}.csv"), index=False)

    # Apply reallocation to update intensities
    for label, data in reallocation.items():
        # If label is a string that represents an integer, convert it to int; otherwise, keep as is.
        parsed_label = int(label) if isinstance(label, str) and label.isdigit() else label
        key = (fov_name, parsed_label)
        if key not in protein_features.index:
            logging.warning(f"[{fov_name}] Skipping missing cell: {key}")
            continue
        for marker, taken_val in data.get("taken_intensity", {}).items():
            sum_col = f"{marker}_Cell_Sum_Intensity"
            if sum_col in protein_features.columns:
                protein_features.at[key, sum_col] -= taken_val
        for marker, realloc_val in data.get("reallocated_intensity", {}).items():
            sum_col = f"{marker}_Cell_Sum_Intensity"
            if sum_col in protein_features.columns:
                protein_features.at[key, sum_col] += realloc_val

    # Debugging: Print available columns before renaming corrected DataFrames
    logging.debug(f"[{fov_name}] Columns before renaming corrected: {list(protein_features.columns)}")

    # Select corrected sum and mean intensities
    try:
        corrected_sum_df = protein_features.reset_index().drop(columns=["FOV"], errors="ignore")[
            ["Label"] + list(original_sums.keys())]
        corrected_mean_df = protein_features.reset_index().drop(columns=["FOV"], errors="ignore")[
            ["Label"] + list(original_means.keys())]
    except KeyError as e:
        logging.error(f"[{fov_name}] Error selecting columns after correction: {e}")
        logging.error(f"[{fov_name}] Available columns: {list(protein_features.columns)}")
        return None

    # Rename columns
    corrected_sum_df = corrected_sum_df.rename(columns=original_sums)
    corrected_mean_df = corrected_mean_df.rename(columns=original_means)

    # Debugging: Print after renaming
    logging.debug(f"[{fov_name}] Columns after renaming in corrected_sum_df: {list(corrected_sum_df.columns)}")

    # Ensure consistent column ordering
    corrected_sum_df = corrected_sum_df[['Label'] + available_markers]
    corrected_mean_df = corrected_mean_df[['Label'] + available_markers]

    # Save corrected CSVs
    corrected_sum_df.to_csv(os.path.join(corrected_sum_dir, f"{fov_name}.csv"), index=False)
    corrected_mean_df.to_csv(os.path.join(corrected_mean_dir, f"{fov_name}.csv"), index=False)

    logging.info(f"[{fov_name}] Intensity settlement completed.")

    return corrected_sum_df


# ---------------------- Stage 6: Normalized Intensity Calculation ---------------------- #

@debug_log
def compute_area_blowup(cell_mask):
    labels = np.unique(cell_mask)
    labels = labels[labels > 0]  # Exclude background (0)

    if len(labels) == 0:
        logging.warning("compute_area_blowup: No cell labels found in mask.")
        return pd.DataFrame(columns=["Label", "Initial_Area", "Blowup_Area"])  # Return empty DataFrame

    blowup = []
    for label in labels:
        label_mask = (cell_mask == label)
        initial_area = np.sum(label_mask)
        blownup_mask = dilation(label_mask, disk(1))
        blownup_area = np.sum(blownup_mask)

        blowup.append({'Label': int(label), 'Initial_Area': int(initial_area), 'Blowup_Area': int(blownup_area)})

    return pd.DataFrame(blowup)


@debug_log
def compute_normalized_intensities_for_fov(fov_folder, cell_mask, corrected_sum_df, normalized_output_dir):
    """
    Computes normalized intensities from the corrected sum intensities DataFrame.
    Normalized intensity = corrected_sum / blowup_area.

    Args:
        fov_folder (str): Path to the FOV folder.
        cell_mask (np.array): Cell mask for this FOV.
        corrected_sum_df (pd.DataFrame): Final sum intensity DataFrame (from settle_debts_intensity).
        normalized_output_dir (str): Output directory for normalized intensities.
    """

    fov_name = os.path.basename(fov_folder)

    # Compute blowup areas
    area_df = compute_area_blowup(cell_mask)

    # Explicitly check if area_df is None or empty
    if area_df is None or area_df.empty:
        logging.error(f"Skipping {fov_name} due to missing or empty mask.")
        return

    # Ensure 'Label' is an integer type for merging
    area_df["Label"] = area_df["Label"].astype(int)

    # Ensure 'Label' is present in corrected_sum_df
    if "Label" not in corrected_sum_df.columns:
        logging.error(f"Skipping {fov_name}: 'Label' column missing in corrected sum intensities.")
        return

    # Ensure 'Label' is an integer
    corrected_sum_df["Label"] = corrected_sum_df["Label"].astype(int)

    # Merge corrected sum intensities with area DataFrame
    merged_df = corrected_sum_df.merge(area_df, on="Label", how="left")

    # Ensure 'Blowup_Area' is numeric
    merged_df["Blowup_Area"] = pd.to_numeric(merged_df["Blowup_Area"], errors="coerce")

    # Get marker intensity columns (excluding 'Label' and 'Blowup_Area')
    marker_columns = sorted([col for col in merged_df.columns if col not in ['Label', 'Blowup_Area']])

    # Normalize intensities: corrected_sum / blowup_area
    normalized_df = merged_df[['Label']].copy()
    for marker in marker_columns:
        # Ensure marker values are numeric
        merged_df[marker] = pd.to_numeric(merged_df[marker], errors="coerce")

        # Normalize: divide by Blowup_Area, avoiding division by zero
        normalized_df[marker] = merged_df[marker] / merged_df["Blowup_Area"].replace(0, np.nan)

    # Ensure marker columns are sorted alphabetically
    normalized_df = normalized_df[['Label'] + marker_columns]

    # Save to CSV
    os.makedirs(normalized_output_dir, exist_ok=True)
    out_csv = os.path.join(normalized_output_dir, f"{fov_name}.csv")
    normalized_df.to_csv(out_csv, index=False)
    logging.info(f"Saved normalized intensities for {fov_name} to {out_csv}")

# ---------------------- Combined FOV Pipeline ---------------------- #
@debug_log
def process_fov_pipeline(fov_folder, morph_features_dir, protein_features_dir,
                          normalized_output_dir, original_sum_dir, original_mean_dir, corrected_sum_dir, corrected_mean_dir,
                         create_nuclear_mask):
    result = {"fov": fov_folder}
    try:
        files = load_fov_files(fov_folder)
        cell_mask = process_cell_mask(fov_folder, files["mask"])
        # Optionally create the nuclear mask.
        if create_nuclear_mask:
            nuclear_mask = process_nuclear_mask(fov_folder, cell_mask, files)
        else:
            nuclear_mask = None
            logging.info("Skipping nuclear mask creation as per flag.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        cell_mask, nuclear_mask = None, None

    try:
        morph_features = extract_morphology_features(fov_folder, morph_features_dir, cell_mask, nuclear_mask)
    except Exception as e:
        logging.error(f"Error extracting features for {fov_folder}: {e}")

    try:
        membrane_mask, memexcl_mask = process_membrane_masks(fov_folder, cell_mask)
    except Exception as e:
        logging.error(f"Error creating membrane-related masks for {fov_folder}: {e}")

    try:
        protein_features = extract_protein_intensity(fov_folder, protein_features_dir,
                                                     morph_features,  # Pass full DataFrame
                                                     cell_mask, membrane_mask, memexcl_mask)
        result["protein_intensity_extracted"] = True
    except Exception as e:
        logging.error(f"Error in protein intensity extraction for {fov_folder}: {e}")
        result["protein_intensity_error"] = str(e)

    try:
        border_int = compute_border_interactions(cell_mask, membrane_mask)
        background_int, _ = compute_background_interactions(fov_folder)
        merged = merge_interactions(border_int, background_int)
        all_interactions = integrate_intensities_for_interactions(fov_folder, merged)
        reallocation = compute_reallocation(all_interactions, protein_features)
    except Exception as e:
        logging.error(f"Error in object intensity analysis for {fov_folder}: {e}")
        result["object_intensity_error"] = str(e)

    try:
        corrected_sum_df = settle_debts_intensity(fov_folder, reallocation, morph_features, protein_features, original_sum_dir, original_mean_dir, corrected_sum_dir, corrected_mean_dir)
        result["intensity_settled"] = True
    except Exception as e:
        logging.error(f"Error settling intensities for {fov_folder}: {e}")
        result["intensity_settlement_error"] = str(e)

    try:
        compute_normalized_intensities_for_fov(fov_folder, cell_mask, corrected_sum_df, normalized_output_dir)
        result["normalized_intensity"] = True
    except Exception as e:
        logging.error(f"Error computing normalized intensities for {fov_folder}: {e}")
        result["normalized_intensity_error"] = str(e)

    return result


# ---------------------- Main Entry Point ---------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Integrated pipeline to process FOV folders. Essential stages are run for each FOV."
    )
    parser.add_argument("--base_path", type=str, required=True, help="Base path containing FOV folders")
    parser.add_argument("--output_base_path", type=str, required=True, help="Base path for output")
    parser.add_argument("--max_workers", type=int, default=32, help="Number of parallel workers (default: 32)")
    parser.add_argument("--create_nuclear_mask", action="store_true", default=False,
                        help="Flag to create the nuclear mask")
    args = parser.parse_args()

    # Compute derived directories based on output_base_path.
    output_base_path = args.output_base_path
    morph_features_dir = os.path.join(output_base_path, "morphology_features")
    protein_features_dir = os.path.join(output_base_path, "protein_features")
    original_sum_dir = os.path.join(output_base_path, "original_sum")
    original_mean_dir = os.path.join(output_base_path, "original_mean")
    corrected_sum_dir = os.path.join(output_base_path, "corrected_sum")
    corrected_mean_dir = os.path.join(output_base_path, "corrected_mean")

    normalized_output_dir = os.path.join(output_base_path, "unhuddle_normalized")

    for d in [output_base_path, morph_features_dir, protein_features_dir, original_sum_dir, original_mean_dir, corrected_sum_dir, corrected_mean_dir, normalized_output_dir]:
        os.makedirs(d, exist_ok=True)

    for mtype in MEASURE_TYPES:
        os.makedirs(os.path.join(output_base_path, mtype), exist_ok=True)

    fov_folders = [os.path.join(args.base_path, fov) for fov in os.listdir(args.base_path)
                   if os.path.isdir(os.path.join(args.base_path, fov))]
    results = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_fov = {executor.submit(process_fov_pipeline, fov, morph_features_dir, protein_features_dir, normalized_output_dir, original_sum_dir, original_mean_dir, corrected_sum_dir, corrected_mean_dir,
                                         args.create_nuclear_mask): fov
                         for fov in fov_folders}
        for future in tqdm(as_completed(future_to_fov), total=len(fov_folders), desc="Processing FOVs"):
            fov = future_to_fov[future]
            try:
                res = future.result()
                results[fov] = res
            except Exception as e:
                logging.error(f"Error processing FOV {fov}: {e}")
                results[fov] = {"fov": fov, "error": str(e)}


if __name__ == "__main__":
    main()

