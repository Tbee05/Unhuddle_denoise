# src/unhuddle/masks.py

import os
import glob
import numpy as np
from skimage import io, measure, morphology
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_fill_holes
from unhuddle.utils import save_image, generate_pseudocolor_mask
import logging
logger = logging.getLogger(__name__)



def load_fov_files(fov_folder, nuclear_markers=None, mask_patterns=["*_0.tiff"]):
    """
    Load mask and marker files for a single FOV.

    Args:
        fov_folder (str): Path to the FOV folder.
        nuclear_markers (list[str]): List of marker names to look for.
        mask_patterns (list[str]): Glob patterns for mask files (e.g. ["*_0.tiff"]).

    Returns:
        dict: {
            "mask": [path],
            "<marker_name>": [file] for each nuclear marker found
        }
    """
    # Find mask files
    mask_files = []
    for pattern in mask_patterns:
        mask_files.extend(glob.glob(os.path.join(fov_folder, pattern)))

    if len(mask_files) != 1:
        raise ValueError(f"Expected exactly 1 mask file in {fov_folder}, found {len(mask_files)}: {mask_files}")

    files = {"mask": mask_files}

    # Find nuclear marker files
    if nuclear_markers:
        for marker in nuclear_markers:
            marker_glob = os.path.join(fov_folder, f"{marker}.ome.tiff")
            matched = glob.glob(marker_glob)
            files[marker] = matched  # list, even if empty

    return files


def process_cell_mask(fov_folder, mask_files):
    cell_mask = io.imread(mask_files[0])
    if len(cell_mask.shape) > 2:
        cell_mask = np.squeeze(cell_mask)
    cell_mask = cell_mask.astype(np.uint16)

    save_image(os.path.join(fov_folder, 'deepcel_mask.tiff'), cell_mask, "cell mask")
    pseudocolor = generate_pseudocolor_mask(cell_mask)
    save_image(os.path.join(fov_folder, 'deepcel_mask_pseudocolor.png'), pseudocolor, "cell mask (pseudocolor)")

    return cell_mask


def process_nuclear_mask(fov_folder, cell_mask, files, nuclear_markers):
    # Load and sum all nuclear marker channels
    nuclear_signal = None
    for marker in nuclear_markers:
        if marker not in files or not files[marker]:
            raise ValueError(f"Nuclear marker '{marker}' not found in files for {fov_folder}")
        img = io.imread(files[marker][0]).astype(np.float32)
        nuclear_signal = img if nuclear_signal is None else nuclear_signal + img

    nuclear_mask = np.zeros_like(cell_mask, dtype=np.uint16)
    for label in np.unique(cell_mask):
        if label == 0:
            continue
        region = (cell_mask == label)
        signal = nuclear_signal * region
        binary = signal > 0
        filled = binary_fill_holes(binary)
        filled = morphology.remove_small_holes(filled, area_threshold=64)
        labeled = measure.label(filled)
        props = measure.regionprops(labeled, intensity_image=signal)
        if len(props) > 1:
            largest = max(props, key=lambda p: p.area)
            filled = labeled == largest.label
        nuclear_mask[filled] = label

    save_image(os.path.join(fov_folder, 'filled_nucmask.tiff'), nuclear_mask, "nuclear mask")
    save_image(os.path.join(fov_folder, 'filled_nucmask_pseudocolor.png'), generate_pseudocolor_mask(nuclear_mask), "nuclear pseudocolor")
    return nuclear_mask


def process_membrane_masks(fov_folder, cell_mask):
    membrane_mask = find_boundaries(cell_mask, mode='inner').astype(np.uint16)
    membrane_labeled = np.where(membrane_mask > 0, cell_mask, 0).astype(np.uint16)

    save_image(os.path.join(fov_folder, 'membrane_mask.tiff'), membrane_labeled, "membrane mask")
    save_image(os.path.join(fov_folder, 'membrane_mask_pseudocolor.png'),
               generate_pseudocolor_mask(membrane_labeled), "membrane pseudocolor")

    exclusion_mask = np.where(membrane_mask > 0, 0, cell_mask)
    # fallback: restore labels that disappeared
    missing_labels = set(np.unique(cell_mask)) - set(np.unique(exclusion_mask))
    for label in missing_labels:
        if label != 0:
            exclusion_mask[cell_mask == label] = label

    save_image(os.path.join(fov_folder, 'membrane_exclusion_mask.tiff'), exclusion_mask, "membrane exclusion mask")
    save_image(os.path.join(fov_folder, 'membrane_exclusion_mask_pseudocolor.png'),
               generate_pseudocolor_mask(exclusion_mask), "membrane exclusion pseudocolor")

    return membrane_labeled, exclusion_mask
