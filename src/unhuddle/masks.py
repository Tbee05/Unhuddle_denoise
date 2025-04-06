# src/unhuddle/masks.py

import os
import glob
import logging
import numpy as np
from skimage import io, measure, morphology
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_fill_holes
from unhuddle.utils import save_image, generate_pseudocolor_mask


def load_fov_files(fov_folder, mask_patterns=["*_0.tiff"]):
    mask_files = []
    for pattern in mask_patterns:
        mask_files.extend(glob.glob(os.path.join(fov_folder, pattern)))

    if len(mask_files) != 1:
        raise ValueError(f"Expected 1 mask file in {fov_folder}, found {len(mask_files)}: {mask_files}")

    files = {
        "mask": mask_files,
        "dna2": glob.glob(os.path.join(fov_folder, '*DNA2.ome.tiff')),
        "dna1": glob.glob(os.path.join(fov_folder, '*DNA1.ome.tiff')),
        "histoneh3": glob.glob(os.path.join(fov_folder, '*HistoneH3.ome.tiff'))
    }
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


def process_nuclear_mask(fov_folder, cell_mask, files):
    dna1 = io.imread(files["dna1"][0])
    dna2 = io.imread(files["dna2"][0])
    h3 = io.imread(files["histoneh3"][0])
    nuclear_signal = dna1 + dna2 + h3

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
