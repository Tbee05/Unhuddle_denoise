# src/unhuddle/features.py

import os
import glob
import logging
import numpy as np
import pandas as pd
from skimage import io, measure, morphology
from scipy import ndimage


from unhuddle.utils import save_image, generate_pseudocolor_mask

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
