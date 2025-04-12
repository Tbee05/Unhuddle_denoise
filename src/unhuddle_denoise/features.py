# src/unhuddle/features.py





import os
import numpy as np
import pandas as pd
from skimage import io, measure
import glob
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)

def extract_morphology_features(fov_folder, morph_features_dir, cell_mask, files, nuclear_markers, nuclear_mask=None):
    """
    Extract features from a FOV. If cell_mask and/or nuclear_mask are provided,
    they will be used; otherwise, the function will load them from disk.
    Returns a tuple (features_list, area_lookup) where area_lookup is a dictionary mapping
    each cell label to its Area.
    """

    features_list = []
    try:
        fov_name = os.path.basename(fov_folder)
        logger.info(f"Extracting features for FOV: {fov_name}")

        # Load nuclear marker images dynamically
        nuclear_images = {}
        for marker in nuclear_markers:
            if marker not in files or len(files[marker]) == 0:
                logger.warning(f"Nuclear marker {marker} not found in files for {fov_folder}")
                continue
            img = io.imread(files[marker][0]).astype(np.float32)
            nuclear_images[marker] = img

        if not nuclear_images:
            logger.error(f"No nuclear markers were loaded for {fov_folder}")
            return ([], {})

        cell_regions = measure.regionprops(cell_mask)
        if not cell_regions:
            logger.error(f"No cell regions found for FOV {fov_name}. Skipping.")
            return (features_list, {})

        if nuclear_mask is not None:
            # Use the first available nuclear image just for intensity mapping (arbitrary for shape only)
            reference_marker = next(iter(nuclear_images.values()))
            nuclear_regions = {
                region.label: region
                for region in measure.regionprops(nuclear_mask, intensity_image=reference_marker)
            }
        else:
            nuclear_regions = {}

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

            # Default NaNs
            nucleus_area = nucleus_eccentricity = nc_area_ratio = np.nan
            nucleus_centroid_row = nucleus_centroid_col = centroid_deviation = np.nan
            marker_means = {}
            marker_integrated = {}

            if nuclear_mask is not None and label in nuclear_regions:
                nucleus_region = nuclear_regions[label]
                nucleus_area = nucleus_region.area
                nucleus_eccentricity = nucleus_region.eccentricity
                nc_area_ratio = nucleus_area / area if area > 0 else np.nan
                nucleus_centroid_row, nucleus_centroid_col = nucleus_region.centroid
                centroid_deviation = np.sqrt((nucleus_centroid_row - centroid_row) ** 2 +
                                             (nucleus_centroid_col - centroid_col) ** 2)

                for marker, image in nuclear_images.items():
                    sub_mask = (nuclear_mask == label).astype(np.uint8)
                    props = measure.regionprops_table(sub_mask, intensity_image=image, properties=["mean_intensity"])
                    mean_val = props["mean_intensity"][0] if "mean_intensity" in props and props["mean_intensity"] else np.nan
                    marker_means[marker] = mean_val
                    marker_integrated[marker] = mean_val * nucleus_area if not np.isnan(mean_val) else np.nan
            else:
                # If nuclear mask not available or label missing
                for marker in nuclear_markers:
                    marker_means[marker] = np.nan
                    marker_integrated[marker] = np.nan

            features = {
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
            }

            # Append marker-specific intensities
            for marker in nuclear_markers:
                features[f"Mean_{marker}_Intensity"] = marker_means.get(marker, np.nan)
                features[f"Integrated_{marker}_Intensity"] = marker_integrated.get(marker, np.nan)

            features_list.append(features)

        morph_features = pd.DataFrame(features_list)
        csv_path = os.path.join(morph_features_dir, f"{fov_name}.csv")
        morph_features.to_csv(csv_path, index=False)
        logger.info(f"Saved features for FOV {fov_name} to {csv_path}")
        return morph_features

    except Exception as e:
        logger.error(f"Error extracting features for FOV {fov_folder}: {e}", exc_info=True)
        return []


def extract_protein_intensity(fov_folder, protein_features_dir, morph_features, cell_mask,
                              membrane_mask, memexcl_mask, nuclear_markers):
    try:
        fov_name = os.path.basename(fov_folder)
        logger.info(f"Processing protein intensities for FOV: {fov_name}")

        # Extract Labels and Areas directly from morph_features
        if not {"Label", "Area"}.issubset(morph_features.columns):
            logger.error(f"Missing 'Label' or 'Area' columns in morph_features for {fov_name}, skipping.")
            return None

        labels = morph_features["Label"].values
        areas = morph_features["Area"].values

        # Get unique labels from the cell mask (excluding background)
        unique_labels = np.unique(cell_mask)
        unique_labels = unique_labels[unique_labels != 0]
        n_labels = len(unique_labels)

        logger.debug(f"Number of labels in cell mask: {n_labels}")
        if len(areas) != n_labels:
            logger.warning(f"Mismatch: {len(areas)} areas vs. {n_labels} labels for {fov_name}")

        # Initialize results dictionary
        results = {"FOV": [fov_name] * n_labels, "Label": unique_labels.tolist()}

        # Map nuclear marker names to expected .ome.tiff filenames (case-insensitive)
        excluded_names = set(m.lower() for m in nuclear_markers)

        # Grab all non-nuclear marker files
        ome_files = [
            f for f in glob.glob(os.path.join(fov_folder, "*.ome.tiff"))
            if os.path.basename(f).replace(".ome.tiff", "").lower() not in excluded_names
        ]

        for ome_file in ome_files:
            marker_name = os.path.basename(ome_file).replace(".ome.tiff", "")
            marker_image = io.imread(ome_file)

            # Compute summed intensities per region
            cell_sum = np.atleast_1d(ndimage.sum(marker_image, labels=cell_mask, index=unique_labels))
            mem_sum = np.atleast_1d(ndimage.sum(marker_image, labels=membrane_mask, index=unique_labels))
            memexcl_sum = np.atleast_1d(ndimage.sum(marker_image, labels=memexcl_mask, index=unique_labels))

            logger.debug(f"Marker {marker_name}: cell_sum: {len(cell_sum)}, "
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
        logger.info(f"Saved protein intensity summary for FOV: {fov_folder}")

        return protein_features  # Return the DataFrame

    except Exception as e:
        logger.error(f"Error in protein intensity extraction for FOV {fov_folder}: {e}")
        return None  # Return None in case of failure
