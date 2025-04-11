# src/unhuddle/pipeline.py

import os
import logging
logger = logging.getLogger(__name__)
from skimage import io
from unhuddle.masks import process_cell_mask, process_nuclear_mask, process_membrane_masks, load_fov_files
from unhuddle.features import extract_morphology_features, extract_protein_intensity
from unhuddle.normalization import compute_normalized_intensities_for_fov
from unhuddle.interactions import (
    compute_border_interactions,
    compute_background_interactions,
    merge_interactions,
    integrate_intensities_for_interactions,
    compute_reallocation_with_checks,
    settle_debts_intensity,
    settle_debts_from_residuals
)
from unhuddle.deepcell import create_deepcell_mask_overlay, process_deepcell_overlay

def process_fov_features_only(
    fov_path,
    morph_features_dir,
    protein_features_dir,
    create_nuclear_mask,
    create_deepcell_mask,
    geckodriver_path,
    deepcell_url,
    mask_pattern,
    nuclear_markers,
    blue_markers,
    log_level,
    deepcell_resolution,
    nuclear_markers_overlay,
    membrane_markers_overlay,
):
    from unhuddle.cli_helpers import setup_logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.debug(f"[{os.getpid()}] Starting feature extraction for {fov_path}")
    result = {"fov": fov_path}

    try:
        if create_deepcell_mask:
            overlay_file = create_deepcell_mask_overlay(
                fov_path,
                red_markers=nuclear_markers_overlay,
                green_markers=membrane_markers_overlay,
                blue_markers=blue_markers
            )
            result["deepcell_overlay_file"] = overlay_file
            if overlay_file:
                try:
                    process_deepcell_overlay(
                        overlay_file,
                        fov_path,
                        deepcell_url,
                        geckodriver_path,
                        deepcell_resolution
                    )
                    result["deepcell_processing"] = "Success"
                except Exception as e:
                    logger.error(f"❌ DeepCell overlay processing failed: {e}")
                    result["deepcell_processing"] = f"Error: {e}"
                    return result
            else:
                logger.error("❌ Overlay file not created.")
                result["deepcell_processing"] = "Overlay file not created"
                return result

        files = load_fov_files(fov_path, nuclear_markers, mask_pattern)
        if "mask" not in files or not files["mask"]:
            logger.error(f"❌ No segmentation mask found in {fov_path} (pattern: {mask_pattern})")
            result["feature_extraction_error"] = "No mask found"
            return result

        cell_mask = process_cell_mask(fov_path, files["mask"])

        if create_nuclear_mask:
            try:
                nuclear_mask = process_nuclear_mask(fov_path, cell_mask, files, nuclear_markers)
                result["nuclear_mask_created"] = True
            except Exception as e:
                logger.warning(f"⚠️ Nuclear mask generation failed: {e}")
                nuclear_mask = None
                result["nuclear_mask_created"] = False
        else:
            nuclear_mask = None
            result["nuclear_mask_created"] = False

        try:
            morph_features = extract_morphology_features(fov_path, morph_features_dir, cell_mask, files, nuclear_markers, nuclear_mask)
            result["morphology_extracted"] = True
        except Exception as e:
            logger.error(f"❌ Morphology feature extraction failed: {e}")
            result["feature_extraction_error"] = f"Morphology: {e}"
            return result

        try:
            membrane_mask, memexcl_mask = process_membrane_masks(fov_path, cell_mask)
            result["membrane_masks_generated"] = True
        except Exception as e:
            logger.error(f"❌ Membrane mask processing failed: {e}")
            result["feature_extraction_error"] = f"Membrane masks: {e}"
            return result

        try:
            protein_features = extract_protein_intensity(
                fov_path, protein_features_dir, morph_features,
                cell_mask, membrane_mask, memexcl_mask, nuclear_markers
            )
            result["protein_intensity_extracted"] = True
        except Exception as e:
            logger.error(f"❌ Protein intensity extraction failed: {e}")
            result["feature_extraction_error"] = f"Protein intensity: {e}"
            return result

    except Exception as e:
        logger.error(f"❌ General feature extraction failed for {fov_path}: {e}")
        result["feature_extraction_error"] = str(e)

    return result

def process_fov_reallocation_only(
    fov_path,
    protein_features,
    original_sum_dir,
    original_norm_dir,
    unhuddle_sum_dir,
    unhuddle_norm_dir,
    unhuddle_denoised_sum_dir,
    unhuddle_denoised_norm_dir,
    markers_for_normalisation,
    use_denoised,
    log_level
):
    from unhuddle.cli_helpers import setup_logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.debug(f"[{os.getpid()}] Starting reallocation for {fov_path}")
    result = {"fov": fov_path}

    try:

        cell_mask_path = os.path.join(fov_path, "deepcel_mask.tiff")
        if not os.path.exists(cell_mask_path):
            raise FileNotFoundError(f"deepcel_mask.tiff not found in {fov_path}")
        cell_mask = io.imread(cell_mask_path)
        logger.debug(f"🧪 Loaded cell_mask from disk with shape {cell_mask.shape} and dtype {cell_mask.dtype}")

        membrane_path = os.path.join(fov_path, "membrane_mask.tiff")
        if not os.path.exists(membrane_path):
            raise FileNotFoundError(f"membrane_mask.tiff not found in {fov_path}")
        membrane_mask = io.imread(membrane_path)
        logger.debug(f"🧪 Loaded membrane_mask from disk with shape {membrane_mask.shape} and dtype {membrane_mask.dtype}")

        border_int = compute_border_interactions(cell_mask, membrane_mask)
        background_int, _ = compute_background_interactions(cell_mask)
        merged = merge_interactions(border_int, background_int)
        all_interactions = integrate_intensities_for_interactions(fov_path, merged)

        reallocation = compute_reallocation_with_checks(
            all_interactions, protein_features, tol=1e-6, use_denoised=use_denoised
        )

        # Run both residual- and canonical-based intensity correction when using denoised mode
        if use_denoised:
            denoised_df = settle_debts_from_residuals(
                fov_folder=fov_path,
                reallocation=reallocation,
                protein_features=protein_features,
                output_dir=unhuddle_denoised_sum_dir,
                cell_mask=cell_mask,
                membrane_mask=membrane_mask,
                normalisation_dir=unhuddle_denoised_norm_dir,
                sensor_markers=markers_for_normalisation
            )
            result["intensity_settled_denoised"] = True

            # Save denoised_df directly
            fov_name = os.path.basename(fov_path)
            denoised_output_path = os.path.join(unhuddle_denoised_sum_dir, f"{fov_name}.csv")
            denoised_df.to_csv(denoised_output_path, index=False)

            fov_name, norm_denoised = compute_normalized_intensities_for_fov(
                fov_folder=fov_path,
                corrected_sum_df=denoised_df,
                sensor_markers=markers_for_normalisation
            )
            if norm_denoised is not None:
                output_file = os.path.join(unhuddle_denoised_norm_dir, f"{fov_name}.csv")
                norm_denoised.to_csv(output_file, index=False)
                result["normalized_intensity_denoised"] = True

        # Always also run canonical for comparison
        original_sum_df, corrected_sum_df = settle_debts_intensity(
            fov_path, reallocation, protein_features,
            original_sum_dir, unhuddle_sum_dir
        )
        result["intensity_settled"] = True

        fov_name, norm_corrected = compute_normalized_intensities_for_fov(
            fov_folder=fov_path,
            corrected_sum_df=corrected_sum_df,
            sensor_markers=markers_for_normalisation
        )
        if norm_corrected is not None:
            output_file = os.path.join(unhuddle_norm_dir, f"{fov_name}.csv")
            norm_corrected.to_csv(output_file, index=False)
            result["normalized_intensity"] = True

        _, norm_original = compute_normalized_intensities_for_fov(
            fov_folder=fov_path,
            corrected_sum_df=original_sum_df,
            sensor_markers=markers_for_normalisation
        )
        if norm_original is not None:
            output_file = os.path.join(original_norm_dir, f"{fov_name}.csv")
            norm_original.to_csv(output_file, index=False)
            result["normalized_original_intensity"] = True

    except Exception as e:
        logger.error(f"Reallocation failed for {fov_path}: {e}")
        result["reallocation_error"] = str(e)

    return result

