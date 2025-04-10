# src/unhuddle/pipeline.py

import os
import logging
logger = logging.getLogger(__name__)


from unhuddle.masks import process_cell_mask, process_nuclear_mask, process_membrane_masks, load_fov_files
from unhuddle.features import extract_morphology_features, extract_protein_intensity
from unhuddle.normalization import compute_normalized_intensities_for_fov
from unhuddle.interactions import (
    compute_border_interactions,
    compute_background_interactions,
    merge_interactions,
    integrate_intensities_for_interactions,
    compute_reallocation_with_checks,
    settle_debts_intensity
)
from unhuddle.deepcell import create_deepcell_mask_overlay, process_deepcell_overlay


def process_fov_pipeline(
    fov_path,
    morph_features_dir,
    protein_features_dir,
    original_sum_dir,
    original_norm_dir,
    unhuddle_sum_dir,
    unhuddle_norm_dir,
    create_nuclear_mask,
    create_deepcell_mask,
    geckodriver_path,
    deepcell_url,
    mask_pattern,
    markers_for_normalisation,
    nuclear_markers,
    blue_markers,
    log_level,
    deepcell_resolution,
    nuclear_markers_overlay,
    membrane_markers_overlay
):
    from unhuddle.cli import setup_logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.debug(f"[{os.getpid()}] Logging ready in subprocess for {fov_path}")
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
                    logger.error(
                        f"❌ DeepCell processing failed for {os.path.basename(fov_path)}: {e}",
                        exc_info=logger.isEnabledFor(logging.DEBUG)
                    )
                    result["deepcell_processing"] = f"Error: {e}"
                    return result
            else:
                logger.warning(
                    f"⚠️ Overlay file not created for {os.path.basename(fov_path)} — skipping FOV"
                )
                result["deepcell_processing"] = "Overlay file not created"
                return result

        try:
            files = load_fov_files(fov_path, nuclear_markers, mask_pattern)
        except ValueError as ve:
            logger.error(
                f"⚠️ Mask loading failed for {os.path.basename(fov_path)}: {ve}",
                exc_info=logger.isEnabledFor(logging.DEBUG)
            )
            result["critical_error"] = f"Mask load error: {ve}"
            return result

        cell_mask = process_cell_mask(fov_path, files["mask"])

        if create_nuclear_mask:
            nuclear_mask = process_nuclear_mask(fov_path, cell_mask, files, nuclear_markers)
        else:
            nuclear_mask = None
            logger.info("Skipping nuclear mask creation as per flag.")

    except Exception as e:
        logger.error(
            f"💥 Unexpected error during mask creation for {os.path.basename(fov_path)}: {e}",
            exc_info=logger.isEnabledFor(logging.DEBUG)
        )
        result["critical_error"] = f"Mask creation failed: {e}"
        return result

    try:
        morph_features = extract_morphology_features(fov_path, morph_features_dir, cell_mask, files, nuclear_markers, nuclear_mask)

    except Exception as e:
        logger.error(f"Error extracting features for {fov_path}: {e}")
        result["feature_extraction_error"] = str(e)

    try:
        membrane_mask, memexcl_mask = process_membrane_masks(fov_path, cell_mask)
    except Exception as e:
        logger.error(f"Error creating membrane-related masks for {fov_path}: {e}")
        result["membrane_mask_error"] = str(e)

    try:
        protein_features = extract_protein_intensity(
            fov_path, protein_features_dir, morph_features,
            cell_mask, membrane_mask, memexcl_mask, nuclear_markers
        )
        result["protein_intensity_extracted"] = True
    except Exception as e:
        logger.error(f"Error in protein intensity extraction for {fov_path}: {e}")
        result["protein_intensity_error"] = str(e)

    try:
        border_int = compute_border_interactions(cell_mask, membrane_mask)
        background_int, _ = compute_background_interactions(cell_mask)
        merged = merge_interactions(border_int, background_int)
        all_interactions = integrate_intensities_for_interactions(fov_path, merged)
        reallocation = compute_reallocation_with_checks(all_interactions, protein_features, tol=1e-6)
    except Exception as e:
        logger.error(f"Error in object intensity analysis for {fov_path}: {e}")
        result["object_intensity_error"] = str(e)

    try:
        original_sum_df, corrected_sum_df = settle_debts_intensity(
            fov_path, reallocation, protein_features, original_sum_dir, unhuddle_sum_dir
        )
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
            raise ValueError("Normalization (original) returned None")
    except Exception as e:
        logger.error(f"Error computing normalized intensities for {fov_path}: {e}")
        result["normalized_intensity_error"] = str(e)

    return result
