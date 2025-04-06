# src/unhuddle/normalization.py

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
