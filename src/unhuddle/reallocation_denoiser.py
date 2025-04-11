import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", message=".*partition.*MaskedArray.*")


def run_denoising_pipeline_on_dataframe(
    df: pd.DataFrame,
    markers: list[str],
    area_col: str = "Area",
    layer_suffix: str = "_ExclusionMembrane_Sum_Intensity",
    signal_anchor_x: float = 15,
    signal_anchor_y: float = 0,
    gridsize: int = 100,
    density_quantile: float = 0.1,
    min_cells_per_bin: int = 10,
    min_area: float = 15,
    store_metadata: bool = False,
) -> pd.DataFrame:
    """
    Adds denoised values for each marker into a DataFrame based on signal/noise cone fitting.
    Area values < `min_area` are clipped instead of excluded.
    Negative residuals are clipped to zero before final area regression.
    """
    area = np.clip(df[area_col].values, min_area, None)
    denoised_df = df.copy()
    metadata = {}

    for marker in markers:
        intensity_col = f"{marker}{layer_suffix}"
        if intensity_col not in df.columns:
            continue

        intensity = df[intensity_col].values
        intensity = intensity.astype(np.float64)

        # Filter area >= min_area for apex inference only
        area_filt = area[area >= min_area]
        intensity_filt = intensity[area >= min_area]

        hb = plt.hexbin(area_filt, intensity_filt, gridsize=gridsize, bins='log', cmap='Greys')
        plt.close()

        counts = hb.get_array()
        xbins = hb.get_offsets()[:, 0]
        ybins = hb.get_offsets()[:, 1]

        density_thresh = np.quantile(counts, density_quantile)
        keep_mask = (counts > density_thresh) & (counts >= min_cells_per_bin)
        if not np.any(keep_mask):
            continue

        top_x = xbins[keep_mask]
        top_y = ybins[keep_mask]
        peak_idx = np.argmax(top_y)
        apex_area = top_x[peak_idx]
        apex_intensity = top_y[peak_idx]

        signal_slope = (apex_intensity - signal_anchor_y) / (apex_area - signal_anchor_x)
        signal_intercept = signal_anchor_y - signal_slope * signal_anchor_x
        signal_fit = signal_slope * area + signal_intercept

        noise_mask = (xbins > apex_area) & (counts > density_thresh) & (counts >= min_cells_per_bin)
        if np.any(noise_mask):
            X_noise = xbins[noise_mask].reshape(-1, 1)
            y_noise = ybins[noise_mask]
            X_noise_rel = X_noise - apex_area
            model = LinearRegression(fit_intercept=False).fit(X_noise_rel, y_noise)
            noise_slope = model.coef_[0]
        else:
            noise_slope = 0.02

        noise_fit = np.where(area > apex_area, noise_slope * (area - apex_area), 0)

        X = np.stack([signal_fit, noise_fit], axis=1)
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(intensity)
        X_clean = X[valid_mask]
        y_clean = intensity[valid_mask]

        full_model = LinearRegression().fit(X_clean, y_clean)
        alpha, beta = full_model.coef_
        intercept_model = full_model.intercept_

        signal_contrib = alpha * signal_fit
        noise_contrib = beta * noise_fit
        residuals = intensity - (signal_contrib + noise_contrib + intercept_model)

        # Clip negative residuals before area normalization
        residuals_clipped = np.clip(residuals, 0, None)

        # Final regression to remove residual area bias
        valid_res = ~np.isnan(residuals_clipped)
        X_area = area[valid_res].reshape(-1, 1)
        y_res = residuals_clipped[valid_res]

        area_model = LinearRegression().fit(X_area, y_res)
        gamma = area_model.coef_[0]
        intercept_area = area_model.intercept_

        # Remove area-dependent baseline
        final_denoised = residuals_clipped - (gamma * area + intercept_area)

        # Robust normalization: clip outliers and scale to [0, 1]
        vmin, vmax = np.percentile(final_denoised[~np.isnan(final_denoised)], [2, 98])
        if vmax > vmin:
            final_denoised = np.clip((final_denoised - vmin) / (vmax - vmin), 0, 1)
        else:
            final_denoised = np.zeros_like(final_denoised)

        denoised_df[f"{marker}_ExclusionMembrane_Denoised_Intensity"] = residuals
        denoised_df[f"{marker}_ExclusionMembrane_FinalDenoised_Intensity"] = final_denoised

        if store_metadata:
            metadata[marker] = {
                "apex_area": apex_area,
                "apex_intensity": apex_intensity,
                "signal_slope": signal_slope,
                "noise_slope": noise_slope,
                "alpha": alpha,
                "beta": beta,
                "intercept_model": intercept_model,
                "area_regression_coef": gamma,
                "area_regression_intercept": intercept_area
            }

    return (denoised_df, metadata) if store_metadata else denoised_df


def compute_denoised_reallocation_factors(protein_csv_paths, protein_features_dir):
    logger = logging.getLogger("unhuddle")

    morph_csv_paths = [
        path.replace("protein_features", "morphology_features")
        for path in protein_csv_paths
    ]

    all_fovs_data = []
    fov_ids = []

    for morph_path, protein_path in zip(morph_csv_paths, protein_csv_paths):
        try:
            morph_df = pd.read_csv(morph_path)
            protein_df = pd.read_csv(protein_path)

            if "Area" not in morph_df.columns:
                raise ValueError(f"'Area' column missing in {morph_path}")
            fov_name = os.path.splitext(os.path.basename(protein_path))[0]

            joint_df = morph_df[["Area"]].join(protein_df, how="inner")
            joint_df["fov"] = fov_name
            all_fovs_data.append(joint_df)
            fov_ids.append(fov_name)

        except Exception as e:
            logger.warning(f"⚠️ Skipping {protein_path}: {e}")

    if not all_fovs_data:
        raise RuntimeError("❌ No valid FOVs found to compute denoised reallocation factors.")

    full_df = pd.concat(all_fovs_data, axis=0, ignore_index=True)

    intensity_cols = [col for col in full_df.columns if col.endswith("_ExclusionMembrane_Sum_Intensity")]
    markers = [col.replace("_ExclusionMembrane_Sum_Intensity", "") for col in intensity_cols]

    logger.info("🚀 Running cohort-wide denoising on %d markers across %d FOVs...", len(markers), len(fov_ids))

    denoised_df = run_denoising_pipeline_on_dataframe(full_df, markers)

    for fov_name, group in denoised_df.groupby("fov"):
        denoised_cols = [col for col in group.columns if col.endswith("_ExclusionMembrane_Denoised_Intensity") or col.endswith("_ExclusionMembrane_FinalDenoised_Intensity")]
        denoised_block = group[denoised_cols].reset_index(drop=True)

        protein_csv_path = os.path.join(protein_features_dir, f"{fov_name}.csv")
        if not os.path.exists(protein_csv_path):
            logger.warning(f"⚠️ Protein CSV not found for FOV '{fov_name}', skipping update.")
            continue

        try:
            protein_df = pd.read_csv(protein_csv_path)
            updated_df = pd.concat([protein_df, denoised_block], axis=1)
            updated_df.to_csv(protein_csv_path, index=False)
            logger.info(f"📝 Appended denoised values to protein CSV for FOV: {fov_name}")
        except Exception as e:
            logger.error(f"❌ Failed to update {protein_csv_path}: {e}")
