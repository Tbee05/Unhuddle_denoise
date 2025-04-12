# src/unhuddle/adata.py
import os
import glob
import numpy as np
import pandas as pd
import warnings
import concurrent.futures
from tqdm import tqdm
from anndata import AnnData, concat
from tifffile import imread
import matplotlib
matplotlib.use("Agg")  # ✅ Set backend first
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", message=".*converted to numpy array with dtype.*")

def build_adata_from_outputs(output_base_path, working_path=None, output_adata_name="adata1.h5ad", max_workers=16):
    """Reconciles processed FOV outputs into a single AnnData object."""
    if working_path is None:
        raise ValueError("🛑 'working_path' must be explicitly provided — segmentation masks are required.")

    if not os.path.isdir(output_base_path):
        raise FileNotFoundError(f"🛑 Output directory not found: {output_base_path}")

    if not os.path.isdir(working_path):
        raise FileNotFoundError(f"🛑 Working path (input FOVs) not found: {working_path}")

    adata_output_path = os.path.join(output_base_path, "adata_objects", output_adata_name)
    qc_dir = os.path.join(output_base_path, "QC")

    os.makedirs(os.path.dirname(adata_output_path), exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    logger.info(f"[INFO] Saving output to: {adata_output_path}")
    logger.info(f"Creating QC figures in: {qc_dir}")

    def get_fov_list():
        files = glob.glob(f"{output_base_path}/unhuddle_normalized/*.csv")
        return [os.path.splitext(os.path.basename(f))[0] for f in files]

    def load_df(path, fov):
        df = pd.read_csv(path)
        df["Label"] = df["Label"].astype(int)
        df["cell_id"] = f"{fov}_" + df["Label"].astype(str)
        return df

    def convert_numeric(df, exclude=("Label", "cell_id")):
        numeric = df.select_dtypes(include=np.number).columns.difference(exclude)
        df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        return df

    def process_fov(fov):
        paths = {
            "intensity": f"{output_base_path}/unhuddle_normalized/{fov}.csv",
            "sum": f"{output_base_path}/unhuddle_sum/{fov}.csv",
            "orig_sum": f"{output_base_path}/original_sum/{fov}.csv",
            "morph": f"{output_base_path}/morphology_features/{fov}.csv",
            "denoised_intensity": f"{output_base_path}/unhuddle_denoised_normalized/{fov}.csv",
            "denoised_sum": f"{output_base_path}/unhuddle_denoised_sum/{fov}.csv",
        }
        if not all(os.path.exists(paths[k]) for k in ["intensity", "sum", "orig_sum", "morph"]):
            return None

        # Load and preprocess
        intensity = convert_numeric(load_df(paths["intensity"], fov))
        sum_unhuddle = convert_numeric(load_df(paths["sum"], fov))
        sum_orig = convert_numeric(load_df(paths["orig_sum"], fov))
        morph = convert_numeric(load_df(paths["morph"], fov))
        denoised_intensity = None
        denoised_sum = None
        if os.path.exists(paths["denoised_intensity"]):
            denoised_intensity = convert_numeric(load_df(paths["denoised_intensity"], fov))
        if os.path.exists(paths["denoised_sum"]):
            denoised_sum = convert_numeric(load_df(paths["denoised_sum"], fov))

        for df in [intensity, sum_unhuddle, sum_orig, morph, denoised_intensity, denoised_sum]:
            df.set_index("cell_id", inplace=True)

        # Morph QC
        if all(col in morph.columns for col in ["Nucleus_Area", "Nucleus_Centroid_Row"]):
            morph["QC_no_nucleus"] = morph[["Nucleus_Area", "Nucleus_Centroid_Row"]].isna().any(axis=1)

        morph["fov"] = fov
        morph["patient_id"] = fov.split("_")[0] if "_" in fov else fov

        # Build obsm
        obsm = {}
        if "Centroid_Row" in morph and "Centroid_Col" in morph:
            obsm["spatial"] = morph[["Centroid_Row", "Centroid_Col"]].values
        if "Nucleus_Centroid_Row" in morph and "Nucleus_Centroid_Col" in morph:
            obsm["nuclear_spatial"] = morph[["Nucleus_Centroid_Row", "Nucleus_Centroid_Col"]].values

        # Final cleanup
        morph.drop(columns=["FOV", "Label", "Centroid_Row", "Centroid_Col",
                            "Nucleus_Centroid_Row", "Nucleus_Centroid_Col"], errors="ignore", inplace=True)
        intensity.drop(columns=["Label"], errors="ignore", inplace=True)

        adata = AnnData(X=intensity, obs=morph, obsm=obsm)
        adata.layers["sum_unhuddle"] = sum_unhuddle.drop(columns=["Label"], errors="ignore").values
        adata.layers["sum_original"] = sum_orig.drop(columns=["Label"], errors="ignore").values
        if denoised_intensity is not None:
            adata.layers["denoised"] = denoised_intensity.drop(columns=["Label"], errors="ignore").values
        if denoised_sum is not None:
            adata.layers["sum_denoised"] = denoised_sum.drop(columns=["Label"], errors="ignore").values

        adata.obs["summed_intensity"] = adata.layers["sum_unhuddle"].sum(axis=1)

        return adata

    fovs = get_fov_list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        adatas = list(tqdm(pool.map(process_fov, fovs), total=len(fovs), desc="Constructing AnnData"))

    adatas = [a for a in adatas if a is not None]
    if not adatas:
        print("[ERROR] No valid FOVs found for AnnData.")
        return

    adata = concat(adatas, join="outer")
    adata.uns["fov-list"] = sorted(adata.obs["fov"].unique().tolist())
    adata.uns["patient_id-list"] = sorted(adata.obs["patient_id"].unique().tolist())
    adata.uns["marker-list"] = list(adata.var_names)

    # Load and attach segmentation masks
    adata.uns["spatial"] = {}
    for fov in fovs:
        mask_path = os.path.join(working_path, fov, "deepcel_mask.tiff")
        if os.path.exists(mask_path):
            adata.uns["spatial"][fov] = {"segmentation": imread(mask_path)}

    adata.write_h5ad(adata_output_path)
    print(f"AnnData saved to: {adata_output_path}\n\n")


