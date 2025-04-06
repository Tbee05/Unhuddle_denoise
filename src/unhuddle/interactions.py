# src/unhudde/interactions.py

import os
import glob
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from skimage import io


def compute_border_interactions(cell_mask, membrane_mask):
    interactions = defaultdict(dict)
    membrane_indices = np.argwhere(membrane_mask > 0)[:, :2]

    for y, x in membrane_indices:
        current = cell_mask[y, x]
        neighbors = set()
        for dy, dx in [(-1, 0), (0, -1), (0, 1), (1, 0),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < cell_mask.shape[0] and 0 <= nx < cell_mask.shape[1]:
                n = cell_mask[ny, nx]
                if n != 0 and n != current:
                    neighbors.add(n)
        if neighbors:
            interactions[f"{y}_{x}"] = {
                "current": int(current),
                "neighbors": sorted(map(int, neighbors)),
                "type": "border"
            }
    return interactions


def compute_background_interactions(cell_mask):
    interactions = {}
    background_indices = np.argwhere(cell_mask == 0)[:, :2]
    neighbor_count = Counter()

    for y, x in background_indices:
        labels = set()
        for dy, dx in [(-1, 0), (0, -1), (0, 1), (1, 0),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < cell_mask.shape[0] and 0 <= nx < cell_mask.shape[1]:
                n = cell_mask[ny, nx]
                if n != 0:
                    labels.add(n)
        if labels:
            interactions[f"{y}_{x}"] = {
                "current": "background",
                "interacts_with": sorted(map(int, labels)),
                "type": "background"
            }
            neighbor_count[len(labels)] += 1
    return interactions, neighbor_count


def merge_interactions(border, background):
    merged = {}
    merged.update(border)
    merged.update(background)
    return merged


def integrate_intensities_for_interactions(fov_folder, interactions):
    ome_tiffs = glob.glob(os.path.join(fov_folder, "*.ome.tiff"))

    for ome_file in ome_tiffs:
        marker = os.path.basename(ome_file).replace(".ome.tiff", "")
        image = io.imread(ome_file)

        for coord, data in interactions.items():
            try:
                y, x = map(int, coord.split("_"))
                value = float(image[y, x]) if 0 <= y < image.shape[0] and 0 <= x < image.shape[1] else None
            except:
                value = None

            if "intensities" not in data:
                data["intensities"] = {}
            data["intensities"][marker] = value

    return interactions

def compute_reallocation_with_checks(interactions, protein_features, tol=1e-6):
    logging.info("Reallocating intensities...")

    mean_intensity = {
        (row["Label"], col.replace("_ExclusionMembrane_Mean_Intensity", "")): row[col]
        for _, row in protein_features.iterrows()
        for col in row.index if col.endswith("_ExclusionMembrane_Mean_Intensity")
    }

    reallocation = defaultdict(lambda: {
        "taken_intensity": defaultdict(float),
        "reallocated_intensity": defaultdict(float)
    })

    for coord, data in interactions.items():
        if "intensities" not in data:
            continue
        interaction_type = data["type"]
        markers = [m for m in data["intensities"].keys() if "DNA" not in m and "Histone" not in m]
        coords = [coord]

        if interaction_type == "border":
            current = data["current"]
            neighbors = data["neighbors"]
            involved = [current] + neighbors
        else:  # background
            involved = data["interacts_with"]

        for marker in markers:
            values = [interactions[c]["intensities"].get(marker, 0.0) for c in coords]
            total = sum(values)
            weights = []
            for i in involved:
                val = mean_intensity.get((i, marker), 0.0)
                weights.append(max(val, 0))

            denom = sum(weights)
            if denom > 0:
                for i, w in zip(involved, weights):
                    frac = w / denom
                    reallocation[i]["reallocated_intensity"][marker] += frac * total
                    if interaction_type == "border" and i == current:
                        reallocation[i]["taken_intensity"][marker] += total
            else:
                if interaction_type == "border":
                    reallocation[current]["reallocated_intensity"][marker] += total
                    reallocation[current]["taken_intensity"][marker] += total
    return reallocation


def settle_debts_intensity(fov_folder, reallocation, protein_features,
                           original_sum_dir, unhuddle_sum_dir):
    fov_name = os.path.basename(fov_folder)

    if not isinstance(protein_features.index, pd.MultiIndex):
        protein_features = protein_features.set_index(["FOV", "Label"])

    sum_cols = [c for c in protein_features.columns if c.endswith("_Cell_Sum_Intensity")]
    markers = [c.replace("_Cell_Sum_Intensity", "") for c in sum_cols]

    orig_df = protein_features.reset_index()[["Label"] + sum_cols]
    orig_df.columns = ["Label"] + markers
    orig_df.to_csv(os.path.join(original_sum_dir, f"{fov_name}.csv"), index=False)

    for label, d in reallocation.items():
        key = (fov_name, label)
        for m in markers:
            col = f"{m}_Cell_Sum_Intensity"
            if col in protein_features.columns:
                protein_features.at[key, col] -= d["taken_intensity"].get(m, 0)
                protein_features.at[key, col] += d["reallocated_intensity"].get(m, 0)

    corr_df = protein_features.reset_index()[["Label"] + sum_cols]
    corr_df.columns = ["Label"] + markers
    corr_df.to_csv(os.path.join(unhuddle_sum_dir, f"{fov_name}.csv"), index=False)

    return orig_df, corr_df
