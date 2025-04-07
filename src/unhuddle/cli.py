# src/unhuddle/cli.py

import os
import argparse
import glob
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from unhuddle.pipeline import process_fov_pipeline


def setup_logging(log_level):
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="UNHUDDLE: Deconvolute and normalize highly multiplex proteomics tissue data."
    )

    parser.add_argument("--base_path", type=str, required=True, help="Base path containing FOV folders")
    parser.add_argument("--output_base_path", type=str, required=True, help="Base path for output")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers (default: 1)")

    parser.add_argument("--create_nuclear_mask", action="store_true", default=False,
                        help="Create nuclear mask for morphology and N/C ratio")
    parser.add_argument("--create_deepcell_mask", action="store_true", default=False,
                        help="Run DeepCell web overlay + segmentation")

    parser.add_argument("--geckodriver_path", type=str, default="geckodriver", help="Path to geckodriver binary")
    parser.add_argument("--deepcell_url", type=str, default="http://www.deepcell.org", help="DeepCell website URL")

    parser.add_argument("--red-markers", nargs="+", default=["DNA1", "DNA2", "HistoneH3"],
                        help="Markers used for red channel (nuclear)")
    parser.add_argument("--green-markers", nargs="+", default=["CD3", "CD45", "Vimentin"],
                        help="Markers used for green channel (membrane/cytoplasm)")
    parser.add_argument("--blue-markers", nargs="+", default=[], help="Optional markers for blue channel")

    parser.add_argument("--fovs", nargs="*", default=None,
                        help="List of specific FOV folders to process")
    parser.add_argument("--mask_pattern", type=str, nargs="+", default=["*_0.tiff"],
                        help="Glob pattern for mask files (default '*_0.tiff')")

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING")
    parser.add_argument("--check_output_exist", action="store_true", default=False,
                        help="Skip FOVs if output already exists in normalization folder")
    parser.add_argument("--markers_for_normalisation", nargs="*", default=None,
                        help="Sensor markers to normalize functional markers (e.g. CD3 CD45 Vimentin)")
    parser.add_argument("--list_available_markers", action="store_true",
                        help="Print available marker names from first FOV")

    args = parser.parse_args()

    if not args.list_available_markers and not args.markers_for_normalisation:
        parser.error("--markers_for_normalisation is required unless --list_available_markers is used.")

    setup_logging(args.log_level)

    # --- Print markers and exit ---
    if args.list_available_markers:
        fov_folders = [
            os.path.join(args.base_path, fov)
            for fov in os.listdir(args.base_path)
            if os.path.isdir(os.path.join(args.base_path, fov))
        ]
        if not fov_folders:
            print("❌ No FOV folders found.")
            return

        first_fov = fov_folders[0]
        ome_files = glob.glob(os.path.join(first_fov, "*.ome.tiff"))
        if not ome_files:
            print("❌ No .ome.tiff files in first FOV.")
            return

        print(f"\n✅ Available markers in FOV '{os.path.basename(first_fov)}':\n")
        for f in sorted(ome_files):
            print(" ", os.path.basename(f).replace(".ome.tiff", ""))
        return

    # --- Create output dirs ---
    out = args.output_base_path
    dirs = {
        "morph": os.path.join(out, "morphology_features"),
        "protein": os.path.join(out, "protein_features"),
        "original_sum": os.path.join(out, "original_sum"),
        "original_norm": os.path.join(out, "original_normalized"),
        "unhuddle_sum": os.path.join(out, "unhuddle_sum"),
        "unhuddle_norm": os.path.join(out, "unhuddle_normalized")
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # --- Collect FOV folders ---
    all_fovs = [
        os.path.join(args.base_path, fov)
        for fov in os.listdir(args.base_path)
        if os.path.isdir(os.path.join(args.base_path, fov))
    ]

    fov_folders = all_fovs
    if args.fovs:
        fov_folders = [f for f in all_fovs if os.path.basename(f) in args.fovs]

    if args.check_output_exist:
        fov_folders = [
            f for f in fov_folders
            if not glob.glob(os.path.join(dirs["unhuddle_norm"], f"{os.path.basename(f)}*"))
        ]

    if not fov_folders:
        print("❌ No FOVs selected for processing.")
        return

    # --- Launch FOV jobs ---
    results = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_fov_pipeline,
                fov,
                dirs["morph"],
                dirs["protein"],
                dirs["original_sum"],
                dirs["original_norm"],
                dirs["unhuddle_sum"],
                dirs["unhuddle_norm"],
                args.create_nuclear_mask,
                args.create_deepcell_mask,
                args.geckodriver_path,
                args.deepcell_url,
                args.mask_pattern,
                args.markers_for_normalisation,
                args.red_markers,
                args.green_markers,
                args.blue_markers
            ): fov for fov in fov_folders
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing FOVs"):
            fov = futures[future]
            try:
                results[fov] = future.result()
            except Exception as e:
                logging.error(f"❌ FOV {fov} failed: {e}")
                results[fov] = {"error": str(e)}

    # --- Print summary ---
    errored = [os.path.basename(fov) for fov, res in results.items() if "error" in res or "critical_error" in res]
    if errored:
        print("\n⚠️ FOVs with errors:\n", "\n".join(errored))
    else:
        print("\n✅ All FOVs processed successfully.")
