# src/unhuddle/cli.py

import os
import argparse
import glob
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm



def setup_logging(log_level):
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True
    )

    logger = logging.getLogger(__name__)
    logger.debug("Logging has been set up")

    if level == logging.DEBUG:
        # Enable verbose logging for key third-party libraries
        noisy_libs = ["selenium", "urllib3", "httpcore", "selenium.webdriver.remote.remote_connection"]
        for lib in noisy_libs:
            lib_logger = logging.getLogger(lib)
            lib_logger.setLevel(logging.DEBUG)
            lib_logger.propagate = True  # ensure logs reach root handlers
        logger.debug("Selenium and network library debug logging enabled")


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

    parser.add_argument("--nuclear-markers", nargs="+", default=None,
                        help="Markers used for red channel (nuclear): eg DNA1 DNA2 HistoneH3")
    parser.add_argument("--membrane-markers", nargs="+", default=None,
                        help="Markers used for green channel (membrane/cytoplasm) eg CD20 CD68 CD11b CD11c CD8a CD3 CD7 CD45RA CD45RO CD15 CD163 Vimentin CD31 CD14")
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
    parser.add_argument("--create_adata", action="store_true",
                        help="Integrates data from all FOVs in a single AnnData object")
    parser.add_argument(
        "--deepcell_resolution",
        type=int,
        choices=[10, 20, 40, 60, 100],
        default=10,
        help="Objective magnification to select in DeepCell UI (e.g., 10, 20, 40)"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logging.debug("Logging has been set up")
    logger = logging.getLogger("unhuddle")
    logger.debug("Logger '%s' is active at level: %s", logger.name, logger.level)

    from unhuddle.pipeline import process_fov_pipeline
    if args.create_deepcell_mask:
        if not args.nuclear_markers or not args.membrane_markers or not args.geckodriver_path:
            parser.error("--nuclear-markers, --green-markers and --geckodriver_path are required when --create_deepcell_mask is used.")

    if not args.list_available_markers and not args.markers_for_normalisation:
        parser.error("--markers_for_normalisation is required unless --list_available_markers is used.")

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

        marker_names = sorted([
            os.path.basename(f).replace(".ome.tiff", "")
            for f in ome_files
        ])

        print(f"\nAvailable markers in FOV '{os.path.basename(first_fov)}':")
        print("list:", " ".join(marker_names))
        print("\n✅ rerun without --available_markers to start the pipeline")
        print()
        return
    out = args.output_base_path
    dirs = {
        "morph": os.path.join(out, "morphology_features"),
        "protein": os.path.join(out, "protein_features"),
        "original_sum": os.path.join(out, "original_sum"),
        "original_norm": os.path.join(out, "original_normalized"),
        "unhuddle_sum": os.path.join(out, "unhuddle_sum"),
        "unhuddle_norm": os.path.join(out, "unhuddle_normalized"),
        "adata": os.path.join(out, "adata_objects")
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

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
                args.nuclear_markers,
                args.membrane_markers,
                args.blue_markers,
                args.log_level,
                args.deepcell_resolution
            ): fov for fov in fov_folders
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing FOVs"):
            fov = futures[future]
            try:
                results[fov] = future.result()
            except Exception as e:
                logging.error(f"❌ FOV {fov} failed: {e}")
                results[fov] = {"error": str(e)}

    errored = [os.path.basename(fov) for fov, res in results.items() if "error" in res or "critical_error" in res]
    if errored:
        print("\n⚠️ FOVs with errors:\n", "\n".join(errored))
    else:
        print("\n✅ All FOVs processed successfully.")
        print("📁 FOV folders updated with masks and overlays, tip: inspect pseudocolored mask renders.")
        print(f"📄 Unhuddle normalized is ready for phenotyping: {dirs["unhuddle_norm"]}")
        print(f"📄 Cell-level morphology metrics: {dirs["morph"]}")
        print(f"📄 Values before Unhuddle correction and/or before normalization: {out}")
        print()

    # --- Optional: Create adata object ---
    if args.create_adata:
        logging.info("\n📦 Creating unified AnnData object...")

        from unhuddle.adata_builder import build_adata_from_outputs


        try:
            build_adata_from_outputs(
                output_base_path=args.output_base_path,
                working_path=args.base_path,
                output_adata_name="adata1.h5ad",
                max_workers=args.max_workers
            )
        except Exception as e:
            logging.error(f"❌ Adata creation failed: {e}")
            print(f"[ERROR] Adata creation failed: {e}")
