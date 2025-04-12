import os
import glob
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm


_LOGGING_INITIALIZED = False

def setup_logging(log_level: str) -> None:
    """
    Set up logging with the specified log level and enable detailed logging for key libraries when in DEBUG.
    Prevents reinitialization across FOV loop calls.
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    level = getattr(logging, log_level.upper(), logging.INFO)

    # Force root logger config
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True
    )

    # Ensure root logger has at least one handler
    root = logging.getLogger()
    if not root.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(stream)
    root.setLevel(level)

    # Propagate and configure all 'unhuddle.*' loggers
    for name in logging.root.manager.loggerDict:
        if name.startswith("unhuddle"):
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = True

    logger = logging.getLogger(__name__)
    logger.debug("🛠️ Logging system has been initialized")

    # Print summary of logger levels (optional diagnostics)
    for name in sorted(logging.root.manager.loggerDict):
        l = logging.getLogger(name)
        logger.debug(f"{name:40} level={logging.getLevelName(l.level)}")

    if level == logging.DEBUG:
        noisy_libs = [
            "selenium", "urllib3", "httpcore",
            "selenium.webdriver.remote.remote_connection"
        ]
        for lib in noisy_libs:
            lib_logger = logging.getLogger(lib)
            lib_logger.setLevel(logging.DEBUG)
            lib_logger.propagate = True

        # 🔇 Suppress matplotlib & PIL debug logs
        for noisy in [
            "matplotlib", "matplotlib.font_manager",
            "matplotlib.pyplot", "PIL", "PIL.Image"
        ]:
            lib_logger = logging.getLogger(noisy)
            lib_logger.setLevel(logging.WARNING)
            lib_logger.propagate = False

        logger.debug("📡 Verbose logging enabled for Selenium and network libraries (matplotlib and PIL suppressed)")

    _LOGGING_INITIALIZED = True



def parse_arguments() -> argparse.Namespace:
    """
    Parse and return the command line arguments.
    """
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

    parser.add_argument("--nuclear_markers", nargs="+", default=None,
                        help="Filter marker list for chromatin signal: eg DNA1 DNA2 HistoneH3")
    parser.add_argument(
        "--nuclear_markers_overlay",
        nargs="+",
        default=None,
        help="Markers to use for DeepCell overlay (red channel - nuclear). If not provided, defaults to --nuclear_markers."
    )
    parser.add_argument(
        "--membrane_markers_overlay",
        nargs="+",
        default=None,
        help="Markers to use for DeepCell overlay (green channel - membrane/cytoplasm). If not provided, defaults to --normalisation_markers."
    )
    parser.add_argument("--blue_markers", nargs="+", default=[], help="Optional markers for blue channel")

    parser.add_argument("--fovs", nargs="*", default=None,
                        help="List of specific FOV folders to process")
    parser.add_argument("--mask_pattern", type=str, nargs="+", default=["*_0.tiff"],
                        help="Glob pattern for mask files (default '*_0.tiff')")

    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING")
    parser.add_argument("--check_output_exist", action="store_true", default=False,
                        help="Skip FOVs if output already exists in normalization folder")
    parser.add_argument("--normalisation_markers", nargs="*", default=None,
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
    parser.add_argument("--use_denoised", action="store_true",
                        help="Experimental, uses cohort level data to denoise reallocation factors")
    return parser.parse_args()


def list_available_markers(args: argparse.Namespace) -> None:
    """
    List available markers from the first FOV found in the base path.
    """
    base_path = args.base_path
    fov_folders = [
        os.path.join(base_path, folder)
        for folder in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, folder))
    ]
    if not fov_folders:
        print("❌ No FOV folders found.")
        return

    first_fov = fov_folders[0]
    ome_files = glob.glob(os.path.join(first_fov, "*.ome.tiff"))

    if not ome_files:
        print("❌ No .ome.tiff files in first FOV.")
        return

    marker_names = sorted([os.path.basename(f).replace(".ome.tiff", "") for f in ome_files])
    print(f"\nAvailable markers in FOV '{os.path.basename(first_fov)}':")
    print("list:", " ".join(marker_names))
    print("\n✅ Rerun without --list_available_markers to start the pipeline\n")


def setup_output_directories(output_base: str) -> dict:
    """
    Create and return the necessary output directories.
    """
    dirs = {
        "morph": os.path.join(output_base, "morphology_features"),
        "protein": os.path.join(output_base, "protein_features"),
        "original_sum": os.path.join(output_base, "original_sum"),
        "original_norm": os.path.join(output_base, "original_normalized"),
        "unhuddle_sum": os.path.join(output_base, "unhuddle_sum"),
        "unhuddle_norm": os.path.join(output_base, "unhuddle_normalized"),
        "unhuddle_denoised_sum": os.path.join(output_base, "unhuddle_denoised_sum"),
        "unhuddle_denoised_norm": os.path.join(output_base, "unhuddle_denoised_normalized"),
        "metadata_denoised": os.path.join(output_base, "metadata_denoise"),
        "adata": os.path.join(output_base, "adata_objects")
    }
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)
    return dirs


def get_fov_folders(args: argparse.Namespace, dirs: dict) -> list:
    """
    Retrieve and filter FOV folders based on user-specified arguments.
    """
    all_fovs = [
        os.path.join(args.base_path, folder)
        for folder in os.listdir(args.base_path)
        if os.path.isdir(os.path.join(args.base_path, folder))
    ]
    fov_folders = all_fovs
    if args.fovs:
        fov_folders = [f for f in all_fovs if os.path.basename(f) in args.fovs]
    if args.check_output_exist:
        fov_folders = [
            f for f in fov_folders
            if not glob.glob(os.path.join(dirs["unhuddle_norm"], f"{os.path.basename(f)}*"))
        ]
    return fov_folders


def result_failed(res: dict) -> bool:
    """
    Check if a result indicates failure.
    """
    return any(
        "error" in key.lower() or ("deepcell" in key.lower() and "error" in str(value).lower())
        for key, value in res.items()
    )


def summarize_results(results: dict, stage_description: str = "") -> None:
    """
    Summarize and print processing results.
    """
    errored = [os.path.basename(fov) for fov, res in results.items() if result_failed(res)]
    successful = [os.path.basename(fov) for fov, res in results.items() if not result_failed(res)]

    print("\n" + "=" * 40)
    if stage_description:
        print(f"Summary for {stage_description}:")
    else:
        print("Processing Summary:")

    if not successful and errored:
        print("❗ All FOVs failed. If overlay exists (basepath/{fov}/overlay.png) and looks good, "
              "please check the DeepCell server and geckodriver path.")
    elif errored:
        print("⚠️ Some FOVs failed:")
        for fov in errored:
            print(f"   ❌ {fov}")
        print("⚠️ Tip: inspect overlay (if exists) basepath/{fov}/overlay.png\n")
        if successful:
            print("✅ Successfully processed FOVs:")
            for fov in successful:
                print(f"   {fov}")
            print()
    else:
        print("✅ All FOVs processed successfully.")
    print("=" * 40 + "\n")

from functools import partial

def dispatch_stage(fov_folders, process_fn, arg_builder_fn, args, dirs, description, max_workers):
    process = partial(process_fn, dirs=dirs, args=args)
    results = run_parallel_stage(
        fov_folders,
        func=lambda fov: process(fov),
        max_workers=max_workers,
        description=description
    )
    return results

def run_parallel_stage(fov_folders: list, func, max_workers: int, description: str) -> dict:
    """
    Run the given processing function in parallel for each FOV using ProcessPoolExecutor.
    """
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, fov): fov for fov in fov_folders}
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            fov = futures[future]
            try:
                result = future.result()
                result["fov"] = fov
                results[fov] = result
            except Exception as e:
                logging.error(f"❌ FOV {fov} crashed: {e}")
                results[fov] = {"fov": fov, "critical_error": str(e), "crashed": True}
    summarize_results(results, stage_description=description)
    return results


def build_feature_args(fov: str, dirs: dict, args: argparse.Namespace):
    """
    Build arguments for the feature extraction stage for a given FOV.
    """
    return (
        fov,
        dirs["morph"],
        dirs["protein"],
        args.create_nuclear_mask,
        args.create_deepcell_mask,
        args.geckodriver_path,
        args.deepcell_url,
        args.mask_pattern,
        args.nuclear_markers,
        args.blue_markers,
        args.log_level,
        args.deepcell_resolution,
        args.nuclear_markers_overlay,
        args.membrane_markers_overlay,
    )


def build_reallocation_args(fov: str, dirs: dict, args: argparse.Namespace):
    protein_path = os.path.join(dirs["protein"], f"{os.path.basename(fov)}.csv")

    protein_df = pd.read_csv(protein_path)


    return (
        fov,
        protein_df,
        dirs["original_sum"],
        dirs["original_norm"],
        dirs["unhuddle_sum"],
        dirs["unhuddle_norm"],
        dirs["unhuddle_denoised_sum"],
        dirs["unhuddle_denoised_norm"],
        args.normalisation_markers,
        args.use_denoised,
        args.log_level
    )




def create_adata(args: argparse.Namespace) -> None:
    """
    Create a unified AnnData object from processed outputs.
    """
    logging.info("\n📦 Creating unified AnnData object...")
    from unhuddle_denoise.adata_builder import build_adata_from_outputs
    try:
        build_adata_from_outputs(
            output_base_path=args.output_base_path,
            working_path=args.base_path,
            output_adata_name="adata1.h5ad",
            max_workers=min(args.max_workers, 8)
        )
    except Exception as e:
        logging.error(f"❌ AnnData creation failed: {e}")
        print(f"[ERROR] AnnData creation failed: {e}\n")
