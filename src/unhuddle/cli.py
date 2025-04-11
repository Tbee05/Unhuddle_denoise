# src/unhuddle/cli.py

import os
import logging
from functools import partial
from unhuddle.cli_helpers import (
    parse_arguments,
    setup_logging,
    list_available_markers,
    setup_output_directories,
    get_fov_folders,
    build_feature_args,
    build_reallocation_args,
    run_parallel_stage,
    result_failed,
    create_adata
)
from unhuddle.pipeline import process_fov_features_only, process_fov_reallocation_only

def process_features(fov, dirs, args):
    return process_fov_features_only(*build_feature_args(fov, dirs, args))

def process_reallocation(fov, dirs, args):
    return process_fov_reallocation_only(*build_reallocation_args(fov, dirs, args))

def main():
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger("unhuddle")
    logger.debug(f"Logger '{logger.name}' is active at level: {logger.level}")

    if args.list_available_markers:
        list_available_markers(args)
        return

    if not args.normalisation_markers or not args.nuclear_markers:
        raise ValueError("Both --normalisation_markers and --nuclear_markers are required unless --list_available_markers is used.")

    if args.create_deepcell_mask and not args.geckodriver_path:
        raise ValueError("--geckodriver_path is required when --create_deepcell_mask is used.")

    if args.nuclear_markers_overlay is None:
        args.nuclear_markers_overlay = args.nuclear_markers
    logging.info(f"Using nuclear_markers_overlay: {args.nuclear_markers_overlay}")

    if args.membrane_markers_overlay is None:
        args.membrane_markers_overlay = args.normalisation_markers
    logging.info(f"Using membrane_markers_overlay: {args.membrane_markers_overlay}")

    if args.use_denoised:
        logging.info("⚙️ Percentile normalization enabled — cohort-level ExclMem_Sum data will be fetched before FOV loop.")

    dirs = setup_output_directories(args.output_base_path)
    fov_folders = get_fov_folders(args, dirs)

    if not fov_folders:
        print("❌ No FOVs selected for processing.")
        return

    # Stage 1: Feature Extraction
    results_stage1 = run_parallel_stage(
        fov_folders,
        func=partial(process_features, dirs=dirs, args=args),
        max_workers=args.max_workers,
        description="🔬 Extracting Features"
    )

    # Step 2. Denoise reallocation factors
    if args.use_denoised:
        logging.info("📊 Computing denoised reallocation factors (cohort-wide) ...")

        protein_csv_paths = [
            os.path.join(dirs["protein"], f)
            for f in os.listdir(dirs["protein"])
            if f.endswith(".csv")
        ]

        from unhuddle.reallocation_denoiser import compute_denoised_reallocation_factors

        denoised_summary_path = os.path.join(dirs["metadata_denoised"], "denoised_reallocation_summary.csv")
        compute_denoised_reallocation_factors(
            protein_csv_paths=protein_csv_paths,
            protein_features_dir=dirs["protein"]
        )

        logging.info(f"✅ Denoised reallocation factors saved to: {denoised_summary_path}")

    # Stage 2: Reallocation and Normalization
    results_stage2 = run_parallel_stage(
        fov_folders,
        func=partial(process_reallocation, dirs=dirs, args=args),
        max_workers=args.max_workers,
        description="🔁 Reallocation + Normalization"
    )

    successful = [os.path.basename(fov) for fov, res in results_stage2.items() if not result_failed(res)]
    if successful:
        print("📁 Processed FOV folders have updated masks and overlays — check the pseudocolored mask renders for validation.")
        print(f"📄 Unhuddle normalized output (partial): {dirs['unhuddle_norm']}")
        print(f"📄 Cell-level morphology metrics: {dirs['morph']}")
        print(f"📄 Raw/pre-normalization values: {args.output_base_path}\n")

    if args.create_adata:
        create_adata(args)

if __name__ == "__main__":
    main()
