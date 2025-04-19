def run_qc_from_memory(args, adata):
    base_path = args.output_base_path
    tsne_dir = os.path.join(base_path, "fitsne_coords")
    qc_output_dir, density_dir, segmentation_dir, storyboard_dir = create_directories(base_path)

    # Step 1: Low-intensity filtering
    adata = low_intensity_filter(adata, args.low_intensity_threshold)

    # Step 2: Optional dimension-reduction
    if os.path.exists(tsne_dir) and os.listdir(tsne_dir):
        adata = load_dimension_reduction_coords(tsne_dir, adata, args.coord_cols)
        tsne_available = True
    else:
        print("⚠️ Skipping t-SNE-based filtering (no coords found)")
        tsne_available = False

    # Step 3: Filtering status init
    adata.obs["filtering_status"] = "Unfiltered"

    # Step 4: Density-based filtering
    adata, region_cells_by_fov = perform_density_filtering(
        adata, qc_output_dir, density_dir
    )
    adata.obs.loc[adata.obs["QC_filter_low_quality_region"], "filtering_status"] = "low quality region"
    adata.obs.loc[adata.obs["QC_low_intensity_filter"], "filtering_status"] = "low intensity cell"

    # Step 5: Optional t-SNE filtering
    if tsne_available:
        adata = perform_tsne_filtering(adata, args.radius)

    # Step 6: Plots, images, tables
    fovs = list(set(adata.obs["fov"]))
    generate_segmentation_images(adata, fovs, density_dir, segmentation_dir, region_cells_by_fov)
    generate_storyboards(qc_output_dir, density_dir, segmentation_dir, storyboard_dir, fovs)
    if tsne_available:
        generate_tsne_plot(adata, qc_output_dir)
    generate_summary_tables(adata, qc_output_dir)

    # Step 7: Save updated adata IN PLACE (overwrite existing)
    adata_output_path = os.path.join(base_path, "adata_objects", args.adata_name)
    adata.write_h5ad(adata_output_path)
    print(f"✅ Updated AnnData saved in place to: {adata_output_path}")

    # Final memory cleanup (if needed)
    del adata
