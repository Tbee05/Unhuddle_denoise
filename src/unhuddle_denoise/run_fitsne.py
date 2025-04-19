# run_fitsne.py
import os
import glob
import numpy as np
import pandas as pd
from fitsne import FItSNE
import logging

def run_fitsne_dimension_reduction(output_base,
                                   input_dir="unhuddle_denoised_normalized",
                                   fitsne_dir="fitsne_coords",
                                   perplexity=30,
                                   threads=1):
    """
    Run FIt-SNE dimension reduction on CSV files from the unhuddle pipeline and store results.

    Parameters:
        output_base (str): Base directory for pipeline outputs.
        input_dir (str): Sub-directory containing CSV input files.
        fitsne_dir (str): Sub-directory to save the FIt-SNE embeddings.
        perplexity (float): Perplexity parameter for FIt-SNE.
        threads (int): Number of threads for parallel execution.

    Output:
        CSV embeddings saved in fitsne_coords directory.
    """

    input_path = os.path.join(output_base, input_dir)
    output_path = os.path.join(output_base, fitsne_dir)
    os.makedirs(output_path, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(input_path, "*.csv")))

    if not csv_files:
        logging.error(f"No CSV files found in {input_path}")
        return

    for csv_file in csv_files:
        fov_name = os.path.splitext(os.path.basename(csv_file))[0]
        logging.info(f"Processing FOV: {fov_name}")

        df = pd.read_csv(csv_file)
        data = df.select_dtypes(include=[np.number]).values

        if data.shape[0] == 0:
            logging.warning(f"Skipping empty dataset: {csv_file}")
            continue

        logging.info(f"Running FIt-SNE on {fov_name} with {data.shape[0]} cells...")
        embedding = FItSNE(data, perplexity=perplexity, nthreads=threads)

        embedding_df = pd.DataFrame(embedding, columns=['fitsne1', 'fitsne2'])
        embedding_df.to_csv(os.path.join(output_path, f"{fov_name}.csv"), index=False)

        logging.info(f"Embedding for {fov_name} saved successfully.")