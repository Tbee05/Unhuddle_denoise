# src/unhuddle/deepcell.py
import logging
logger = logging.getLogger(__name__)

import os
import glob
import re
import numpy as np
import tifffile
import tempfile
import shutil
import zipfile
import requests
from skimage.io import imsave

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile


import os
import re
import glob
import numpy as np
import tifffile
from skimage.io import imsave
import logging

logger = logging.getLogger(__name__)

def create_deepcell_mask_overlay(fov_path,
                                  red_markers,
                                  green_markers,
                                  blue_markers=None):
    """
    Create a DeepCell overlay mask from single-slice TIFFs in fov_path.
    Expected file pattern: {marker}.ome.tiff
    The overlay is saved in the fov_path as overlay_<FOV_ID>.tiff and .png.
    """
    if blue_markers is None:
        blue_markers = []

    pattern = os.path.join(fov_path, "*.ome.tiff")
    file_list = glob.glob(pattern)
    if not file_list:
        logger.warning(f"No marker files found in {fov_path} for DeepCell mask creation.")
        return None

    regex = re.compile(r"^(?P<marker>[^.]+)\.ome\.tiff$", re.IGNORECASE)
    marker_dict = {}
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        m = regex.match(file_name)
        if m:
            marker = m.group("marker")
            marker_dict.setdefault(marker, []).append(file_path)
        else:
            logger.info(f"File '{file_name}' does not match expected pattern. Skipping.")

    if not marker_dict:
        logger.warning(f"No valid marker files found in {fov_path} for DeepCell mask creation.")
        return None

    red_channel = None
    green_channel = None
    blue_channel = None if not blue_markers else None

    def scale_contrast(arr, low=0.5, high=99.5):
        """Percentile-based scaling to 0–255."""
        p_low, p_high = np.percentile(arr, (low, high))
        arr = (arr - p_low) / (p_high - p_low + 1e-6)
        return np.clip(arr * 255, 0, 255).astype(np.uint8)

    for marker, files in marker_dict.items():
        marker_img = None
        for file_path in files:
            try:
                img = tifffile.imread(file_path).astype(np.float32)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue
            marker_img = img if marker_img is None else marker_img + img

        if marker_img is None:
            continue

        if red_channel is None:
            H, W = marker_img.shape
            red_channel = np.zeros((H, W), dtype=np.float32)
            green_channel = np.zeros((H, W), dtype=np.float32)
            if blue_markers:
                blue_channel = np.zeros((H, W), dtype=np.float32)

        if marker in red_markers:
            red_channel += marker_img
        elif marker in green_markers:
            green_channel += marker_img
        elif blue_markers and marker in blue_markers:
            blue_channel += marker_img
        else:
            logger.debug(f"Marker '{marker}' not assigned to any channel for overlay. Skipping.")

    if red_channel is None or green_channel is None:
        logger.warning(f"Required markers for red and green channels not found in {fov_path}.")
        return None

    def sum_and_cap(arr):
        return np.clip(arr, 0, 1)  # Cap at 1, assuming input is in [0–1] or close

    red_8 = (sum_and_cap(red_channel) * 255).astype(np.uint8)
    green_8 = (sum_and_cap(green_channel) * 255).astype(np.uint8)
    blue_8 = (sum_and_cap(blue_channel) * 255).astype(np.uint8) if blue_channel is not None else np.zeros_like(red_8)

    overlay = np.stack((red_8, green_8, blue_8), axis=-1)

    fov_id = os.path.basename(os.path.normpath(fov_path))
    overlay_tiff = os.path.join(fov_path, f"overlay.tiff")
    overlay_png = os.path.join(fov_path, f"overlay.png")

    try:
        tifffile.imwrite(overlay_tiff, overlay, photometric="rgb")
        logger.info(f"DeepCell overlay TIFF saved to {overlay_tiff}")

        imsave(overlay_png, overlay)
        logger.info(f"DeepCell overlay PNG saved to {overlay_png}")
    except Exception as e:
        logger.error(f"Error saving DeepCell overlay for {fov_path}: {e}")
        return None

    return overlay_tiff



def safe_get(driver, url, retries=3, delay=5):
    import time
    for attempt in range(retries):
        try:
            driver.get(url)
            return
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to load {url} after {retries} retries.")


def process_deepcell_overlay(overlay_file, output_dir, deepcell_url, geckodriver_path, deepcell_resolution=10, wait=300):
    logger.debug("process_deepcell_overlay entered")
    MAG_TO_LABEL = {
        10: "10x (1 μm/pixel)",
        20: "20x (0.5 μm/pixel)",
        40: "40x (0.25 μm/pixel)",
        60: "60x (0.1667 μm/pixel)",
        100: "100x (0.1 μm/pixel)"
    }

    tmpdir = tempfile.mkdtemp()
    profile = FirefoxProfile()
    profile.set_preference("pdfjs.disabled", True)

    options = Options()
    options.profile = profile
    options.add_argument("--headless")

    driver = Firefox(service=Service(geckodriver_path), options=options)

    try:
        safe_get(driver, deepcell_url)
        WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".MuiButton-containedSecondary"))).click()
        WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.ID, "input-resolution-select"))).click()
        res_label = MAG_TO_LABEL.get(deepcell_resolution, "10x (1 μm/pixel)")
        res_xpath = f"//li[contains(.,'{res_label}')]"
        WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, res_xpath))).click()
        logger.debug(f"Selected resolution on DeepCell UI: {res_label}")

        WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, res_xpath))).click()
        logger.debug(f"Selected resolution on DeepCell UI: {deepcell_resolution}")

        upload_input = driver.find_element(By.XPATH, "//input[@type='file']")
        logger.debug(f"Overlay file to upload: {overlay_file}")
        logger.debug(f"Resolved absolute path: {os.path.abspath(overlay_file)}")
        logger.debug(f"File exists? {os.path.exists(overlay_file)}")
        abs_overlay_path = os.path.abspath(overlay_file)
        logger.debug(f"Passing to send_keys: {abs_overlay_path}")
        upload_input.send_keys(abs_overlay_path)

        WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.ID, "submitButton"))).click()

        download_button = WebDriverWait(driver, wait).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(),'Download Results')]"))
        )
        zip_url = download_button.get_attribute('href')
        resp = requests.get(zip_url, timeout=180)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download results, HTTP {resp.status_code}")

        zip_path = os.path.join(tmpdir, "results.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)

        extract_dir = os.path.join(output_dir, f"{os.path.basename(overlay_file).replace('overlay_', 'deepcell_mask_').replace('.ome.tiff', '')}_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        tifs = sorted([f for f in os.listdir(extract_dir) if f.endswith(".tif")])
        if len(tifs) != 2:
            raise RuntimeError(f"Expected 2 TIFFs, got {len(tifs)}")

        for i, tif_name in enumerate(tifs):
            out_path = os.path.join(output_dir, f"deepcell_mask_{i}.tiff")
            shutil.move(os.path.join(extract_dir, tif_name), out_path)
            logger.info(f"Saved DeepCell result: {out_path}")
        shutil.rmtree(extract_dir, ignore_errors=True)


    finally:
        driver.quit()
        shutil.rmtree(tmpdir, ignore_errors=True)
