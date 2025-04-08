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

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile


def create_deepcell_mask_overlay(fov_path, red_markers, green_markers, blue_markers=None):
    logger.debug("create_deepcell_mask_overlay entered")
    if blue_markers is None:
        blue_markers = []

    marker_files = glob.glob(os.path.join(fov_path, "*.ome.tiff"))
    regex = re.compile(r"^(?P<marker>[^.]+)\.ome\.tiff$", re.IGNORECASE)

    channels = {"red": None, "green": None, "blue": None}
    for f in marker_files:
        m = regex.match(os.path.basename(f))
        if not m:
            continue
        marker = m.group("marker")
        img = tifffile.imread(f).astype(np.float32)

        if marker in red_markers:
            channels["red"] = img if channels["red"] is None else channels["red"] + img
        elif marker in green_markers:
            channels["green"] = img if channels["green"] is None else channels["green"] + img
        elif marker in blue_markers:
            channels["blue"] = img if channels["blue"] is None else channels["blue"] + img

    def clip(img):
        return np.clip(img, 0, 255).astype(np.uint8)

    # Red and Green channels are required
    for key in ["red", "green"]:
        if channels[key] is None:
            logger.warning(f"Missing channel: {key} — cannot generate overlay.")
            return None

    # Fill blue with zeros if missing
    if channels["blue"] is None:
        channels["blue"] = np.zeros_like(channels["red"], dtype=np.float32)
        logger.info("Blue channel missing — filled with zeros.")

    # Stack RGB channels and write photometric RGB TIFF
    overlay_rgb = np.stack([
        clip(channels["red"]),
        clip(channels["green"]),
        clip(channels["blue"])
    ], axis=-1)

    fov_id = os.path.basename(fov_path)
    overlay_path = os.path.join(fov_path, f"overlay_{fov_id}.tiff")

    try:
        tifffile.imwrite(overlay_path, overlay_rgb, photometric="rgb")
        logger.info(f"Overlay saved to {overlay_path}")
        return overlay_path
    except Exception as e:
        logger.error(f"Failed to save RGB overlay TIFF: {e}")
        return None


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

    finally:
        driver.quit()
        shutil.rmtree(tmpdir, ignore_errors=True)
