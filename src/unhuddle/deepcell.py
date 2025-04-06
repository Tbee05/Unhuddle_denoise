# src/unhuddle/deepcell.py

import os
import glob
import re
import logging
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


import os
import glob
import re
import logging
import numpy as np
import tifffile


def create_deepcell_mask_overlay(fov_path, red_markers, green_markers, blue_markers=None):
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
            logging.warning(f"Missing channel: {key} — cannot generate overlay.")
            return None

    # Fill blue with zeros if missing
    if channels["blue"] is None:
        channels["blue"] = np.zeros_like(channels["red"], dtype=np.float32)
        logging.info("Blue channel missing — filled with zeros.")

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
        logging.info(f"Overlay saved to {overlay_path}")
        return overlay_path
    except Exception as e:
        logging.error(f"Failed to save RGB overlay TIFF: {e}")
        return None


def safe_get(driver, url, retries=3, delay=5):
    import time
    for attempt in range(retries):
        try:
            driver.get(url)
            return
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to load {url} after {retries} retries.")


def process_deepcell_overlay(overlay_file, output_dir, deepcell_url, geckodriver_path, wait=300):
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
        WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "//li[contains(.,'10x (1 μm/pixel)')]"))).click()

        upload_input = driver.find_element(By.XPATH, "//input[@type='file']")
        upload_input.send_keys(overlay_file)
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
            logging.info(f"Saved DeepCell result: {out_path}")

    finally:
        driver.quit()
        shutil.rmtree(tmpdir, ignore_errors=True)
