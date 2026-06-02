"""Loading and parsing of raw inputs.

  - parse_dls          : DLS xlsx -> diameter distribution
  - load_tiff_stack    : one 3-channel TIFF -> {protein, lipid, transmitted}
  - load_all_images    : all TIFFs in a directory
  - analyze_dark_frames: per-channel offset + read noise from dark frames

Channels: 0=protein(488nm), 1=lipid(561nm), 2=transmitted light.
Ported verbatim from the archive's pipeline.py (Modules 1-3).
"""

import glob
import os

import numpy as np
import openpyxl
import tifffile


def parse_dls(xlsx_path, weighting='number', max_diameter_nm=500):
    """
    Parse DLS xlsx file and return diameter distribution.

    Args:
        xlsx_path: path to DLS Excel file
        weighting: 'number', 'volume', or 'intensity'
        max_diameter_nm: exclude particles larger than this (aggregates/dust)

    Returns:
        diameters: array of diameter values in nm
        probabilities: normalized probability for each diameter
        records: dict with individual record data
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb['Sheet1']

    # Find section headers
    section_map = {'intensity': 'X Intensity', 'volume': 'X Volume', 'number': 'X Number'}
    target_header = section_map[weighting]

    # Find the row where target section starts
    start_row = None
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True), 1):
        if row[0] == target_header:
            start_row = i + 1  # data starts on next row
            break

    if start_row is None:
        raise ValueError(f"Could not find '{target_header}' section in {xlsx_path}")

    # Read data until next section or end
    diameters = []
    records = {1: [], 2: [], 3: []}

    for row in ws.iter_rows(min_row=start_row, max_row=ws.max_row, values_only=True):
        # Stop if we hit another header
        if row[0] is not None and isinstance(row[0], str):
            break
        try:
            d = float(row[0])
            if d > max_diameter_nm:
                continue
            vals = [float(v) if v is not None else 0.0 for v in row[1:4]]
            diameters.append(d)
            for j in range(3):
                records[j+1].append(vals[j])
        except (TypeError, ValueError):
            continue

    diameters = np.array(diameters)
    avg = np.mean([records[1], records[2], records[3]], axis=0)

    # Normalize to probability distribution
    total = avg.sum()
    if total > 0:
        probabilities = avg / total
    else:
        raise ValueError("DLS distribution sums to zero")

    wb.close()

    return diameters, probabilities, records


def load_tiff_stack(path):
    """
    Load a 3-channel TIFF. Returns dict with 'protein', 'lipid', 'transmitted'.
    Channels: 0=protein(488), 1=lipid(561), 2=transmitted light.
    """
    img = tifffile.imread(path)

    # Handle different possible shapes
    if img.ndim == 2:
        raise ValueError(f"Expected 3-channel TIFF, got single 2D image: {path}")
    elif img.ndim == 3:
        if img.shape[0] == 3:
            return {'protein': img[0].astype(np.float64),
                    'lipid': img[1].astype(np.float64),
                    'transmitted': img[2].astype(np.float64)}
        elif img.shape[2] == 3:
            return {'protein': img[:,:,0].astype(np.float64),
                    'lipid': img[:,:,1].astype(np.float64),
                    'transmitted': img[:,:,2].astype(np.float64)}

    raise ValueError(f"Unexpected TIFF shape: {img.shape}")


def load_all_images(image_dir, pattern="*.tif*"):
    """Load all TIFF images from a directory."""
    paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No TIFF files found in {image_dir} with pattern {pattern}")

    images = []
    for p in paths:
        try:
            images.append(load_tiff_stack(p))
        except Exception as e:
            print(f"  Warning: skipping {os.path.basename(p)}: {e}")

    print(f"  Loaded {len(images)} images from {image_dir}")
    return images


def analyze_dark_frames(dark_dir, pattern="*.tif*"):
    """
    Analyze dark frames to get per-channel offset and read noise.

    Returns:
        dict with per-channel 'offset' (mean) and 'read_noise_var' (variance)
    """
    print("=== Dark Frame Analysis ===")
    darks = load_all_images(dark_dir, pattern)

    if len(darks) < 5:
        print(f"  Warning: only {len(darks)} dark frames. More is better (20+ recommended).")

    results = {}
    for channel in ['protein', 'lipid']:
        stack = np.array([d[channel] for d in darks])  # shape: (N, 512, 512)

        # Per-pixel mean (offset map) and variance (read noise map)
        offset_map = np.mean(stack, axis=0)
        variance_map = np.var(stack, axis=0)

        # Summary statistics (median across pixels for robustness)
        offset = np.median(offset_map)
        read_noise_var = np.median(variance_map)

        results[channel] = {
            'offset': float(offset),
            'read_noise_var': float(read_noise_var),
            'offset_map': offset_map,
            'variance_map': variance_map,
        }

        print(f"  {channel}: offset={offset:.1f}, read_noise_std={np.sqrt(read_noise_var):.2f}")

    return results
