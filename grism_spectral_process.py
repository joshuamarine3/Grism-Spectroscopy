import os
import re
import sys
import glob
import csv
import shutil
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import numpy.ma as ma
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

from astropy.io import fits
from astropy.io.fits import getdata
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord

from scipy.ndimage import rotate, gaussian_filter1d, maximum_filter1d
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import medfilt, find_peaks, detrend
from scipy.special import wofz
from scipy import stats
from scipy.interpolate import CubicSpline

from lmfit.models import VoigtModel, ExponentialModel, SplineModel, LinearModel, GaussianModel, PolynomialModel, ConstantModel
from lmfit import report_fit

import pickle
import traceback

import importlib
import grism_utils_v2
importlib.reload(grism_utils_v2) # Ensure most up to date version of grism_utils_v2 is used
from grism_utils_v2 import spectrum


#########################################
# Utility functions for grism spectral processing
#########################################

def _basename(path):
    return os.path.basename(str(path))

def get_obs_time(img_file):
    """
    Return astropy Time object for a FITS file.
    """
    with fits.open(img_file) as hdul:
        hdr = hdul[1].header  # you already use extension 1
        date_obs = hdr.get("DATE-OBS")

        if date_obs is None:
            raise ValueError(f"No DATE-OBS in {img_file}")

    return Time(date_obs, format='isot', scale='utc')

def get_filter_from_header(img_file):
    with fits.open(img_file) as hdul:
        hdr = hdul[1].header

    filt = str(hdr.get("FILTER", "")).strip().lower()

    if "hrg" in filt:
        return "hrg"
    if "lrg" in filt:
        return "lrg"

    raise ValueError(f"Could not determine filter from header: {img_file}")

def is_centered(img_file, ext=1):
    """
    Return True if:
        CENTERED == True
    or
        CENTERED == False and WINSTABL == True
    """

    try:
        with fits.open(img_file) as hdul:
            hdr = hdul[ext].header

        centered = hdr.get("CENTERED", False)

        if centered:
            return True

        return hdr.get("WINSTABL", False)

    except Exception as e:
        print(
            f"WARNING: could not determine centering for "
            f"{os.path.basename(img_file)}: {e}"
        )
        return False
    
def filter_centered_files(files, label="images", verbose=True):
    """
    Keep only centered images.
    """

    kept = []
    rejected = []

    for f in files:
        if is_centered(f):
            kept.append(f)
        else:
            rejected.append(f)

    if verbose:
        print(
            f"{len(kept)} of {len(files)} "
            f"{label} are centered"
        )

        if len(rejected) > 0:
            print(
                f"Rejected {len(rejected)} "
                f"{label} due to centering:"
            )

            for r in rejected:
                print(f"  {os.path.basename(r)}")

    return kept, rejected

def _spec_file_key(S):
    """
    Key used to match a restored calibration object to a calibration image file.
    """
    if hasattr(S, "grism_image") and S.grism_image is not None:
        return _basename(S.grism_image)
    return None

def _build_spec_lookup(calib_spectra):
    """
    Build lookup: basename(grism_image) -> spectrum object
    """
    lookup = {}
    for S in calib_spectra:
        key = _spec_file_key(S)
        if key is not None:
            lookup[key] = S
    return lookup


def _save_calib_list_pickle(objects, filename, skip_flagged=False):
    """
    Save only selected attributes from a list of class objects.

    Parameters
    ----------
    objects : list
        List of class instances.
    filename : str
        Output pickle filename.
    include_attrs : list of str or None
        Attributes to save. If None, uses a compact calibration-focused default.
    skip_flagged : bool
        If True, skip objects with flagged=True.
    """

    include_attrs = [
        "object_name",
        "obs_date",
        "filter",
        "camera",
        "grism_image",
        "flagged",
        "hdr",
        "imsize_x",
        "imsize_y",
        "airmass",
        "exp_time",
        "moonangle",
        "moonphase",
        "trace_model",
        "telluric_pixel",
        "wavelength_range",
        "wavelength_correction",
        "wave_calib",
        "wave_r2",
        "wave_centroids",
        "pix_centroids",
        "wave_grid",
        "gain_smooth",
        "mean_fwhm",
        "median_fwhm",
        "std_fwhm"
    ]

    payload = []

    for obj in objects:
        if skip_flagged and getattr(obj, "flagged", False):
            continue

        row = {}

        for attr in include_attrs:
            val = getattr(obj, attr, None)

            try:
                # ensure it is pickleable
                pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL)
                row[attr] = val
            except Exception:
                row[attr] = repr(val)

        payload.append(row)

    with open(filename, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # print(f"Saved {len(payload)} objects to {filename}")

def restore_spectrum_objects_from_pickle(pkl_file, spectrum_class):
    """
    Restore lightweight spectrum-like objects from a pickle file
    containing a list of dictionaries.

    Parameters
    ----------
    pkl_file : str
        Path to pickle file.
    spectrum_class : class
        Your spectrum class from grism_utils_v2.py

    Returns
    -------
    objects : list
        List of restored spectrum_class instances.
    """
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    objects = []

    for row in data:
        obj = spectrum_class.__new__(spectrum_class)  # bypass __init__

        for key, val in row.items():
            setattr(obj, key, val)

        # # Rebuild recessary attributes
        # # Assumes you saved wave_grid and either gain_smooth or gain
        # if hasattr(obj, "wave_grid"):
        #     if hasattr(obj, "gain_smooth"):
        #         obj.gain_spline = interp1d(
        #             obj.wave_grid,
        #             obj.gain_smooth,
        #             bounds_error=False,
        #             fill_value="extrapolate"
        #         )
        #     elif hasattr(obj, "gain"):
        #         obj.gain_spline = interp1d(
        #             obj.wave_grid,
        #             obj.gain,
        #             bounds_error=False,
        #             fill_value="extrapolate"
        #         )

        objects.append(obj)

    return objects

# ------------------------------------------------------------
# Helper functions for building averaged calibration objects
# ------------------------------------------------------------

def match_science_to_calib(
    img_files,
    calib_objects,
    new_calib_img_files=None,
    tbin=900,
    verbose=True
):
    """
    Build non-overlapping calibration bins in time, assign each science image
    to the bin containing its nearest calibration entry, and keep only bins
    relevant to at least one science image.

    Calibration entries can come from:
      1. restored calibration objects from pkl files, using S.obs_date
      2. new calibration FITS files in the current basepath, using get_obs_time(file)

    Returns
    -------
    result : dict
        {
            "science_matches": list of dict,
            "calib_bins": list of dict,
            "calib_times": astropy Time array,
            "sorted_calib_entries": list of dict,
        }
    """

    if new_calib_img_files is None:
        new_calib_img_files = []

    if len(img_files) == 0:
        raise ValueError("img_files is empty.")

    if len(calib_objects) == 0 and len(new_calib_img_files) == 0:
        raise ValueError("No calibration objects or new calibration files provided.")

    # ----------------------------------------------------------
    # Build unified calibration entries
    # ----------------------------------------------------------
    calib_entries = []
    seen_keys = set()

    # Existing calibration objects from pkl
    for S in calib_objects:
        obs_date = getattr(S, "obs_date", None)

        if obs_date is None:
            continue

        try:
            t = Time(obs_date, scale="utc")
        except Exception:
            try:
                t = Time(obs_date)
            except Exception:
                continue

        f = getattr(S, "grism_image", None)
        key = ("object", os.path.basename(str(f)) if f is not None else id(S), float(t.unix))

        if key in seen_keys:
            continue
        seen_keys.add(key)

        calib_entries.append({
            "source": "pkl",
            "object": S,
            "file": f,
            "time": t,
            "basename": os.path.basename(str(f)) if f is not None else "UNKNOWN_PKL_OBJECT",
            "flagged": getattr(S, "flagged", False),
        })
    
    existing_files = [e['file'] for e in calib_entries]

    # New calibration files from current basepath
    for f in new_calib_img_files:
        try:
            t = get_obs_time(f)
        except Exception as e:
            if verbose:
                print(f"Skipping new calibration file with unreadable DATE-OBS: {os.path.basename(f)} ({e})")
            continue

        key = ("object", os.path.basename(str(f)), float(t.unix))

        if key in seen_keys:
            continue
        seen_keys.add(key)

        calib_entries.append({
            "source": "file",
            "object": None,
            "file": f,
            "time": t,
            "basename": os.path.basename(str(f)),
            "flagged": False,
        })

    if len(calib_entries) == 0:
        raise ValueError("No calibration entries with valid observation times.")

    # ----------------------------------------------------------
    # Sort calibration entries by time
    # ----------------------------------------------------------
    calib_entries = sorted(calib_entries, key=lambda e: e["time"].unix)

    sorted_calib_entries = calib_entries
    sorted_calib_times = Time([e["time"] for e in sorted_calib_entries])

    # ----------------------------------------------------------
    # Build non-overlapping bins from calibration entries
    # ----------------------------------------------------------
    raw_bins = []
    i = 0

    while i < len(sorted_calib_entries):
        ref_entry = sorted_calib_entries[i]
        ref_time = ref_entry["time"]

        bin_entries = [ref_entry]
        bin_times = [ref_time]
        bin_indices = [i]

        i += 1

        while i < len(sorted_calib_entries):
            dt_sec = sorted_calib_entries[i]["time"].unix - ref_time.unix

            if dt_sec <= tbin:
                bin_entries.append(sorted_calib_entries[i])
                bin_times.append(sorted_calib_entries[i]["time"])
                bin_indices.append(i)
                i += 1
            else:
                break

        bin_unix = np.array([t.unix for t in bin_times])
        mid_unix = np.median(bin_unix)
        rep_local_idx = int(np.argmin(np.abs(bin_unix - mid_unix)))

        raw_bins.append({
            "bin_id": len(raw_bins),
            "entries": bin_entries,
            "files": [e["file"] for e in bin_entries],
            "objects": [e["object"] for e in bin_entries],
            "sources": [e["source"] for e in bin_entries],
            "times": bin_times,
            "indices": bin_indices,
            "representative_entry": bin_entries[rep_local_idx],
            "representative_file": bin_entries[rep_local_idx]["file"],
            "representative_object": bin_entries[rep_local_idx]["object"],
            "representative_time": bin_times[rep_local_idx],
            "start_time": bin_times[0],
            "end_time": bin_times[-1],
            "n_entries": len(bin_entries),
            "n_files": len(bin_entries),  # backwards-compatible naming
            "n_pkl": sum(e["source"] == "pkl" for e in bin_entries),
            "n_new_files": sum(e["source"] == "file" for e in bin_entries),
        })

    # ----------------------------------------------------------
    # Assign science images to nearest calibration entry / bin
    # ----------------------------------------------------------
    science_times = Time([get_obs_time(f).isot for f in img_files], scale="utc")
    science_matches = []

    for sci_file, sci_time in zip(img_files, science_times):
        dt = np.abs((sorted_calib_times - sci_time).to("s").value)
        nearest_idx = int(np.argmin(dt))

        nearest_entry = sorted_calib_entries[nearest_idx]
        nearest_dt_sec = float(dt[nearest_idx])

        matched_bin = None
        for b in raw_bins:
            if nearest_idx in b["indices"]:
                matched_bin = b
                break

        if matched_bin is None:
            raise RuntimeError("Internal error: nearest calibration entry did not map to any bin.")

        science_matches.append({
            "science_file": sci_file,
            "science_time": sci_time,

            "nearest_calib_entry": nearest_entry,
            "nearest_calib_file": nearest_entry["file"],
            "nearest_calib_object": nearest_entry["object"],
            "nearest_calib_source": nearest_entry["source"],
            "nearest_calib_time": nearest_entry["time"],
            "nearest_dt_sec": nearest_dt_sec,

            "bin_id": matched_bin["bin_id"],

            "calib_group_entries": matched_bin["entries"],
            "calib_group_files": matched_bin["files"],
            "calib_group_objects": matched_bin["objects"],
            "calib_group_sources": matched_bin["sources"],
            "calib_group_times": matched_bin["times"],

            "representative_entry": matched_bin["representative_entry"],
            "representative_file": matched_bin["representative_file"],
            "representative_object": matched_bin["representative_object"],
            "representative_time": matched_bin["representative_time"],
        })

    # ----------------------------------------------------------
    # Keep only bins actually used by at least one science image
    # ----------------------------------------------------------
    used_bin_ids = sorted(set(row["bin_id"] for row in science_matches))
    used_bins = [raw_bins[bid] for bid in used_bin_ids]

    old_to_new = {old: new for new, old in enumerate(used_bin_ids)}

    calib_bins = []

    for b in used_bins:
        b_new = dict(b)
        b_new["original_bin_id"] = b["bin_id"]
        b_new["bin_id"] = old_to_new[b["bin_id"]]
        calib_bins.append(b_new)

    for row in science_matches:
        old_bin = row["bin_id"]
        row["original_bin_id"] = old_bin
        row["bin_id"] = old_to_new[old_bin]

    # refresh bin-linked fields in science_matches
    bin_lookup = {b["bin_id"]: b for b in calib_bins}

    for row in science_matches:
        b = bin_lookup[row["bin_id"]]

        row["calib_group_entries"] = b["entries"]
        row["calib_group_files"] = b["files"]
        row["calib_group_objects"] = b["objects"]
        row["calib_group_sources"] = b["sources"]
        row["calib_group_times"] = b["times"]

        row["representative_entry"] = b["representative_entry"]
        row["representative_file"] = b["representative_file"]
        row["representative_object"] = b["representative_object"]
        row["representative_time"] = b["representative_time"]

    # ----------------------------------------------------------
    # Verbose summary
    # ----------------------------------------------------------
    if verbose:
        date_start = sorted_calib_times[0].isot if len(sorted_calib_times) > 0 else "N/A"
        date_end = sorted_calib_times[-1].isot if len(sorted_calib_times) > 0 else "N/A"

        print("")
        print("Calibration bin summary")
        print("-----------------------")
        print(f"Found {len(calib_bins)} science-relevant calibration bins")
        print(f"Calibration date range available: {date_start}  →  {date_end}")
        print(f"Total calibration entries available: {len(sorted_calib_entries)}")
        print(f"  From pkl objects: {sum(e['source'] == 'pkl' for e in sorted_calib_entries)}")
        print(f"  From new files:   {sum(e['source'] == 'file' for e in sorted_calib_entries)}")
        print(f"Science images: {len(img_files)}")
        print(f"Bin width criterion: {tbin:.0f} s ({tbin/60:.1f} min)")
        print("")

        sci_by_bin = {b["bin_id"]: [] for b in calib_bins}

        for row in science_matches:
            sci_by_bin[row["bin_id"]].append(os.path.basename(row["science_file"]))

        for b in calib_bins:
            rep = b["representative_entry"]
            rep_name = os.path.basename(str(rep["file"])) if rep["file"] is not None else rep["basename"]

            print(
                f"Bin {b['bin_id']:02d}: "
                f"{b['n_entries']} calibration entrie(s), "
                f"{b['start_time'].isot} → {b['end_time'].isot}, "
                f"rep = {rep_name}, "
                f"pkl={b['n_pkl']}, new_files={b['n_new_files']}"
            )

            print("  Calibration entries:")
            for e in b["entries"]:
                name = os.path.basename(str(e["file"])) if e["file"] is not None else e["basename"]
                flag_txt = " FLAGGED" if e.get("flagged", False) else ""
                print(f"    - [{e['source']}] {name}  [{e['time'].isot}]{flag_txt}")

            print("  Science images:")
            for sci_name in sci_by_bin[b["bin_id"]]:
                print(f"    - {sci_name}")

            print("")

    return {
        "science_matches": science_matches,
        "calib_bins": calib_bins,
        "calib_times": sorted_calib_times,
        "sorted_calib_entries": sorted_calib_entries,

        # backwards-compatible-ish fields
        "sorted_calib_files": [e["file"] for e in sorted_calib_entries],
        "sorted_calib_objects": [e["object"] for e in sorted_calib_entries],
    }

def _fit_poly_to_curve(x, y, degree):
    """
    Fit polynomial coeffs in np.polyval order (high -> low).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < degree + 1:
        raise ValueError("Not enough valid points to fit polynomial.")
    return np.polyfit(x[good], y[good], degree)

def _average_wavelength_solution(calib_group, npix=None, degree=None):
    good = [
        S for S in calib_group
        if not getattr(S, "flagged", False)
        and hasattr(S, "wave_calib")
        and getattr(S, "wave_calib", None) is not None
    ]

    if len(good) == 0:
        raise ValueError("No non-flagged calibration objects in group have wave_calib.")

    if npix is None:
        if hasattr(good[0], "imsize_x") and good[0].imsize_x is not None:
            npix = int(good[0].imsize_x)
        else:
            npix = 4096

    if degree is None:
        degree = len(np.asarray(good[0].wave_calib)) - 1

    x_common = np.arange(npix, dtype=float)

    wave_curves = []
    for S in good:
        c = np.asarray(S.wave_calib, dtype=float)
        wave_curves.append(np.polyval(c, x_common))

    wave_curves = np.array(wave_curves)
    wave_med = np.nanmean(wave_curves, axis=0)

    coeff_med = _fit_poly_to_curve(x_common, wave_med, degree=degree)

    return x_common, wave_med, coeff_med

def _average_gain_solution(calib_group):
    good = [
        S for S in calib_group
        if not getattr(S, "flagged", False)
        and hasattr(S, "wave_grid")
        and hasattr(S, "gain_smooth")
        and getattr(S, "wave_grid", None) is not None
        and getattr(S, "gain_smooth", None) is not None
    ]

    if len(good) == 0:
        raise ValueError("No non-flagged calibration objects in group have wave_grid/gain_smooth.")

    wave_mins = []
    wave_maxs = []

    for S in good:
        wave_i = np.asarray(S.wave_grid, dtype=float)
        gain_i = np.asarray(S.gain_smooth, dtype=float)

        if len(wave_i) != len(gain_i):
            continue

        valid = np.isfinite(wave_i) & np.isfinite(gain_i)
        if np.sum(valid) < 2:
            continue

        wave_mins.append(np.nanmin(wave_i[valid]))
        wave_maxs.append(np.nanmax(wave_i[valid]))

    if len(wave_mins) == 0:
        raise ValueError("No usable non-flagged gain curves after validation.")

    wmin = max(wave_mins)
    wmax = min(wave_maxs)

    if wmin >= wmax:
        raise ValueError("No overlapping wave_grid range in non-flagged calibration group.")

    n_common = min(len(np.asarray(S.wave_grid)) for S in good)
    wave_common = np.linspace(wmin, wmax, n_common)

    gain_curves = []

    for S in good:
        wave_i = np.asarray(S.wave_grid, dtype=float)
        gain_i = np.asarray(S.gain_smooth, dtype=float)

        if len(wave_i) != len(gain_i):
            continue

        valid = np.isfinite(wave_i) & np.isfinite(gain_i)
        wave_i = wave_i[valid]
        gain_i = gain_i[valid]

        if len(wave_i) < 2:
            continue

        order = np.argsort(wave_i)
        wave_i = wave_i[order]
        gain_i = gain_i[order]

        gain_curves.append(np.interp(wave_common, wave_i, gain_i))

    if len(gain_curves) == 0:
        raise ValueError("Could not interpolate any non-flagged gain curves.")

    gain_curves = np.array(gain_curves)
    gain_med = np.nanmean(gain_curves, axis=0)

    gain_interp = interp1d(
        wave_common,
        gain_med,
        bounds_error=False,
        fill_value="extrapolate"
    )

    return wave_common, gain_med, gain_interp

def build_averaged_calib_object(calib_group, spectrum_class, bin_id=None, npix=None, wave_degree=None):
    """
    Build a lightweight spectrum-like calibration object from a calibration group.
    Compatible with downstream code expecting .wave_calib, .gain_spline, etc.
    """
    if len(calib_group) == 0:
        raise ValueError("calib_group is empty.")

    template = calib_group[0]

    x_common, wave_med, coeff_med = _average_wavelength_solution(
        calib_group,
        npix=npix,
        degree=wave_degree
    )

    wave_common, gain_med, gain_interp = _average_gain_solution(calib_group)

    avg = spectrum_class.__new__(spectrum_class)

    # copy a few metadata attrs from template if present
    for attr in ["filter", "camera", "hdr"]:
        if hasattr(template, attr):
            setattr(avg, attr, getattr(template, attr))

    avg.object_name = f"AVG_CALIB_BIN_{bin_id}" if bin_id is not None else "AVG_CALIB"
    avg.grism_image = None
    avg.obs_date = None
    avg.flagged = False

    # wavelength-calibration attrs
    avg.wave_calib = coeff_med
    avg.wave_r2 = None
    avg.imsize_x = len(x_common)
    avg.wave_curve_avg = wave_med

    # gain-calibration attrs
    avg.wave_grid = wave_common
    avg.gain_smooth = gain_med
    avg.gain = gain_med.copy()
    avg.gain_spline = gain_interp

    # bookkeeping
    avg.bin_id = bin_id
    avg.calib_group_members = [getattr(S, "grism_image", None) for S in calib_group]

    return avg

# ------------------------------------------------------------
# Missing-calibration derivation
# ------------------------------------------------------------

def derive_single_calib_spectrum(grism_image, spectrum_class, filter, ref_dir):
    """
    Derive one calibration spectrum object from a calibration image.
    If derivation fails, return the partially-created object with flagged=True.
    """
    S = spectrum_class(grism_image, calib_spectrum=None, filter=filter, calib_img=True)

    try:
        trace_center, cutouts, subim = S.fit_trace(plot=False, curved=True)

        spec = S.extract_spectrum(
            show_box=False,
            plot=False,
            curved_box=True,
        )

        telluric_pixel = S.fit_telluric(plot=False)

        # If fit_telluric failed internally, catch that here
        if not hasattr(S, "telluric_pixel"):
            raise AttributeError("telluric_pixel was not created.")

        wavelength_correction = S.derive_wavelength_correction()
        wave_calib = S.derive_wavelength_solution(show_points=False, plot=False)
        waves = S.wavelength_calibrate()
        ref_wave, ref_flux = S.load_stelib_spectrum(ref_dir)
        wave_grid, flux_data_interp, flux_ref_interp = S.match_and_interpolate()
        gain_curve = S.derive_gain_calibration(plot=False)

        S.flagged = False

    except Exception as e:
        print(f"  Flagging failed calibration image: {grism_image}")
        print(f"  Reason: {e}")

        S.flagged = True
        S.failure_reason = str(e)

    return S

def ensure_required_calib_spectra(
    matches,
    ZWO_calib_spectra,
    QHY_calib_spectra,
    spectrum_class,
    filter,
    ref_dir = None,
    save_updated=True,
    zwo_pkl=None,
    qhy_pkl=None,
    verbose=True
):
    zwo_lookup = _build_spec_lookup(ZWO_calib_spectra)
    qhy_lookup = _build_spec_lookup(QHY_calib_spectra)

    needed_calib_files = sorted({
        _basename(f)
        for row in matches["science_matches"]
        for f in row["calib_group_files"]
    })

    needed_fullpaths = {}
    for row in matches["science_matches"]:
        for f in row["calib_group_files"]:
            needed_fullpaths[_basename(f)] = f

    new_zwo = []
    new_qhy = []
    new_flagged = []
    skipped_existing_flagged = []

    for key in needed_calib_files:

        # Already exists in ZWO list
        if key in zwo_lookup:
            if getattr(zwo_lookup[key], "flagged", False):
                skipped_existing_flagged.append(key)
            continue

        # Already exists in QHY list
        if key in qhy_lookup:
            if getattr(qhy_lookup[key], "flagged", False):
                skipped_existing_flagged.append(key)
            continue

        fullpath = needed_fullpaths[key]

        if verbose:
            print(f"Deriving missing calibration spectrum: {key}")

        S = derive_single_calib_spectrum(fullpath, spectrum_class, filter, ref_dir)

        cam = getattr(S, "camera", None)

        if cam == "ASI Camer":
            ZWO_calib_spectra.append(S)
            zwo_lookup[key] = S
            new_zwo.append(key)
        elif cam == "QHYCCD-Ca":
            QHY_calib_spectra.append(S)
            qhy_lookup[key] = S
            new_qhy.append(key)
        else:
            # fallback
            ZWO_calib_spectra.append(S)
            zwo_lookup[key] = S
            new_zwo.append(key)
            if verbose:
                print(f"  Warning: unknown camera '{cam}' for {key}; appended to ZWO list.")

        if getattr(S, "flagged", False):
            new_flagged.append(key)

    if save_updated:
        if zwo_pkl is None:
            zwo_pkl = f"Calib_Spectra_{filter}_ZWO.pkl"
        if qhy_pkl is None:
            qhy_pkl = f"Calib_Spectra_{filter}_QHY.pkl"

        _save_calib_list_pickle(ZWO_calib_spectra, zwo_pkl)
        _save_calib_list_pickle(QHY_calib_spectra, qhy_pkl)

        if verbose:
            print("")
            print(f"Saved updated ZWO calibration list -> {zwo_pkl}")
            print(f"Saved updated QHY calibration list -> {qhy_pkl}")

    return {
        "ZWO_calib_spectra": ZWO_calib_spectra,
        "QHY_calib_spectra": QHY_calib_spectra,
        "new_zwo_files": new_zwo,
        "new_qhy_files": new_qhy,
        "new_flagged_files": new_flagged,
        "skipped_existing_flagged": skipped_existing_flagged,
    }

# ------------------------------------------------------------
# Build averaged per-bin calibration objects
# ------------------------------------------------------------

def build_bin_averaged_calibrations(
    matches,
    ZWO_calib_spectra,
    QHY_calib_spectra,
    spectrum_class,
    derivation_summary=None,
    verbose=True
):
    zwo_lookup = _build_spec_lookup(ZWO_calib_spectra)
    qhy_lookup = _build_spec_lookup(QHY_calib_spectra)

    bin_avg_lookup = {}
    failed_bins = []
    skipped_flagged_count = 0
    used_calib_count = 0

    for bin_info in matches["calib_bins"]:
        bin_id = bin_info["bin_id"]
        group_files = bin_info["files"]

        calib_group = []

        for f in group_files:
            key = _basename(f)

            if key in zwo_lookup:
                S = zwo_lookup[key]
            elif key in qhy_lookup:
                S = qhy_lookup[key]
            else:
                continue

            if getattr(S, "flagged", False):
                skipped_flagged_count += 1
                continue

            calib_group.append(S)

        used_calib_count += len(calib_group)

        if len(calib_group) == 0:
            failed_bins.append(bin_id)
            continue

        try:
            avg_obj = build_averaged_calib_object(
                calib_group,
                spectrum_class=spectrum_class,
                bin_id=bin_id
            )
            bin_avg_lookup[bin_id] = avg_obj

        except Exception as e:
            print(f"  Failed to build averaged calibration for bin {bin_id}: {e}")
            failed_bins.append(bin_id)

    if verbose:
        n_bins_total = len(matches["calib_bins"])
        n_bins_built = len(bin_avg_lookup)
        n_bins_failed = len(failed_bins)

        n_new_zwo = 0
        n_new_qhy = 0
        n_new_flagged = 0

        if derivation_summary is not None:
            n_new_zwo = len(derivation_summary.get("new_zwo_files", []))
            n_new_qhy = len(derivation_summary.get("new_qhy_files", []))
            n_new_flagged = len(derivation_summary.get("new_flagged_files", []))

        print("")
        print("Averaged calibration summary")
        print("----------------------------")
        print(f"Science-relevant calibration bins found: {n_bins_total}")
        print(f"Averaged calibration objects built:      {n_bins_built}")
        print(f"Bins that failed to build:               {n_bins_failed}")
        print(f"Calibration images used in bin averages: {used_calib_count}")
        print(f"Flagged calibration entries skipped:     {skipped_flagged_count}")
        print(f"Calibration images newly derived:        {n_new_zwo + n_new_qhy}")
        print(f"  ZWO appended:                          {n_new_zwo}")
        print(f"  QHY appended:                          {n_new_qhy}")
        print(f"  Newly derived but flagged:             {n_new_flagged}")

        if n_bins_failed > 0:
            print(f"Failed bin IDs: {failed_bins}")

    return bin_avg_lookup

# ------------------------------------------------------------
# Assign averaged calibrations to science images
# ------------------------------------------------------------

def attach_bin_averaged_calibrations_to_science_matches(matches, bin_avg_lookup, fallback=True, verbose=True):
    """
    Add an averaged calibration object to each science match row.

    If fallback=True, science images assigned to failed bins are reassigned
    to the nearest successful bin by bin_id.
    """

    successful_bins = sorted(bin_avg_lookup.keys())

    if len(successful_bins) == 0:
        raise ValueError("No successful averaged calibration bins available.")

    n_fallback = 0

    for row in matches["science_matches"]:
        original_bin = row["bin_id"]

        if original_bin in bin_avg_lookup:
            assigned_bin = original_bin
            used_fallback = False
        else:
            if not fallback:
                assigned_bin = None
                used_fallback = False
            else:
                assigned_bin = min(
                    successful_bins,
                    key=lambda b: abs(b - original_bin)
                )
                used_fallback = True
                n_fallback += 1

        row["original_bin_id"] = original_bin
        row["assigned_bin_id"] = assigned_bin
        row["used_fallback_calib"] = used_fallback
        row["averaged_calib_spectrum"] = (
            bin_avg_lookup.get(assigned_bin, None)
            if assigned_bin is not None else None
        )

    if verbose:
        print("")
        print("Science calibration assignment summary")
        print("--------------------------------------")
        print(f"Successful calibration bins: {successful_bins}")
        print(f"Science images reassigned by fallback: {n_fallback}")

        if n_fallback > 0:
            for row in matches["science_matches"]:
                if row["used_fallback_calib"]:
                    print(
                        f"  {os.path.basename(row['science_file'])}: "
                        f"bin {row['original_bin_id']} → bin {row['assigned_bin_id']}"
                    )

    return matches

def save_1d_spectrum_fits(S, output_path, match_row=None):
    import os
    import numpy as np
    from astropy.io import fits

    wave = np.asarray(S.waves, dtype=float)
    flux = np.asarray(S.cal_spec, dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    # DS9-friendly: primary image is the flux vector
    hdu = fits.PrimaryHDU(data=flux.astype(np.float32))
    hdr = hdu.header

    # copy useful source header cards
    if hasattr(S, "hdr") and S.hdr is not None:
        for key in [
            "BLKNAME", "DATE-OBS", "EXPTIME", "AIRMASS",
            "FILTER", "INSTRUME", "OBSNAME", "TELRA", "TELDEC",
            "MOONANGL", "MOONPHAS"
        ]:
            if key in S.hdr:
                try:
                    hdr[key] = S.hdr[key]
                except Exception:
                    pass

    hdr["PRODTYPE"] = "1D_SPEC"
    hdr["BUNIT"] = "ergs/s/cm2/angstrom"
    hdr["CTYPE1"] = "Wavelength"
    hdr["CUNIT1"] = "Angstrom"
    hdr["CRPIX1"] = 1
    hdr["CRVAL1"] = float(wave[0])

    if len(wave) > 1:
        hdr["CDELT1"] = float(np.nanmedian(np.diff(wave)))
    else:
        hdr["CDELT1"] = 1.0

    hdr['FWHM'] = (float(getattr(S, "median_fwhm", np.nan)), "Spectral trace median fwhm (pixels)")
    hdr['FWHMSTD'] = (float(getattr(S, "std_fwhm", np.nan)), "Spectral trace fwhm std (pixels)")
    hdr["TELLPIX"] = (float(getattr(S, "telluric_pixel", np.nan)), "X-pixel of telluric O2 B-Band")
    hdr['TELLFAIL'] = (getattr(S, "telluric_fail", False), "Telluric fit failed, defaulted to median value")
    hdr["WAVECAL"] = True
    hdr["GAINCAL"] = True

    if match_row is not None:

        calib_files = match_row.get("calib_group_files", [])

        hdr["CALBIN"] = int(
            match_row.get(
                "assigned_bin_id",
                match_row.get("bin_id", -1)
            )
        )

        hdr["NCALIB"] = len(calib_files)

        hdr["FALLBK"] = bool(
            match_row.get("used_fallback_calib", False)
        )

        # store calibration filenames
        for i, f in enumerate(calib_files[:20]):

            fname = os.path.basename(str(f))

            # FITS card values should stay short-ish
            if len(fname) > 60:
                fname = fname[:60]

            hdr[f"CAL{i:02d}"] = fname

    hdr.add_history("Reduced with grism_spectral_process.py")

    hdu.writeto(output_path, overwrite=True, output_verify="fix")

#########################################
# Main spetral processing workflow
#########################################

def main():
    parser = argparse.ArgumentParser(description="Process grism spectra with averaged calibrations.")
    # parser.add_argument("--filter", type=str, required=True, help="Filter name ('hrg' or 'lrg').")
    parser.add_argument("--calib_dir", type=str, default='.', help="Directory containing calibration spectrum pickles (default is current directory)")
    parser.add_argument("--basepath", type=str, required=True, help="Raw spectral images base path")
    parser.add_argument("--file", type=str, required=False, help="Science image file to process (processes all in basepath if not specified)")
    parser.add_argument('--outpath', type=str, default='.', help='Directory to save outputs (default is current directory)')
    parser.add_argument('--verbose', default = False, action='store_true', help='Enable verbose output')
    parser.add_argument("--update-calibs", action="store_true")
    parser.add_argument("--show_steps", default = False, action = 'store_true', help = 'Show all steps w/ plots along the way')

    args = parser.parse_args()

    basepath = os.path.abspath(os.path.expanduser(args.basepath))
    calib_dir = os.path.abspath(os.path.expanduser(args.calib_dir))
    outpath = os.path.abspath(os.path.expanduser(args.outpath))
    os.makedirs(outpath, exist_ok=True)

    if args.file:
        file_arg = os.path.expanduser(args.file)

        if not os.path.isabs(file_arg):
            file_arg = os.path.join(basepath, file_arg)

        all_candidate_files = [os.path.abspath(file_arg)]
    else:
        all_candidate_files = sorted(glob.glob(os.path.join(basepath, "*.fz")))

    files_by_filter = {"hrg": [], "lrg": []}

    for f in all_candidate_files:
        try:
            filt = get_filter_from_header(f)
            files_by_filter[filt].append(f)
        except Exception as e:
            print(f"Skipping {os.path.basename(f)}: {e}")

    # # filter = args.filter.lower()
    # if args.file:
    #     with fits.open(os.path.join(args.basepath, args.file)) as hdul:
    #         hdr = hdul[1].header
    #         filter = hdr['FILTER']
    for filter, filter_files in files_by_filter.items():

        if len(filter_files) == 0:
            continue
        
        if args.verbose:
            print("")
            print(f"Processing {len(filter_files)} {filter.upper()} file(s)")
            print("----------------------------------------")
            print(f"Selected filter: {filter}")
        if filter not in ['hrg', 'lrg'] or type(filter) is not str:
            print(f"Error: Invalid filter '{args.filter}'. Must be 'hrg' or 'lrg'.")
            sys.exit(1)

        calib_targets = ['hr_718', 'hr_3454', 'hr_4468', 'hr_4963']

        calib_target_pattern = re.compile(
            r"(?<![A-Za-z0-9])(" + "|".join(re.escape(t) for t in calib_targets) + r")(?![A-Za-z0-9])",
            re.IGNORECASE
        )

        new_calib_img_files = [
            f for f in filter_files
            if calib_target_pattern.search(os.path.basename(f))
        ]

        img_files = [
            f for f in filter_files
            if not calib_target_pattern.search(os.path.basename(f))
        ]

        if len(img_files) == 0:
            print(f"No science files found for {filter.upper()}; skipping.")
            continue

        # Import existing calibration spectra class objects from pickles

        ZWO_calib_spectra = restore_spectrum_objects_from_pickle(
            os.path.join(calib_dir, f"Calib_Spectra_{filter}_ZWO.pkl"),
            spectrum
        )

        QHY_calib_spectra = restore_spectrum_objects_from_pickle(
            os.path.join(calib_dir, f"Calib_Spectra_{filter}_QHY.pkl"),
            spectrum
        )

        all_fz = sorted(glob.glob(os.path.join(basepath, f"*{filter}*.fz")))

        if args.file:
            img_files = [os.path.join(basepath, args.file)]
        else:
            img_files = [
                f for f in all_fz
                if not calib_target_pattern.search(os.path.basename(f))
            ]

        img_files, rejected_science = filter_centered_files(
            img_files,
            label=f"{filter.upper()} science images",
            verbose=args.verbose
        )

        new_calib_img_files, rejected_calibs = filter_centered_files(
            new_calib_img_files,
            label=f"{filter.upper()} calibration images",
            verbose=args.verbose
        )

        matches = match_science_to_calib(
            img_files=img_files,
            calib_objects=ZWO_calib_spectra + QHY_calib_spectra,
            new_calib_img_files=new_calib_img_files,
            tbin=900,
            verbose=args.verbose
        )

        update_result = ensure_required_calib_spectra(
            matches=matches,
            ZWO_calib_spectra=ZWO_calib_spectra,
            QHY_calib_spectra=QHY_calib_spectra,
            spectrum_class=spectrum,
            filter=filter,
            ref_dir = os.path.join(calib_dir, "STELIB"),
            save_updated=args.update_calibs,
            zwo_pkl=os.path.join(calib_dir, f"Calib_Spectra_{filter}_ZWO.pkl"),
            qhy_pkl=os.path.join(calib_dir, f"Calib_Spectra_{filter}_QHY.pkl"),
            verbose=args.verbose
        )

        ZWO_calib_spectra = update_result["ZWO_calib_spectra"]
        QHY_calib_spectra = update_result["QHY_calib_spectra"]

        bin_avg_lookup = build_bin_averaged_calibrations(
            matches=matches,
            ZWO_calib_spectra=ZWO_calib_spectra,
            QHY_calib_spectra=QHY_calib_spectra,
            spectrum_class=spectrum,
            derivation_summary=update_result,
            verbose=args.verbose
        )

        matches = attach_bin_averaged_calibrations_to_science_matches(
            matches=matches,
            bin_avg_lookup=bin_avg_lookup,
            fallback=True,
            verbose=args.verbose
        )

        science_matches = matches["science_matches"]

        for i, row in enumerate(science_matches):
            grism_image = row["science_file"]
            match = row.get("averaged_calib_spectrum", None)

            try:
                if match is None:
                    raise ValueError("No averaged calibration spectrum assigned.")

                S = spectrum(grism_image, calib_spectrum=match, filter=filter)

                print('')
                print(f'Processing image {i + 1} of {len(science_matches)}: {S.object_name} {S.obs_date} ({S.exp_time} s):')
                print('')
                try:
                    trace_center, cutouts, subim = S.fit_trace(plot=args.show_steps, show_points = True, curved=True, method = 'gauss')
                except:
                    trace_center, cutouts, subim = S.fit_trace(plot=args.show_steps, show_points = True, curved=True, method = 'max')
                spec = S.extract_spectrum(
                    show_box=args.show_steps,
                    plot=args.show_steps,
                    curved_box=True
                )
                try:
                    telluric_pixel = S.fit_telluric_old(plot=args.show_steps, debugging = False)
                except:
                    S.telluric_fail = True
                    if args.verbose:
                        print('Telluric pixel fit failed, defaulting to median value')
                    if S.camera == 'ASI Camer':
                        x_guess = 3075.5 if S.filter == 'hrg' else 3000.6
                    elif S.camera == 'QHYCCD-Ca':
                        x_guess = 2975.5 if S.filter == 'hrg' else 2947.5
                    telluric_pixel = S.fit_telluric_old(x_guess = x_guess, plot = args.show_steps, debugging = False, manual_override = True) # Use this line instead if you want to manually override the telluric fit
                waves = S.wavelength_calibrate(plot=args.show_steps)
                cal_spec = S.gain_calibrate(plot=args.show_steps)

                dataframe = pd.DataFrame({
                    "Wavelength": S.waves,
                    "Flux": S.cal_spec
                })

                # img_name_csv = grism_image.split("/")[-1].replace(".fts.fz", ".csv")
                # output_path_csv = os.path.join(outpath, f'{img_name_csv}')

                # img_name_fits = grism_image.split("/")[-1].replace(".fts.fz", "_1d.fits")
                # output_path_fits = os.path.join(outpath, f'{img_name_fits}')

                img_name_csv = os.path.basename(grism_image).replace(".fts.fz", ".csv")
                output_path_csv = os.path.join(outpath, img_name_csv)

                img_name_fits = os.path.basename(grism_image).replace(".fts.fz", "_1d.fits")
                output_path_fits = os.path.join(outpath, img_name_fits)

                dataframe.to_csv(output_path_csv, index=False)
                save_1d_spectrum_fits(S, output_path_fits, match_row = row)

                if args.verbose:
                    print(f'Successfully wrote file to {output_path_csv} and {output_path_fits}')

            except Exception as e:
                print(f"Failed on {os.path.basename(grism_image)}: {e}")
                # traceback.print_exc() # uncomment for full error traceback
                try:
                    S.flagged = True
                except:
                    pass
                continue

if __name__ == "__main__":
    main()


# HD 201522 two traces
# lrg vs hrg mismatches