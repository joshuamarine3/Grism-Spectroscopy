"""
Drop-in helper functions for ModelSpectraFitting_Josh.ipynb.

Adds:
1. Stacking normalized CSV spectra by observing date.
2. Velocity-shift fitting with an initial guess and tolerance.
3. Result DataFrame construction for future iteration over all spectra.
4. Diagnostics for FOM versus model parameters.
5. Plot overlay that ingests the fit result and labels velocity shift.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from IPython.display import clear_output

C_KMS = 299792.458
HALPHA_REST = 6562.8

def read_spectrum_csv(path):
    """Read a normalized spectrum CSV robustly; returns wavelength, flux."""
    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=(0, 1),
        invalid_raise=False,
    )
    data = np.atleast_2d(data)
    wave = data[:, 0]
    flux = data[:, 1]
    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]
    order = np.argsort(wave)
    return wave[order], flux[order]


def extract_date_from_spectrum_filename(path):
    """Extract YYYY-MM-DD from names like Marine_cwl_Lam_Eri_hrg_120s_2025-12-22T06-26-09_norm.csv."""
    base = os.path.basename(path)
    match = re.search(r"(\d{4}-\d{2}-\d{2})T\d{2}-\d{2}-\d{2}", base)
    if match is None:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", base)
    if match is None:
        raise ValueError(f"Could not find date in filename: {base}")
    return match.group(1)


def stack_spectra_by_day(csv_files, stack_mode="sum", renormalize=True, continuum_percentile=50):
    """
    Stack normalized spectra by calendar date.

    Parameters
    ----------
    csv_files : list[str]
        Paths to *_norm.csv spectra.
    stack_mode : {'sum', 'mean'}
        'sum' literally adds normalized fluxes. 'mean' averages them.
    renormalize : bool
        If True, divides stacked flux by its median/percentile continuum so FOM still sees continuum near 1.
        This is usually what you want if stack_mode='sum'.
    continuum_percentile : float
        Percentile used for renormalization. 50 is median.

    Returns
    -------
    stacked : dict
        keyed by date string. Each value has wavelength, flux, files, n_spectra.
    """
    by_day = {}
    for path in sorted(csv_files):
        day = extract_date_from_spectrum_filename(path)
        by_day.setdefault(day, []).append(path)

    stacked = {}
    for day, files in by_day.items():
        waves_fluxes = [read_spectrum_csv(path) for path in files]

        # Use the first spectrum's grid as the common grid.
        # This keeps the output simple and avoids oversized union grids.
        base_wave = waves_fluxes[0][0]
        flux_grid = []

        for wave, flux in waves_fluxes:
            interp_flux = interp1d(
                wave,
                flux,
                bounds_error=False,
                fill_value=np.nan,
            )(base_wave)
            flux_grid.append(interp_flux)

        flux_grid = np.array(flux_grid)

        if stack_mode == "sum":
            stacked_flux = np.nansum(flux_grid, axis=0)
        elif stack_mode == "mean":
            stacked_flux = np.nanmean(flux_grid, axis=0)
        else:
            raise ValueError("stack_mode must be 'sum' or 'mean'")

        good = np.isfinite(base_wave) & np.isfinite(stacked_flux)
        base_wave = base_wave[good]
        stacked_flux = stacked_flux[good]

        if renormalize:
            cont = np.nanpercentile(stacked_flux, continuum_percentile)
            if np.isfinite(cont) and cont != 0:
                stacked_flux = stacked_flux / cont

        stacked[day] = {
            "date": day,
            "wavelength": base_wave,
            "flux": stacked_flux,
            "files": files,
            "n_spectra": len(files),
        }

    return stacked

def load_model(file_path):
    # Load the spectral data (wavelengths and fluxes) into a pandas dataframe
    dataframe = pd.read_table(file_path, sep="\\s+", names=["Wavelength","Flux","Checksum"], index_col=0, skiprows=5)

    # Extract model parameters from file name
    filename_params = (file_path.split('.')[1]).split('_')
    rho_0_exp = float(filename_params[5].split('e')[1].replace('m','-'))
    rho_0_mant = float('.'.join(filename_params[5].split('e')[0].split('d')))
    rho_0 = rho_0_mant*10**rho_0_exp
    r_disk = float('.'.join(filename_params[6].replace('rd','').split('p')))

    # Read the header information and filename parameters into a dictionary to access later
    with open(file_path,"r") as file:
        header_lines = file.readlines()[:5]
        file.close()

    # Split header lines by whitespace to separate keywords and values
    for i, line in enumerate(header_lines):
        line = [str.strip() for str in line.split(' ') if str]
        header_lines[i] = line

    # Store model parameters and header info in a dictionary
    header = {}
    header['Power Index'] = float('.'.join(filename_params[4].split('d'))) # disk density power law exponent
    header['rho_0'] = rho_0 # base disk density
    header['Disk Radius'] = r_disk # outer disk radius, in units of R_eq
    header['Wavelength'] = float(header_lines[1][3]) # Reference wavelength of some kind?
    header['Equivalent Width'] = float(header_lines[1][6]) # Equivalent width of the H-alpha line, in angstroms
    header['Pdiff'] = float(header_lines[1][10]) # Not really sure about this
    header['Rotational Velocity'] = float(header_lines[2][3]) # Rotational velocity of the star, in km/s
    header['v sin i'] = float(header_lines[2][6]) # Observable rotational velocity (based on inclination angle), in km/s
    header['Inclination Angle'] = float(header_lines[2][9]) # Inclination angle of the star/disk, in degrees
    header['Radius'] = float(header_lines[3][3]) # Radius of the star, in R_sun
    header['Mass'] = float(header_lines[3][6]) # Mass of the star, in M_sun
    header['T_eff'] = float(header_lines[3][9]) # Surface temperature of the star, in K
    header['log g'] = float(header_lines[3][12]) # log of star's surface gravity
    header['v_frac'] = float(header_lines[4][3]) # rotational velocity as fraction of critical velocity (?)
    header['R_pole'] = float(header_lines[4][9]) # Polar radius of the star, in R_sun
    header['R_eq'] = float(header_lines[4][12]) # Equatorial radius of the star, in R_sun

    return dataframe, header

# convolve a model file to a lower spectral resolution
def convolve_model(model_data, fwhm):
  #Load data and get array of wavelengths and fluxes
  model_wavelengths = np.array(model_data["Wavelength"])
  norm_model_flux = model_data['Flux']/model_data['Flux'][1] # Normalize flux to the continuum value

  d_wavelength = (model_wavelengths[len(model_wavelengths)-1]-model_wavelengths[0])/len(model_wavelengths-1) # in angstroms
  sigma = (fwhm.value/(2*np.sqrt(2*np.log(2))))/d_wavelength # in indices
  # print(sigma)

  gaussian_pos_x_vals = np.arange(0, 3*sigma, 1)
  gaussian_neg_x_vals = np.flip(np.arange(-1, -3*sigma, -1))
  # the positive and negative arrays are to ensure the values are symmetric around zero and will always be odd in number.
  gaussian_x_vals = np.append(gaussian_neg_x_vals, gaussian_pos_x_vals)# array covering 3 standard deviations on either side of zero in steps of 1.

  gaussian = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(gaussian_x_vals/sigma)**2/2) # normal distribution gaussian function value for each x value with std dev sigma. sum is about one.
  # print(sum(gaussian))
  norm_gaussian = gaussian/sum(gaussian) # force sum to be exactly one (very small change)

  # adding padding equal to the last value on either side to avoid edge distortions
  padded_norm_model_flux = np.append(np.repeat(norm_model_flux[1], len(norm_gaussian)/2), norm_model_flux)
  padded_norm_model_flux = np.append(padded_norm_model_flux, np.repeat(norm_model_flux[len(norm_model_flux)],len(norm_gaussian)/2))

  convolved_model_flux = np.convolve(padded_norm_model_flux, norm_gaussian,"valid")
  # ^valid means that only values where the two arrays fully overlap are in the output array
  # output is the same size as norm_model_flux

  return [model_wavelengths,convolved_model_flux]


def write_stacked_spectra(stacked, outdir, prefix="stacked", overwrite=True):
    """Write stacked spectra dict from stack_spectra_by_day() to CSV files."""
    os.makedirs(outdir, exist_ok=True)
    outfiles = []
    for day, spec in stacked.items():
        outfile = os.path.join(outdir, f"{prefix}_{day}_norm.csv")
        if os.path.exists(outfile) and not overwrite:
            raise FileExistsError(outfile)
        pd.DataFrame({
            "Wavelength": spec["wavelength"],
            "Flux": spec["flux"],
        }).to_csv(outfile, index=False)
        outfiles.append(outfile)
    return outfiles


def crop_arrays_to_range(data_wavelengths, data_fluxes, model_wavelengths):
    """Crop data arrays to model wavelength limits."""
    lo = np.nanmin(model_wavelengths)
    hi = np.nanmax(model_wavelengths)
    mask = (
        np.isfinite(data_wavelengths)
        & np.isfinite(data_fluxes)
        & (data_wavelengths >= lo)
        & (data_wavelengths <= hi)
    )
    return data_wavelengths[mask], data_fluxes[mask]


def load_spectrum_input(spectrum):
    """
    Accept either a CSV path or an already loaded spectrum.
    Supported loaded forms: dict with wavelength/flux, or [wave, flux] / (wave, flux).
    """
    if isinstance(spectrum, str):
        return read_spectrum_csv(spectrum)
    if isinstance(spectrum, dict):
        return np.asarray(spectrum["wavelength"], dtype=float), np.asarray(spectrum["flux"], dtype=float)
    return np.asarray(spectrum[0], dtype=float), np.asarray(spectrum[1], dtype=float)


def shift_wavelengths_by_velocity(wavelengths, velocity_shift_kms):
    """Positive velocity redshifts model wavelengths."""
    return np.asarray(wavelengths, dtype=float) * (1 + velocity_shift_kms / C_KMS)


def fom_for_velocity_shift(model_wavelengths, model_fluxes, data_wavelengths, data_fluxes,
                           velocity_shift_kms=0.0, core_weighted=False,
                           fit_window=None, model_center=HALPHA_REST):
    """Compute Sigut-like FOM after shifting model by velocity_shift_kms."""
    shifted_wave = shift_wavelengths_by_velocity(model_wavelengths, velocity_shift_kms)

    data_wave, data_flux = crop_arrays_to_range(data_wavelengths, data_fluxes, shifted_wave)

    if fit_window is not None:
        window_center = model_center * (1 + velocity_shift_kms / C_KMS)
        half_width = fit_window / 2
        win = (data_wave >= window_center - half_width) & (data_wave <= window_center + half_width)
        data_wave = data_wave[win]
        data_flux = data_flux[win]

    if len(data_wave) < 3:
        return np.inf

    interp_model_flux = interp1d(
        shifted_wave,
        model_fluxes,
        bounds_error=False,
        fill_value=np.nan,
    )(data_wave)

    good = np.isfinite(interp_model_flux) & np.isfinite(data_flux) & (data_flux != 0)
    interp_model_flux = interp_model_flux[good]
    data_flux = data_flux[good]

    if len(data_flux) < 3:
        return np.inf

    if core_weighted:
        weights = np.abs(interp_model_flux - 1)
        if np.nansum(weights) == 0:
            weights = np.ones_like(interp_model_flux)
    else:
        weights = np.ones_like(interp_model_flux)

    return (1 / np.sum(weights)) * np.sum(weights * np.abs((interp_model_flux - data_flux) / data_flux)) * 100


def test_model(model, spectrum, fwhm, core_weighted=False,
               velocity_guess_kms=0.0, velocity_tolerance_kms=100.0,
               fit_velocity=True, fit_window=None, model_center=HALPHA_REST,
               return_result=True, model_index=None, n_models=None,
               progress_every=5):
    """
    Fit one model to one spectrum, optionally optimizing a velocity shift.

    Requires your notebook's load_model() and convolve_model() functions to already exist.
    """
    if model_index is not None and n_models is not None and model_index % progress_every == 0:
        clear_output(wait=True)
        print(f"Model {model_index}/{n_models}")

    model_data, header = load_model(model)
    model_wavelengths, convolved_model_flux = convolve_model(model_data, fwhm)
    data_wavelengths, data_fluxes = load_spectrum_input(spectrum)

    if fit_velocity:
        lo = velocity_guess_kms - velocity_tolerance_kms
        hi = velocity_guess_kms + velocity_tolerance_kms
        result = minimize_scalar(
            lambda v: fom_for_velocity_shift(
                model_wavelengths,
                convolved_model_flux,
                data_wavelengths,
                data_fluxes,
                velocity_shift_kms=v,
                core_weighted=core_weighted,
                fit_window=fit_window,
                model_center=model_center,
            ),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 0.01},
        )
        velocity_shift_kms = float(result.x)
        figure_of_merit = float(result.fun)
    else:
        velocity_shift_kms = float(velocity_guess_kms)
        figure_of_merit = float(fom_for_velocity_shift(
            model_wavelengths,
            convolved_model_flux,
            data_wavelengths,
            data_fluxes,
            velocity_shift_kms=velocity_shift_kms,
            core_weighted=core_weighted,
            fit_window=fit_window,
            model_center=model_center,
        ))

    lambda_shift = model_center * velocity_shift_kms / C_KMS

    if not return_result:
        return figure_of_merit

    return {
        "model": model,
        "fom": figure_of_merit,
        "velocity_shift_kms": velocity_shift_kms,
        "lambda_shift_A": lambda_shift,
        "header": header,
    }


def fit_models_for_spectrum(spectrum, models, fwhm, core_weighted=True,
                            velocity_guess_kms=0.0, velocity_tolerance_kms=100.0,
                            fit_velocity=True, fit_window=80.0, model_center=HALPHA_REST):
    """Fit all models to one spectrum and return a sorted DataFrame."""
    rows = []
    for j, model in enumerate(models):
        result = test_model(
            model,
            spectrum,
            fwhm,
            core_weighted=core_weighted,
            velocity_guess_kms=velocity_guess_kms,
            velocity_tolerance_kms=velocity_tolerance_kms,
            fit_velocity=fit_velocity,
            fit_window=fit_window,
            model_center=model_center,
            return_result=True,
            model_index=j,
            n_models=len(models),
        )
        h = result["header"]
        rows.append({
            "model": result["model"],
            "fom": result["fom"],
            "velocity_shift_kms": result["velocity_shift_kms"],
            "lambda_shift_A": result["lambda_shift_A"],
            "Mass": h.get("Mass", np.nan),
            "Power Index": h.get("Power Index", np.nan),
            "rho_0": h.get("rho_0", np.nan),
            "Disk Radius": h.get("Disk Radius", np.nan),
            "Inclination Angle": h.get("Inclination Angle", np.nan),
            "Radius": h.get("Radius", np.nan),
            "Equivalent Width": h.get("Equivalent Width", np.nan),
            "Rotational Velocity": h.get("Rotational Velocity", np.nan),
        })

    results = pd.DataFrame(rows).sort_values("fom").reset_index(drop=True)
    results["f_rel"] = results["fom"] / results["fom"].min()
    return results


def fit_all_spectra(spectra, models, fwhm, **fit_kwargs):
    """
    Future-facing wrapper: fit many spectra.

    spectra can be:
    - dict from stack_spectra_by_day()
    - list of CSV paths
    - list of loaded [wave, flux] pairs
    """
    all_results = []

    if isinstance(spectra, dict):
        iterable = [(key, spec) for key, spec in spectra.items()]
    else:
        iterable = [(str(i), spec) for i, spec in enumerate(spectra)]

    for spec_id, spectrum in iterable:
        print(f"Fitting spectrum {spec_id}")
        results = fit_models_for_spectrum(spectrum, models, fwhm, **fit_kwargs)
        results.insert(0, "spectrum_id", spec_id)
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


def plot_model_overlay(spectrum, dataframe, header, fwhm, fit_result=None,
                       velocity_shift_kms=0.0, xlim=None, ylim=None):
    """Plot data, shifted unconvolved model, shifted convolved model, and parameter labels."""
    if fit_result is not None:
        velocity_shift_kms = fit_result.get("velocity_shift_kms", velocity_shift_kms)
        fom = fit_result.get("fom", None)
    else:
        fom = None

    norm_flux = np.asarray(dataframe["Flux"], dtype=float) / np.asarray(dataframe["Flux"], dtype=float)[1]
    original_wave = shift_wavelengths_by_velocity(np.asarray(dataframe["Wavelength"], dtype=float), velocity_shift_kms)

    model_wave, convolved_flux = convolve_model(dataframe, fwhm)
    model_wave = shift_wavelengths_by_velocity(model_wave, velocity_shift_kms)

    data_wave, data_flux = load_spectrum_input(spectrum)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(original_wave, norm_flux, color="C0", linestyle=":", linewidth=1.5, label="Unconvolved model")
    ax.plot(model_wave, convolved_flux, color="C1", linewidth=2, label="Convolved model")
    ax.scatter(data_wave, data_flux, facecolors="none", edgecolors="red", marker="o", label="Observed spectrum")

    ax.set_xlabel("Wavelength (Angstroms)")
    ax.set_ylabel("Continuum-normalized flux")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    box = dict(boxstyle="round", facecolor="lightsteelblue", alpha=0.5)

    left_text = "\n".join((
        r"$M_{star} = %.2f \ M_{\odot}$" % header["Mass"],
        r"$n = %.2f$" % header["Power Index"],
        r"$\rho_0 = %.2e$" % header["rho_0"],
        r"$R_d = %.2f \ R_{*}$" % header["Disk Radius"],
        r"$i = %.0f^\circ$" % header["Inclination Angle"],
    ))
    ax.text(0.02, 0.95, left_text, transform=ax.transAxes, va="top", ha="left", bbox=box)

    lambda_shift = HALPHA_REST * velocity_shift_kms / C_KMS
    right_lines = [
        r"$R_{star} = %.2f \ R_{\odot}$" % header["Radius"],
        r"EW $= %.2f \ \AA$" % header["Equivalent Width"],
        r"$v_{\mathrm{rot}} = %.2f \ \mathrm{km/s}$" % header["Rotational Velocity"],
        r"$\Delta \lambda = %.3f \ \AA$" % lambda_shift,
        r"$v_{\mathrm{shift}} = %.2f \ \mathrm{km/s}$" % velocity_shift_kms,
    ]
    if fom is not None:
        right_lines.append(r"FOM $= %.4f$" % fom)

    ax.text(0.98, 0.95, "\n".join(right_lines), transform=ax.transAxes, va="top", ha="right", bbox=box)
    ax.legend()
    plt.show()


def plot_fom_parameter_diagnostics(results, accepted_fom=1.15):
    """Visualize FOM versus each fitted model parameter."""
    params = ["Mass", "Power Index", "rho_0", "Disk Radius", "Inclination Angle", "velocity_shift_kms"]
    for param in params:
        if param not in results.columns:
            continue
        plt.figure(figsize=(7, 5))
        y = results["f_rel"] if "f_rel" in results.columns else results["fom"] / results["fom"].min()
        plt.scatter(results[param], y, facecolors="none", edgecolors="C0")
        plt.axhline(accepted_fom, linestyle="--", label=f"Accepted threshold = {accepted_fom}")
        if param == "rho_0":
            plt.xscale("log")
        plt.xlabel(param)
        plt.ylabel("Relative FOM")
        plt.title(f"Relative FOM vs {param}")
        plt.legend()
        plt.show()


def summarize_best_fit(results, n=10):
    """Show compact best-fit table."""
    cols = [
        "fom", "f_rel", "velocity_shift_kms", "lambda_shift_A",
        "Mass", "Power Index", "rho_0", "Disk Radius", "Inclination Angle", "model"
    ]
    cols = [c for c in cols if c in results.columns]
    return results.head(n)[cols]
