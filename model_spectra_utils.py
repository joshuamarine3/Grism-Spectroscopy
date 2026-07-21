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
from IPython.display import clear_output, display

from tqdm.auto import tqdm
import multiprocessing as mp

from mpl_toolkits.mplot3d import Axes3D

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

from astropy.stats import sigma_clip

def normalize_spectrum(
    spec,
    continuum_windows=((6510, 6540), (6590, 6620)),
    poly_order=1,
    sigma=3.0,
    maxiters=5,
    plot=False,
):
    try:
        wave = np.asarray(spec["wave"], dtype=float)
        flux = np.asarray(spec["flux"], dtype=float)
    except:
        wave = np.asarray(spec["wavelength"], dtype=float)
        flux = np.asarray(spec["flux"], dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    cont_mask = np.zeros_like(wave, dtype=bool)

    for wmin, wmax in continuum_windows:
        cont_mask |= (wave >= wmin) & (wave <= wmax)

    if np.sum(cont_mask) < poly_order + 2:
        raise ValueError(
            f"Not enough continuum points for {spec.get('date-obs', 'unknown spectrum')}"
        )

    cont_wave = wave[cont_mask]
    cont_flux = flux[cont_mask]

    clipped = sigma_clip(
        cont_flux,
        sigma=sigma,
        maxiters=maxiters,
        masked=True,
    )

    fit_wave = cont_wave[~clipped.mask]
    fit_flux = cont_flux[~clipped.mask]

    coeffs = np.polyfit(fit_wave, fit_flux, poly_order)
    continuum = np.polyval(coeffs, wave)

    valid = np.isfinite(continuum) & (continuum != 0)

    norm_flux = np.full_like(flux, np.nan)
    norm_flux[valid] = flux[valid] / continuum[valid]

    out = dict(spec)
    out["wave"] = wave
    out["flux_raw"] = flux
    out["flux"] = norm_flux
    out["continuum"] = continuum
    out["continuum_coeffs"] = coeffs
    out["continuum_windows"] = continuum_windows

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(wave, flux, lw=1, label="Raw spectrum")
        plt.plot(wave, continuum, lw=2, label="Linear continuum fit")
        plt.scatter(fit_wave, fit_flux, s=10, label="Used continuum points")
        for wmin, wmax in continuum_windows:
            plt.axvspan(wmin, wmax, alpha=0.15)
        plt.xlabel("Wavelength [Å]")
        plt.ylabel("Flux")
        plt.title(spec.get("date_obs", "BeSS spectrum"))
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(wave, norm_flux, lw=1)
        plt.axhline(1, color="k", ls="--", alpha=0.5)
        plt.xlabel("Wavelength [Å]")
        plt.ylabel("Normalized flux")
        plt.title("Continuum-normalized spectrum")
        plt.tight_layout()
        plt.show()

    return out


def stack_spectra_by_day(csv_files, stack_mode="sum", renormalize=False, continuum_percentile=50):
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
  fwhm = fwhm.value if hasattr(fwhm, "value") else float(fwhm)
  sigma = (fwhm/(2*np.sqrt(2*np.log(2))))/d_wavelength # in indices
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
        try:
            return np.asarray(spectrum["wavelength"], dtype=float), np.asarray(spectrum["flux"], dtype=float)
        except KeyError:
            return np.asarray(spectrum["wave"], dtype=float), np.asarray(spectrum["flux"], dtype=float)
    return np.asarray(spectrum[0], dtype=float), np.asarray(spectrum[1], dtype=float)


def shift_wavelengths_by_velocity(wavelengths, velocity_shift_kms):
    """Positive velocity redshifts model wavelengths."""
    return np.asarray(wavelengths, dtype=float) * (1 + velocity_shift_kms / C_KMS)

def compute_profile_shape_weights(
    wave,
    data_flux,
    model_flux,
    use_flux_weight=True,
    use_slope_weight=True,
    use_curvature_weight=True,
    min_weight=0.05,
):
    terms = {}
    components = []

    if use_flux_weight:
        flux_term = np.abs(data_flux - 1.0)
        if np.nanmax(flux_term) > 0:
            flux_term = flux_term / np.nanmax(flux_term)
        terms["flux_weight"] = flux_term
        components.append(flux_term)

    if use_slope_weight:
        dF = np.gradient(model_flux, wave)
        slope_term = np.abs(dF)
        if np.nanmax(slope_term) > 0:
            slope_term = slope_term / np.nanmax(slope_term)
        terms["slope_weight"] = slope_term
        components.append(slope_term)

    if use_curvature_weight:
        dF = np.gradient(model_flux, wave)
        d2F = np.gradient(dF, wave)
        curvature_term = np.abs(d2F)
        if np.nanmax(curvature_term) > 0:
            curvature_term = curvature_term / np.nanmax(curvature_term)
        terms["curvature_weight"] = curvature_term
        components.append(curvature_term)

    if len(components) == 0:
        total_weight = np.ones_like(model_flux)
    else:
        total_weight = np.sqrt(np.sum([c**2 for c in components], axis=0))

    if np.nanmax(total_weight) > 0:
        total_weight = total_weight / np.nanmax(total_weight)

    total_weight = np.maximum(total_weight, min_weight)

    terms["total_weight"] = total_weight

    return total_weight, terms

def fom_for_velocity_shift(
    model_wavelengths,
    model_fluxes,
    data_wavelengths,
    data_fluxes,
    velocity_shift_kms=0.0,
    fit_window=50.0,
    model_center=HALPHA_REST,
    core_weighted=True,
    return_diagnostics=False,
):

    shifted_wave = shift_wavelengths_by_velocity(
        model_wavelengths,
        velocity_shift_kms
    )

    data_wave, data_flux = crop_arrays_to_range(
        data_wavelengths,
        data_fluxes,
        shifted_wave
    )

    if fit_window is not None:
        window_center = model_center * (1 + velocity_shift_kms / C_KMS)
        half_width = fit_window / 2

        win = (
            (data_wave >= window_center - half_width) &
            (data_wave <= window_center + half_width)
        )

        data_wave = data_wave[win]
        data_flux = data_flux[win]

    if len(data_wave) < 3:
        if return_diagnostics:
            return np.inf, {}
        return np.inf

    interp_model_flux = interp1d(
        shifted_wave,
        model_fluxes,
        bounds_error=False,
        fill_value=np.nan,
    )(data_wave)

    good = (
        np.isfinite(interp_model_flux) &
        np.isfinite(data_flux) &
        np.isfinite(data_wave) &
        (data_flux != 0)
    )

    data_wave = data_wave[good]
    data_flux = data_flux[good]
    interp_model_flux = interp_model_flux[good]

    if len(data_flux) < 3:
        if return_diagnostics:
            return np.inf, {}
        return np.inf

    # if core_weighted:
    #     weights = np.abs(data_flux - 1.0)

    #     if np.sum(weights) == 0:
    #         weights = np.ones_like(data_flux)
    # else:
    #     weights = np.ones_like(data_flux)

    if core_weighted:
        weights, weight_terms = compute_profile_shape_weights(
            data_wave,
            data_flux,
            interp_model_flux,
            use_flux_weight=True,
            use_slope_weight=True,
            use_curvature_weight=True,
            min_weight=0.25,
        )
    else:
        weights = np.ones_like(data_flux)
        weight_terms = {"total_weight": weights}

    fom = (
        np.sum(weights * np.abs((interp_model_flux - data_flux) / data_flux))
        / np.sum(weights)
        * 100
    )

    if return_diagnostics:
        diagnostics = {
            "data_wave": data_wave,
            "data_flux": data_flux,
            "model_flux": interp_model_flux,
            "weights": weights,
            "velocity_shift_kms": velocity_shift_kms,
            "fit_window": fit_window,
            "model_center": model_center,
            "fom": fom,
        }

        return fom, diagnostics

    return fom

def plot_fom_diagnostics(diagnostics, cmap="inferno"):
    data_wave = diagnostics["data_wave"]
    data_flux = diagnostics["data_flux"]
    model_flux = diagnostics["model_flux"]
    weights = diagnostics["weights"]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
        constrained_layout=True
    )

    ax = axes[0]

    sc = ax.scatter(
        data_wave,
        data_flux,
        c=weights,
        cmap=cmap,
        s=28,
        label="Observed spectrum"
    )

    ax.plot(
        data_wave,
        model_flux,
        color="C1",
        linewidth=2,
        label="Shifted model"
    )

    ax.set_ylabel("Normalized flux")
    ax.set_title(
        f"FOM diagnostic: FOM = {diagnostics['fom']:.3f}, "
        f"v = {diagnostics['velocity_shift_kms']:.2f} km/s"
    )
    ax.legend()

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("FOM weight")

    axes[1].plot(data_wave, weights, color="C2")
    axes[1].set_ylabel("Weight")

    for a in axes:
        a.axvline(data_wave.min(), color="k", linestyle=":", alpha=0.4)
        a.axvline(data_wave.max(), color="k", linestyle=":", alpha=0.4)

    plt.show()


def test_model(model, spectrum, fwhm, core_weighted=True,
               velocity_guess_kms=0.0, velocity_tolerance_kms=100.0,
               fit_velocity=True, fit_window=60.0, model_center=6562.8,
               return_result=True, model_index=None, n_models=None,
               progress_every=100):
    """
    Fit one model to one spectrum.
    No printing here; progress is handled by fit_models_for_spectrum().
    """

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

        figure_of_merit = float(
            fom_for_velocity_shift(
                model_wavelengths,
                convolved_model_flux,
                data_wavelengths,
                data_fluxes,
                velocity_shift_kms=velocity_shift_kms,
                core_weighted=core_weighted,
                fit_window=fit_window,
                model_center=model_center,
            )
        )

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
                            fit_velocity=True, fit_window=60.0,
                            model_center=6562.8,
                            skip_bad_models=True,
                            verbose=1,
                            spectrum_id=None,
                            tqdm_position = 1):
    """
    Fit all models to one spectrum.

    verbose:
        0 = silent
        1 = per-model tqdm progress bar
    """

    rows = []
    bad_models = []

    if verbose >= 1:
        model_iter = tqdm(
            enumerate(models),
            total=len(models),
            desc=f"Models {spectrum_id}" if spectrum_id is not None else "Models",
            unit="model",
            leave=False,
            position=tqdm_position,
        )
    else:
        model_iter = enumerate(models)

    for j, model in model_iter:

        try:
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
                "model_index": j,
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
                "T_eff": h.get("T_eff", np.nan),
            })

        except Exception as e:
            bad_models.append({
                "model_index": j,
                "model": model,
                "error_type": type(e).__name__,
                "error": str(e),
            })

            if not skip_bad_models:
                raise

    bad_models = pd.DataFrame(bad_models)

    if len(rows) == 0:
        if verbose >= 1:
            print("No models fit successfully.")
            if len(bad_models) > 0:
                display(bad_models.head(20))
        return pd.DataFrame(), bad_models

    results = pd.DataFrame(rows).sort_values("fom").reset_index(drop=True)
    results["f_rel"] = results["fom"] / results["fom"].min()

    if verbose >= 1 and len(bad_models) > 0:
        print(f"Skipped {len(bad_models)} bad models.")
        display(bad_models.head(20))

    return results, bad_models


def fit_all_spectra(spectra, models, fwhm, verbose=1, **fit_kwargs):
    """
    Fit all spectra against all models.

    Parameters
    ----------
    spectra : dict or iterable
        Dictionary of spectra keyed by spectrum_id, or iterable of spectra.
    models : list
        List of model filenames.
    fwhm : float
        Instrumental FWHM. Ignored for spectra containing an 'R' keyword.
    verbose : int
        0 = silent
        1 = overall spectrum progress bar
        2 = spectrum progress bar + per-model progress bars

    Returns
    -------
    pandas.DataFrame
        Combined results for every spectrum.
    """

    all_results = []

    if isinstance(spectra, dict):
        iterable = list(spectra.items())
    else:
        iterable = [(str(i), spec) for i, spec in enumerate(spectra)]

    if verbose >= 1:
        spectrum_iter = tqdm(
            iterable,
            desc="Fitting spectra",
            unit="spec",
        )
    else:
        spectrum_iter = iterable

    for spec_id, spectrum in spectrum_iter:

        spectrum_fwhm = fwhm

        if isinstance(spectrum, dict) and "R" in spectrum:
            R = spectrum["R"]

            if R is not None and np.isfinite(R) and R > 0:
                spectrum_fwhm = HALPHA_REST / R
            else:
                raise ValueError(
                    f"Invalid R value for spectrum {spec_id}: {R}"
                )

        results, _ = fit_models_for_spectrum(
            spectrum,
            models,
            spectrum_fwhm,
            **fit_kwargs,
            verbose=1 if verbose >= 2 else 0,
            spectrum_id=spec_id,
        )

        if len(results) == 0:
            continue

        results = results.copy()

        results.insert(0, "spectrum_id", spec_id)
        results.insert(1, "fwhm", spectrum_fwhm)

        all_results.append(results)

        if verbose >= 1 and hasattr(spectrum_iter, "set_postfix"):
            spectrum_iter.set_postfix(
                spectrum=str(spec_id),
                best_fom=f"{results['fom'].min():.4g}",
            )

    if len(all_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def plot_model_overlay(spectrum, dataframe, header, fwhm, fit_result=None,
                       velocity_shift_kms=0.0, xlim=None, ylim=None, verbose=False, fit_window = 50.0, core_weighted = True):
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

    if verbose:
        data_wave, data_flux = load_spectrum_input(spectrum)
        model_wave, convolved_flux = convolve_model(dataframe, fwhm)

        _, diagnostics = fom_for_velocity_shift(
            model_wave,
            convolved_flux,
            data_wave,
            data_flux,
            velocity_shift_kms=velocity_shift_kms,
            fit_window=fit_window,
            model_center=HALPHA_REST,
            core_weighted=core_weighted,
            return_diagnostics=True,
        )

        plot_fom_diagnostics(diagnostics)

    ax.text(0.98, 0.95, "\n".join(right_lines), transform=ax.transAxes, va="top", ha="right", bbox=box)
    ax.legend()
    plt.show()


def plot_parameter_diagnostics(results, accepted_fom=1.15, bins="auto"):
    """
    Plot histograms of model parameters for all accepted models
    (f_rel <= accepted_fom).

    Parameters
    ----------
    results : pandas.DataFrame
        Output from fit_models_for_spectrum().
    accepted_fom : float
        Relative FOM threshold (e.g. 1.15).
    bins : int or str
        Histogram bin specification.
    """

    accepted = results[results["f_rel"] <= accepted_fom]

    if len(accepted) == 0:
        print("No accepted models.")
        return

    params = [
        "Mass",
        "Power Index",
        "rho_0",
        "Disk Radius",
        "Inclination Angle",
        "velocity_shift_kms",
    ]

    for param in params:

        if param not in accepted.columns:
            continue

        plt.figure(figsize=(7,5))

        values = accepted[param].dropna()

        if len(values) == 0:
            plt.close()
            continue

        if param == "rho_0":
            plt.hist(
                np.log10(values),
                bins=bins,
                edgecolor="black"
            )
            plt.xlabel(r"$\log_{10}(\rho_0)$")
        else:
            plt.hist(
                values,
                bins=bins,
                edgecolor="black"
            )
            plt.xlabel(param)

        plt.ylabel("Count")
        plt.title(
            f"{param}\nAccepted models ($f_{{rel}} \\leq {accepted_fom}$)"
        )

        plt.tight_layout()
        plt.show()


def summarize_best_fit(results, n=10):
    """Show compact best-fit table."""
    cols = [
        "fom", "f_rel", "velocity_shift_kms", "lambda_shift_A",
        "Mass", "Power Index", "rho_0", "Disk Radius", "Inclination Angle", "model"
    ]
    cols = [c for c in cols if c in results.columns]
    return results.head(n)[cols]

def _fit_single_model_worker(args):
    """Worker task that fits exactly ONE model to ONE spectrum."""
    spec_id, spectrum, model, fwhm, j, n_models, fit_kwargs = args
    
    spectrum_fwhm = fwhm
    if isinstance(spectrum, dict) and "R" in spectrum:
        R = spectrum["R"]
        if R is not None and np.isfinite(R) and R > 0:
            spectrum_fwhm = HALPHA_REST / R
        else:
            return spec_id, None, {
                "model_index": j,
                "model": model,
                "error_type": "ValueError",
                "error": f"Invalid R value for spectrum {spec_id}: {R}",
            }, spectrum_fwhm

    try:
        result = test_model(
            model,
            spectrum,
            spectrum_fwhm,
            return_result=True,
            model_index=j,
            n_models=n_models,
            **fit_kwargs
        )
        h = result["header"]
        row = {
            "model": result["model"],
            "model_index": j,
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
            "T_eff": h.get("T_eff", np.nan),
        }
        return spec_id, row, None, spectrum_fwhm

    except Exception as e:
        bad_model = {
            "model_index": j,
            "model": model,
            "error_type": type(e).__name__,
            "error": str(e),
        }
        return spec_id, None, bad_model, spectrum_fwhm


def fit_all_spectra_parallel(
    spectra,
    models,
    fwhm=None,
    max_workers=None,
    verbose=1,
    chunksize=1,
    **fit_kwargs,
):
    """
    Multiprocessing version tracking both Spectrum and Model-level progress.
    """
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    if isinstance(spectra, dict):
        iterable = list(spectra.items())
    else:
        iterable = [(str(i), spec) for i, spec in enumerate(spectra)]

    # Flatten the tasks: one task per spectrum per model
    tasks = []
    for spec_id, spectrum in iterable:
        for j, model in enumerate(models):
            tasks.append((spec_id, spectrum, model, fwhm, j, len(models), fit_kwargs))

    total_models_to_fit = len(tasks)
    
    # Dictionaries to aggregate results per spectrum
    spectrum_rows = {spec_id: [] for spec_id, _ in iterable}
    spectrum_bad = {spec_id: [] for spec_id, _ in iterable}
    spectrum_fwhms = {}

    ctx = mp.get_context("spawn")

    # Set up nested progress bars in the main process
    if verbose >= 1:
        spec_bar = tqdm(total=len(iterable), desc="Spectra Completed", unit="spec", position=0, leave=True)
        model_bar = tqdm(total=total_models_to_fit, desc="Total Models Fit", unit="model", position=1, leave=True)
    else:
        spec_bar, model_bar = None, None

    completed_spectra = set()

    with ctx.Pool(processes=max_workers) as pool:
        result_iter = pool.imap_unordered(
            _fit_single_model_worker,
            tasks,
            chunksize=chunksize,
        )

        for spec_id, row, bad_model, spec_fwhm in result_iter:
            spectrum_fwhms[spec_id] = spec_fwhm
            
            if row is not None:
                spectrum_rows[spec_id].append(row)
            if bad_model is not None:
                spectrum_bad[spec_id].append(bad_model)

            if model_bar:
                model_bar.update(1)

            # Check if all models for this spectrum are done
            total_expected_for_spec = len(models)
            total_received_for_spec = len(spectrum_rows[spec_id]) + len(spectrum_bad[spec_id])
            
            if total_received_for_spec == total_expected_for_spec and spec_id not in completed_spectra:
                completed_spectra.add(spec_id)
                if spec_bar:
                    spec_bar.update(1)
                    # Show the best FOM found so far for this finished spectrum
                    foms = [r["fom"] for r in spectrum_rows[spec_id] if "fom" in r]
                    best_fom = min(foms) if foms else np.nan
                    spec_bar.set_postfix(spectrum=str(spec_id), best_fom=f"{best_fom:.4g}")

    if spec_bar: spec_bar.close()
    if model_bar: model_bar.close()

    # Reconstruct the DataFrames exactly like your serial version
    all_results = []
    all_bad_models = []

    for spec_id, _ in iterable:
        rows = spectrum_rows[spec_id]
        bads = spectrum_bad[spec_id]
        spec_fwhm = spectrum_fwhms.get(spec_id, fwhm)

        if len(rows) > 0:
            df_res = pd.DataFrame(rows).sort_values("fom").reset_index(drop=True)
            df_res["f_rel"] = df_res["fom"] / df_res["fom"].min()
            df_res.insert(0, "spectrum_id", spec_id)
            df_res.insert(1, "fwhm", spec_fwhm)
            all_results.append(df_res)

        if len(bads) > 0:
            df_bad = pd.DataFrame(bads)
            df_bad.insert(0, "spectrum_id", spec_id)
            df_bad.insert(1, "fwhm", spec_fwhm)
            all_bad_models.append(df_bad)

    all_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    all_bad_models = pd.concat(all_bad_models, ignore_index=True) if all_bad_models else pd.DataFrame()

    return all_results, all_bad_models

def _pretty_param_label(param):
    labels = {
        "log_rho_0": r"$\log_{10}(\rho_0)$",
        "rho_0": r"$\rho_0$",
        "Disk Radius": r"Disk radius $R_d$",
        "Power Index": r"Power index $n$",
        "Inclination Angle": r"Inclination $i$",
        "velocity_shift_kms": r"$v_{\rm shift}$ [km/s]",
        "lambda_shift_A": r"$\Delta\lambda$ [$\AA$]",
        "Equivalent Width": r"EW [$\AA$]",
        "Rotational Velocity": r"$v_{\rm rot}$ [km/s]",
        "Mass": r"Mass [$M_\odot$]",
    }

    return labels.get(param, param)


def plot_parameter_time_distribution(
    all_results,
    params=None,
    accepted_fom=1.15,
    show_summary=True,
    show_heatmap=True,
    show_3d_surface=False,
    show_3d_waterfall=False,
    bins=30,
    cmap="inferno",
    figsize_per_panel=(4.5, 3.4),
    max_xtick_labels=8,
    elev=28,
    azim=-60,
    tbin=None,
    date_range = None
):
    """
    Plot accepted-model parameter distributions over time.

    Parameters
    ----------
    all_results : pandas.DataFrame
        Combined fit results containing spectrum_id, fom/f_rel, and model
        parameters.

    params : list[str] or None
        Parameters to plot.

    accepted_fom : float
        Maximum relative FOM accepted.

    show_summary, show_heatmap, show_3d_surface, show_3d_waterfall : bool
        Select plot types.

    bins : int
        Number of parameter-value histogram bins.

    tbin : float or None
        Time-bin width in days.

        If None, each spectrum is plotted independently.

        If specified, the complete sequence of time bins between the first and
        last observations is shown. Bins without data remain empty, producing
        visible gaps in the plots.
    """

    if params is None:
        params = [
            "Mass",
            "log_rho_0",
            "Disk Radius",
            "Power Index",
            "Inclination Angle",
            "velocity_shift_kms",
        ]

    df = all_results.copy()

    required_columns = {"spectrum_id", "fom"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            "all_results is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Recalculate relative FOM independently for every spectrum if needed.
    if "f_rel" not in df.columns:
        minimum_fom = df.groupby("spectrum_id")["fom"].transform("min")
        df["f_rel"] = df["fom"] / minimum_fom

    accepted = df.loc[df["f_rel"] <= accepted_fom].copy()

    if accepted.empty:
        print("No accepted models.")
        return

    # Add log-density parameter where possible.
    if "rho_0" in accepted.columns:
        positive_density = accepted["rho_0"] > 0

        accepted["log_rho_0"] = np.nan
        accepted.loc[positive_density, "log_rho_0"] = np.log10(
            accepted.loc[positive_density, "rho_0"]
        )

        params = [
            "log_rho_0" if param == "rho_0" else param
            for param in params
        ]

    plot_modes = []

    if show_summary:
        plot_modes.append("summary")
    if show_heatmap:
        plot_modes.append("heatmap")
    if show_3d_surface:
        plot_modes.append("3d_surface")
    if show_3d_waterfall:
        plot_modes.append("3d_waterfall")

    if not plot_modes:
        print("Nothing selected to plot.")
        return

    # ------------------------------------------------------------------
    # Construct the time axis
    # ------------------------------------------------------------------

    accepted["datetime"] = pd.to_datetime(
        accepted["spectrum_id"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Restrict to requested date range
    # --------------------------------------------------------

    if date_range is not None:

        start_date, end_date = date_range

        if start_date is not None:
            start_date = pd.to_datetime(start_date)

        if end_date is not None:
            end_date = pd.to_datetime(end_date)

        mask = np.ones(len(accepted), dtype=bool)

        if start_date is not None:
            mask &= accepted["datetime"] >= start_date

        if end_date is not None:
            mask &= accepted["datetime"] <= end_date

        accepted = accepted.loc[mask].copy()

        if accepted.empty:
            print("No spectra fall within the requested date range.")
            return

    invalid_dates = accepted["datetime"].isna()

    if invalid_dates.any():
        invalid_ids = accepted.loc[
            invalid_dates,
            "spectrum_id",
        ].drop_duplicates()

        print(
            f"Removing {invalid_dates.sum()} rows with unparseable dates. "
            f"Examples: {invalid_ids.head(5).tolist()}"
        )

        accepted = accepted.loc[~invalid_dates].copy()

    if accepted.empty:
        print("No spectra have valid dates.")
        return

    if tbin is None:
        accepted["time_group"] = accepted["spectrum_id"]

        groups = sorted(
            accepted["time_group"].dropna().unique(),
            key=lambda value: pd.to_datetime(str(value)),
        )

        xlabels = [str(group) for group in groups]
        has_data = np.ones(len(groups), dtype=bool)
        time_axis_label = "Spectrum"

    else:
        if not np.isscalar(tbin) or not np.isfinite(tbin) or tbin <= 0:
            raise ValueError("tbin must be a positive number of days.")

        # Anchor bins at midnight on the date of the earliest observation.
        if date_range is not None and start_date is not None:
            t0 = start_date.normalize()
        else:
            t0 = accepted["datetime"].min().normalize()

        elapsed_days = (
            accepted["datetime"] - t0
        ).dt.total_seconds() / 86400.0

        accepted["time_group"] = np.floor(
            elapsed_days / float(tbin)
        ).astype(int)

        # Include every bin from the beginning through the last observation.
        # This is what preserves gaps in data coverage.
        final_group = int(accepted["time_group"].max())
        groups = np.arange(final_group + 1, dtype=int)

        occupied_groups = set(accepted["time_group"].unique())
        has_data = np.array(
            [group in occupied_groups for group in groups],
            dtype=bool,
        )

        xlabels = []

        for group in groups:
            start = t0 + pd.Timedelta(days=group * tbin)
            end = start + pd.Timedelta(days=tbin)

            # Use a single centered date for compact labels.
            center = start + pd.Timedelta(days=tbin / 2)
            xlabels.append(center.strftime("%Y-%m-%d"))

        time_axis_label = f"{tbin:g}-day time bin"

    x_positions = np.arange(len(groups))

    if len(groups) == 0:
        print("No time groups available.")
        return

    tick_step = max(
        1,
        int(np.ceil(len(groups) / max_xtick_labels)),
    )

    xticks_to_show = x_positions[::tick_step]
    xlabels_to_show = [
        xlabels[index]
        for index in xticks_to_show
    ]

    n_rows = len(params)
    n_cols = len(plot_modes)

    fig = plt.figure(
        figsize=(
            figsize_per_panel[0] * n_cols,
            figsize_per_panel[1] * n_rows,
        )
    )

    color_mapper = plt.get_cmap(cmap)
    plot_mode_index = {
        mode: index
        for index, mode in enumerate(plot_modes)
    }

    # ------------------------------------------------------------------
    # Plot every parameter
    # ------------------------------------------------------------------

    for row_index, param in enumerate(params):

        if param not in accepted.columns:
            print(f"Skipping missing parameter: {param}")
            continue

        values = pd.to_numeric(
            accepted[param],
            errors="coerce",
        ).dropna()

        if values.empty:
            print(f"Skipping parameter with no finite values: {param}")
            continue

        y_min = float(np.nanmin(values))
        y_max = float(np.nanmax(values))

        if np.isclose(y_min, y_max):
            pad = 0.5 if y_min == 0 else abs(y_min) * 0.05

            if pad == 0:
                pad = 0.5

            y_min -= pad
            y_max += pad

        y_edges = np.linspace(y_min, y_max, bins + 1)
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        # Initialize empty time bins as NaN rather than zero.
        # Zero would imply a measured distribution with no counts;
        # NaN correctly represents no observation.
        hist2d = np.full(
            (bins, len(groups)),
            np.nan,
            dtype=float,
        )

        med = np.full(len(groups), np.nan)
        p16 = np.full(len(groups), np.nan)
        p84 = np.full(len(groups), np.nan)

        for group_index, group in enumerate(groups):
            group_values = pd.to_numeric(
                accepted.loc[
                    accepted["time_group"] == group,
                    param,
                ],
                errors="coerce",
            ).dropna()

            if group_values.empty:
                continue

            group_array = group_values.to_numpy(dtype=float)

            med[group_index] = np.nanmedian(group_array)
            p16[group_index] = np.nanpercentile(group_array, 16)
            p84[group_index] = np.nanpercentile(group_array, 84)

            counts, _ = np.histogram(
                group_array,
                bins=y_edges,
            )

            hist2d[:, group_index] = (
                counts / len(group_array)
            ) * 100.0

        finite_hist = hist2d[np.isfinite(hist2d)]

        if finite_hist.size:
            z_max = float(np.nanmax(finite_hist))
        else:
            z_max = 1.0

        if not np.isfinite(z_max) or z_max <= 0:
            z_max = 1.0

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        if show_summary:
            subplot_index = (
                row_index * n_cols
                + plot_mode_index["summary"]
                + 1
            )

            ax = fig.add_subplot(
                n_rows,
                n_cols,
                subplot_index,
            )

            # NaN values automatically break the line across empty bins.
            ax.plot(
                x_positions,
                med,
                marker="o",
                label="Median",
            )

            ax.fill_between(
                x_positions,
                p16,
                p84,
                where=np.isfinite(p16) & np.isfinite(p84),
                alpha=0.25,
                interpolate=False,
                label="16–84% range",
            )

            ax.set_xlim(-0.5, len(groups) - 0.5)
            ax.set_ylim(y_min, y_max)

            ax.set_title(_pretty_param_label(param))
            ax.set_xlabel(time_axis_label)
            ax.set_ylabel(_pretty_param_label(param))

            ax.set_xticks(xticks_to_show)
            ax.set_xticklabels(
                xlabels_to_show,
                rotation=45,
                ha="right",
            )

            if row_index == 0:
                ax.legend()

        # --------------------------------------------------------------
        # Heatmap
        # --------------------------------------------------------------

        if show_heatmap:
            subplot_index = (
                row_index * n_cols
                + plot_mode_index["heatmap"]
                + 1
            )

            ax = fig.add_subplot(
                n_rows,
                n_cols,
                subplot_index,
            )

            # Mask NaNs so unobserved time bins are blank.
            masked_hist = np.ma.masked_invalid(hist2d)

            local_cmap = plt.get_cmap(cmap).copy()
            local_cmap.set_bad(color="white", alpha=0.0)

            im = ax.imshow(
                masked_hist,
                origin="lower",
                aspect="auto",
                cmap=local_cmap,
                extent=[
                    -0.5,
                    len(groups) - 0.5,
                    y_edges[0],
                    y_edges[-1],
                ],
                interpolation="nearest",
                vmin=0,
                vmax=z_max,
            )

            ax.set_xlim(-0.5, len(groups) - 0.5)
            ax.set_ylim(y_min, y_max)

            ax.set_xlabel(time_axis_label)
            ax.set_ylabel(
                _pretty_param_label(param),
                fontsize=22,
            )

            ax.set_xticks(xticks_to_show)
            ax.set_xticklabels(
                xlabels_to_show,
                rotation=45,
                ha="right",
            )

            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )
            cbar.set_label("% of accepted models")

        # --------------------------------------------------------------
        # 3D surface
        # --------------------------------------------------------------

        if show_3d_surface:
            subplot_index = (
                row_index * n_cols
                + plot_mode_index["3d_surface"]
                + 1
            )

            ax = fig.add_subplot(
                n_rows,
                n_cols,
                subplot_index,
                projection="3d",
            )

            X, Y = np.meshgrid(
                y_centers,
                x_positions,
            )

            Z = hist2d.T

            # Mask all cells belonging to empty time bins.
            Z_masked = np.ma.masked_invalid(Z)

            surf = ax.plot_surface(
                X,
                Y,
                Z_masked,
                cmap=cmap,
                linewidth=0,
                antialiased=True,
                alpha=0.85,
                vmin=0,
                vmax=z_max,
            )

            ax.view_init(
                elev=elev,
                azim=azim,
            )

            ax.set_xlim(y_min, y_max)
            ax.set_ylim(-0.5, len(groups) - 0.5)
            ax.set_zlim(0, z_max * 1.05)

            try:
                ax.set_box_aspect((2.5, 2.0, 1.2))
            except Exception:
                pass

            ax.set_title(
                f"{_pretty_param_label(param)} surface"
            )
            ax.set_xlabel(_pretty_param_label(param))
            ax.set_ylabel(time_axis_label)
            ax.set_zlabel("% density")

            ax.set_yticks(xticks_to_show)
            ax.set_yticklabels(
                xlabels_to_show,
                fontsize=8,
            )

            fig.colorbar(
                surf,
                ax=ax,
                shrink=0.65,
                pad=0.08,
                label="% density",
            )

        # --------------------------------------------------------------
        # 3D waterfall
        # --------------------------------------------------------------

        if show_3d_waterfall:
            subplot_index = (
                row_index * n_cols
                + plot_mode_index["3d_waterfall"]
                + 1
            )

            ax = fig.add_subplot(
                n_rows,
                n_cols,
                subplot_index,
                projection="3d",
            )

            for group_index, group in enumerate(groups):
                pct_counts = hist2d[:, group_index]

                # Skip empty bins entirely, leaving a real gap.
                if not np.any(np.isfinite(pct_counts)):
                    continue

                peak = np.nanmax(pct_counts)

                if np.isfinite(peak) and z_max > 0:
                    normalized_peak = np.clip(
                        peak / z_max,
                        0,
                        1,
                    )
                else:
                    normalized_peak = 0

                line_color = color_mapper(normalized_peak)

                ax.plot(
                    y_centers,
                    np.full_like(
                        y_centers,
                        group_index,
                        dtype=float,
                    ),
                    pct_counts,
                    linewidth=1.8,
                    color=line_color,
                    alpha=0.9,
                )

            ax.view_init(
                elev=elev,
                azim=azim,
            )

            ax.set_xlim(y_min, y_max)
            ax.set_ylim(-0.5, len(groups) - 0.5)
            ax.set_zlim(0, z_max * 1.05)

            try:
                ax.set_box_aspect((2.5, 2.0, 1.2))
            except Exception:
                pass

            ax.set_title(
                f"{_pretty_param_label(param)} waterfall"
            )
            ax.set_xlabel(_pretty_param_label(param))
            ax.set_ylabel(time_axis_label)
            ax.set_zlabel("% density")

            ax.set_yticks(xticks_to_show)
            ax.set_yticklabels(
                xlabels_to_show,
                fontsize=8,
            )

    title_suffix = (
        ""
        if tbin is None
        else rf", $\Delta t={tbin:g}$ days"
    )

    fig.suptitle(
        rf"Accepted model distributions over time: "
        rf"$F/F_{{\min}} \leq {accepted_fom}$"
        + title_suffix,
        fontsize=30,
        y=1.02,
    )

    plt.tight_layout()
    plt.show()