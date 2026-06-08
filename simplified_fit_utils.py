import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from lmfit.models import VoigtModel, ExponentialModel, SplineModel, LinearModel, GaussianModel, PseudoVoigtModel, Model
from scipy.special import wofz
from astropy.io import fits
import glob
from astropy.time import Time
import re
from copy import copy
from astropy.table import Table

def extract_datetime_from_filename(filename):
    """Extracts and parses the datetime from the filename."""
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', filename)
    if match:
        datetime_str = match.group(1)
        date, time = datetime_str.split('T')
        formatted_time = time.replace('-', ':', 2).replace('_', '.')
        return datetime.strptime(f"{date}T{formatted_time}", "%Y-%m-%dT%H:%M:%S")
    return None

def organize_files_by_date(file_list):
    """Sorts files by extracted datetime."""
    files_with_dates = []
    for file in file_list:
        extracted_date = extract_datetime_from_filename(file)
        if extracted_date:
            files_with_dates.append((extracted_date, file))

    # Sort by datetime
    files_with_dates.sort(key=lambda x: x[0])

    # Extract sorted filenames
    sorted_files = [file for _, file in files_with_dates]
    return sorted_files

def calculate_equivalent_width(x, y, fit, bkg_fit, centroid, width):
    # Define the region of interest (ROI) as +/- 4 * FWHM around the centroid
    roi_min = centroid - width
    roi_max = centroid + width

    # Mask the data within the ROI
    mask = (x >= roi_min) & (x <= roi_max)
    x_roi = x[mask]
    y_roi = y[mask]
    fit_roi = fit[mask] #+ bkg_fit[mask]
    bkg_fit_roi = bkg_fit[mask]

    # Continuum flux (use the background fit in this case)
    continuum = bkg_fit_roi

    # Compute the equivalent width
    ew = np.trapezoid(1 - (fit_roi / continuum), x_roi)

    return ew

def calculate_vr_ratio(x_halpha, y_halpha, fwhm, centroid_em1, centroid_em2):
    """
    Calculate the V/R ratio for a double-peaked emission profile.

    Parameters:
    - wave_fit: The best fit profile (array of y-values).
    - centroid_em1: Centroid of the first (blue) emission peak (float).
    - centroid_em2: Centroid of the second (red) emission peak (float).

    Returns:
    - vr_ratio: The calculated V/R ratio (float).
    """

    # Ensure centroids are correctly identified as blue and red peaks
    if centroid_em1 < centroid_em2:
        blue_centroid = centroid_em1
        red_centroid = centroid_em2
    else:
        blue_centroid = centroid_em2
        red_centroid = centroid_em1

    wv_per_ind = (x_halpha[-1] - x_halpha[0])/len(x_halpha)

    # Find the indices corresponding to the centroids
    blue_index = np.argmin(np.abs(x_halpha - blue_centroid))
    red_index = np.argmin(np.abs(x_halpha - red_centroid))
    blue_low = int(np.round(blue_index - 2))
    blue_high = int(np.round(blue_index + 2))
    red_low = int(np.round(red_index - 2))
    red_high = int(np.round(red_index + 2))

    print(f'Blue Calc Range: x = {blue_low*wv_per_ind + x_halpha[0]:.3f} to {blue_high*wv_per_ind + x_halpha[0]:.3f}')
    print(f'Red Calc Range: x = {red_low*wv_per_ind + x_halpha[0]:.3f} to {red_high*wv_per_ind + x_halpha[0]:.3f}')

    # Extract the intensities at the centroids
    # intensity_blue = np.median(y_halpha[blue_low:blue_high])
    # intensity_red = np.median(y_halpha[red_low:red_high])
    intensity_blue = np.max(y_halpha[blue_low:blue_high])
    intensity_red = np.max(y_halpha[red_low:red_high])


    # Calculate the V/R ratio
    vr_ratio = intensity_blue / intensity_red

    print(f"Blue Peak Intensity: {intensity_blue:.5f}")
    print(f"Red Peak Intensity: {intensity_red:.5f}")
    print(f"V/R Ratio: {vr_ratio:.5f}")

     # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_halpha, y_halpha, label="Best Fit", color="black")
    plt.axvline(blue_centroid, color="blue", linestyle="--", label=f"Blue Peak (λ = {blue_centroid:.2f} Å, I = {intensity_blue:.5f})")
    plt.axvline(red_centroid, color="red", linestyle="--", label=f"Red Peak (λ = {red_centroid:.2f} Å, I = {intensity_red:.5f})")
    plt.scatter([blue_centroid, red_centroid], [intensity_blue, intensity_red], color=["blue", "red"], zorder=5)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Intensity")
    plt.title("Double-Peaked Emission Profile and V/R Ratio")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.show()

    return vr_ratio

def voigt_fwhm(sigma, gamma):
    """Calculate the FWHM of a Voigt profile given sigma and gamma."""
    # Approximation for Voigt FWHM
    fwhm_gaussian = 2 * np.sqrt(2 * np.log(2)) * sigma
    fwhm_lorentzian = 2 * gamma
    fwhm_voigt = 0.5346 * fwhm_lorentzian + np.sqrt(0.2166 * fwhm_lorentzian**2 + fwhm_gaussian**2)
    return fwhm_voigt

def voigt_height(x, height, center, sigma, gamma):
    z = (x - center + 1j*gamma) / (sigma * np.sqrt(2))
    z0 = 1j*gamma / (sigma * np.sqrt(2))
    return height * np.real(wofz(z)) / np.real(wofz(z0))

def sigma_from_fwhm(fwhm, gamma):
    term = (fwhm - 1.0692*gamma)**2 - 0.8664*gamma**2
    return np.sqrt(term / (8*np.log(2)))

# def voigt_height(x, height, center, fwhm, gamma):
#     # sigma = (fwhm - 0.5346 * (2 * gamma)) / (2 * np.sqrt(2 * np.log(2)))
#     sigma = sigma_from_fwhm(fwhm, gamma)
#     z = (x - center + 1j*gamma) / (sigma * np.sqrt(2))
#     z0 = 1j*gamma / (sigma * np.sqrt(2))
#     return height * np.real(wofz(z)) / np.real(wofz(z0))

def fit_spectral_profiles(x, y, heights, centroids, y_variability, x_tolerance, min_gamma, max_gamma, min_sigma, max_sigma, chi2_threshold=0.0017, fixed_components = None): 

    fixed_components = [] if fixed_components is None else fixed_components
    
    if len(heights) != len(centroids):
        raise ValueError("Length of heights and centroids must be the same.")
    if len(heights) == 0 or len(centroids) == 0 or len(y_variability) == 0 or len(x_tolerance) == 0:
        raise ValueError("At least one component must be provided.")
    if len(y_variability) == 1:
        y_variability = [y_variability[0]] * len(heights)
    if len(x_tolerance) == 1:
        x_tolerance = [x_tolerance[0]] * len(heights)
    if len(min_gamma) == 1:
        min_gamma = [min_gamma[0]] * len(heights)
    if len(max_gamma) == 1:
        max_gamma = [max_gamma[0]] * len(heights)
    if len(min_sigma) == 1:
        min_sigma = [min_sigma[0]] * len(heights)
    if len(max_sigma) == 1:
        max_sigma = [max_sigma[0]] * len(heights)

    # Create all models first
    bkg = LinearModel(prefix=f'bkg_')
    mod = bkg
    
    # Build the combined model with all Voigt components
    for i in range(len(heights)):
        voigt = Model(voigt_height, prefix=f'v_{i}_')
        mod += voigt
    
    # Create parameters from the final combined model
    pars = mod.make_params()
    
    # Apply background guess to get reasonable starting values for the background
    pars.update(bkg.guess(y, x))
    
    # Set parameters for each Voigt component
    # for i in range(len(heights)):
    #     print(f'Initial Guess for Component {i}: Height = {heights[i]}, Centroid = {centroids[i]}')
    #     # pars[f'v_{i}_fwhm'].set(value=min_fwhm[i]*2, vary=True, min=min_fwhm[i])
    #     pars[f'v_{i}_gamma'].set(value=0.5, vary=True, min=min_gamma[i], max=max_gamma[i])
    #     pars[f'v_{i}_sigma'].set(value=1, vary=True, min=min_sigma[i], max=max_sigma[i])
    #     # pars[f'v_{i}_amplitude'].set(value=amplitudes[i], vary=True)
    #     pars[f'v_{i}_center'].set(value=centroids[i], vary=True, min=centroids[i]-x_tolerance[i], max=centroids[i]+x_tolerance[i])
    #     if heights[i] > 0:
    #         pars[f'v_{i}_height'].set(value = heights[i],min=heights[i]/y_variability[i], max=heights[i]*y_variability[i], vary=True)
    #     else:
    #         pars[f'v_{i}_height'].set(value=heights[i], min=heights[i]*y_variability[i], max=heights[i]/y_variability[i], vary=True)

    for i in range(len(heights)): # Set parameters accounting for fixed components (except centroid)
        is_fixed = i in fixed_components

        print(f'Initial Guess for Component {i}: Height = {heights[i]}, Centroid = {centroids[i]}')

        pars[f'v_{i}_gamma'].set(value=0.5, vary=not is_fixed, min=min_gamma[i], max=max_gamma[i],)
        pars[f'v_{i}_sigma'].set(value=1, vary=not is_fixed, min=min_sigma[i], max=max_sigma[i],)
        pars[f'v_{i}_center'].set( value=centroids[i], vary=True, min=centroids[i] - x_tolerance[i], max=centroids[i] + x_tolerance[i])

        if heights[i] > 0:
            pars[f'v_{i}_height'].set(value=heights[i], min=heights[i] / y_variability[i], max=heights[i] * y_variability[i], vary=not is_fixed)
        else:
            pars[f'v_{i}_height'].set(value=heights[i], min=heights[i] * y_variability[i], max=heights[i] / y_variability[i], vary=not is_fixed)

    print(f"Combined model parameters: {mod.param_names}")
    print(f"Initial parameter values: {pars.valuesdict()}")

    # Perform the fit
    out = mod.fit(y, pars, x=x)
    
    # Print the actual fitted parameters from the result
    print(f"Final parameter values: {out.params.valuesdict()}")
    chi2_red = out.redchi
    print(f'Reduced Chi-Squared Value: {chi2_red:.8f}')

    if chi2_red > chi2_threshold:
        print(f"Poor fit: chi-squared above threshold. Consider adjusting initial guesses or model complexity.")
        return None

    return out

def calc_amplitude(height, sigma, gamma):
    amplitude =  height * sigma * np.sqrt(2 * np.pi) / np.real(wofz(1j*gamma / (sigma * np.sqrt(2))))
    return amplitude

# def calc_fwhm(sigma, gamma):
#     fwhm = 0.5343 * (2 * gamma) + np.sqrt(0.2169 * (2 * gamma)**2 + (2 * np.sqrt(2 * np.log(2)) * sigma)**2)
#     return fwhm

def calculate_equivalent_width_abs(x, y, best_fit, bkg_fit, centroid, fwhm):
    """
    Calculate the equivalent width (EW) of a spectral feature.

    Parameters:
    - x (array): Wavelength array.
    - y (array): Observed flux array.
    - best_fit (array): Best-fit flux array from the Voigt model.
    - bkg_fit (array): Background continuum fit array.
    - centroid (float): Centroid of the spectral feature.
    - fwhm (float): Full width at half maximum of the feature.

    Returns:
    - float: The calculated equivalent width (EW).
    """
    # Define the region of interest (ROI) as +/- 4 * FWHM around the centroid
    roi_min = centroid - 4 * fwhm
    roi_max = centroid + 4 * fwhm

    # Mask the data within the ROI
    mask = (x >= roi_min) & (x <= roi_max)
    x_roi = x[mask]
    y_roi = y[mask]
    best_fit_roi = best_fit[mask]
    bkg_fit_roi = bkg_fit[mask]
    # abs_fit_roi = abs_fit[mask]

    # Continuum flux (use the background fit in this case)
    continuum = bkg_fit_roi

    # Compute the equivalent width
    ew = np.trapz(1 - (best_fit_roi / continuum), x_roi)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Observed Flux", alpha=0.5, color="blue")
    plt.plot(x, best_fit, label="Best Fit", color="green", linestyle="--")
    # plt.plot(x, bkg_fit, label="Background Continuum", color="orange", linestyle=":")

    # Highlight the region of interest (ROI)
    plt.fill_between(x_roi, y_roi, bkg_fit_roi, color="gray", alpha=0.3, label=f"Integrated Region (EW = {ew:.2f})")

    # Indicate the centroid
    plt.axvline(centroid, color="red", linestyle="--", label=f"Centroid (x = {centroid:.2f})")

    # Add labels and legend
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title("Equivalent Width Calculation")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    return ew

# ----------------------------
# BeSSSpectra: WCS-free
# ----------------------------
def BeSSSpectra(file):
    """
    Load a 1-D BeSS spectrum from FITS.

    Returns:
        wavelengths: np.ndarray (linear solution from header)
        spectrum: np.ndarray
        header: FITS header
    """
    try:
        f = fits.open(file)
        header = f[0].header
        spectrum = f[0].data

        # Linear wavelength solution from header
        crval1 = header.get('CRVAL1', 6563.0)  # default near Hα
        cdelt1 = header.get('CDELT1', 1.0)
        crpix1 = header.get('CRPIX1', 1.0)

        wavelengths = crval1 + (np.arange(len(spectrum)) + 1 - crpix1) * cdelt1

        return wavelengths, spectrum, header

    except Exception as e:
        print(f"⚠️ Failed to read {file}: {e}")
        return None, None, None

# ----------------------------
# sort_files: wavelength filter + DATE-OBS from header
# ----------------------------
def sort_files(file_paths, h_alpha=6563.0):
    """
    Collect FITS files covering Hα, store observation dates from header.

    Returns:
        valid_files: list of paths
        valid_times: list of astropy Time objects (from DATE-OBS)
    """
    files = glob.glob(file_paths)
    valid_files = []
    valid_times = []

    print(f"Found {len(files)} files in directory.")

    for file in files:
        try:
            wavelengths, spectrum, header = BeSSSpectra(file)
            min = np.min(wavelengths)
            max = np.max(wavelengths)
            peak = np.max(spectrum)
            if wavelengths is None or peak > 1000:
                continue

            # Wavelength coverage filter
            if min > h_alpha or max < h_alpha or np.abs(min - h_alpha) > 1500 or np.abs(max - h_alpha) > 1500:
                continue

            # Get observation date from header
            date_obs = header.get('DATE-OBS', None)
            if date_obs is None:
                print(f"⚠️ {file} has no DATE-OBS")
                continue

            try:
                date_obj = Time(date_obs, format='isot', scale='utc')
            except Exception:
                # Fallback for non-standard DATE-OBS
                date_obj = Time(datetime.strptime(date_obs[:10], "%Y-%m-%d"))

            valid_files.append(file)
            valid_times.append(date_obj)

        except Exception as e:
            print(f"⚠️ Skipping {file} due to error: {e}")
            continue

    # Sort by date
    if valid_files:
        sorted_pairs = sorted(zip(valid_times, valid_files))
        valid_times, valid_files = zip(*sorted_pairs)
    else:
        valid_times, valid_files = [], []

    print(f"{len(valid_files)} files passed Hα filter and have DATE-OBS.")

    return list(valid_files), list(valid_times)

# ----------------------------
# spec_grid: extract spectra around Hα
# ----------------------------
def spec_grid(target, radius, bess_files_sorted, h_alpha=6564.46):
    """
    Extract spectral segments around Hα from 1-D spectra.

    Args:
        target: placeholder, not used here
        radius: number of pixels on either side of target wavelength
        bess_files_sorted: list of FITS file paths
        h_alpha: target rest wavelength

    Returns:
        result: concatenated np.ndarray of segments
    """
    result = np.array([])

    for fname in bess_files_sorted:
        wavelengths, spectrum, header = BeSSSpectra(fname)
        if wavelengths is None:
            continue

        # Find closest pixel to Hα
        index = np.argmin(np.abs(wavelengths - h_alpha))
        if index - radius < 0 or index + radius > len(spectrum):
            continue

        segment = spectrum[index - radius:index + radius]
        result = np.append(result, segment)

    return result

def stack_daily_spectra(spectra, days_from_start, wavelength_corrections):
  unique_days = np.unique(days_from_start)
  stacked_wavelength_corrections = np.zeros(len(unique_days))
  stacked = copy(spectra[0:len(unique_days)])
  for j in range(len(unique_days)):
    d = unique_days[j]
    x = np.zeros(0)
    y = np.zeros(0)
    for i in range(len(spectra)):
      if days_from_start[i] == d:
        x = np.append(x, spectra[i]['Wavelength'])
        y = np.append(y, spectra[i]['Flux'])
        stacked_wavelength_corrections[j] = wavelength_corrections[i]
    tab = Table([x,y], names=('Wavelength', 'Flux'))
    stacked[j] = tab.group_by('Wavelength')
  return stacked, stacked_wavelength_corrections

def select_tellurics(folder, molecules, min_wavelength, max_wavelength, min_intensity = 0):
    tellurics = dict([])
    for molecule in molecules:
        telluric_files = glob.glob(f'{folder}*_{molecule}_*.txt')
        for file in telluric_files:
            data = np.loadtxt(file, skiprows=1)
            wavelengths = data[:, 0] * 10000  # Convert from microns to Angstroms
            intensities = data[:, 1]
            for i in range(len(wavelengths)):
                if intensities[i] >= min_intensity and min_wavelength <= wavelengths[i] <= max_wavelength:
                    tellurics[molecule + f'_{wavelengths[i]:.0f}'] = wavelengths[i]
    return tellurics