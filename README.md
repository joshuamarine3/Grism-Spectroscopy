- **Spectroscopic data processing**
  - Be_grism_proc_auto_JRM_new.ipynb
    - This is the main processing notebook for the spectroscopic data. It takes a 2D spectrum
    image, extracts a 1D spectrum, applies wavelength calibration, applies flux/gain 
    calibration. It is also capable of deriving these solutions when needed.
  - grism_utils_v2.py
    - This contains the class structure and various functionalities for processing spectra
    including fitting the spectral trace, extracting the 1D spectrum, deriving wavelength
    + gain calibration from standard stars, and applying wavelength + gain calibration to 
    science spectra
  - STELIB, ESO, CALSPEC
    - These folders contain reference spectra for spectrophotometric standard stars used in
    the calibration derivation procedure.
  - .pkl files
    - These contain historic wavelength and gain solutions already derived such that they do
    not need to be rederived each time. The ZWO files are for the previous backend + camera 
    while the QHY files are for the current backend + camera (as of 2026/05/06).
  - Grism_Calib_Testing_{camera}_{filter}.ipynb
    - These notebooks are for deriving the wavelength and gain calibration in case the procedure
    needs to be updated and calibration needs to be rederived for historic data. There are 
    four files for each camera (ZWO + QHY) and each filter (lrg + hrg).
  - Wave_Calib_{source name}_{filter}.ipynb
    - These notebooks are for inspecting individual calibrator sources to determine optimal
    features to use as points in the wavelength solution derivation and optimal regions to 
    mask for the gain curve solution derivation.
  - Grism_Focus_Optimizer.ipynb
    - This notebook contains a procedure for determining the optimal focus position for both
    the lrg and hrg. This requires a dataset of lrg and hrg images with varying focal positions
    and broadband images taken after an autofocus routine to determine the focus offsets of the
    lrg and hrg relative to the broadband images.

- **Spectroscopic Data Analysis**
  - Be_Spectral_Analysis_Master_Local.ipynb
    - This notebook is able to fit to various forms that spectral lines can take including
    absorption, double absorption, single-peaked emission, and double-peaked emission. It 
    tracks the amplitudes, centroids, fwhms, equivalent widths, and V/R (if applicable) over
    time. It can also fit a spectroscopic keplerian rv curve for centroids of binary systems.
  - fit_utils.py
    - This contains various functions for fitting spectroscopic data
