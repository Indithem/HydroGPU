from tqdm import tqdm
from utils import make_logger, GeoTIFFHandler, load_tif_image
import cupy as cp
import numpy as np
from natsort import natsorted
import os
import warnings
from collections import defaultdict
from make_rainfall_vs_runoff_plots import make_all_plots, per_watershed_avg_runoff, update_series

# SOIL_TIFF = '../tifs/lgb/INPUTS/soil_clipped.tif'
# LULC_TIFF = '../tifs/lgb/INPUTS/lulc_clipped.tif'
# SLOPE_TIFF = '../tifs/lgb/INPUTS/dem_clipped.tif'
REFERENCE_TIFF = '../tifs/masalia/sr1.tif'
#MWS_TIFF = '../tifs/lgb/mwshed.tif'
# DESTINATION_INDEX_TIFF = '../tifs/lgb/3h_destination_index.tif' # Jump Matrix
DESTINATION_INDEX_TIFF = '../tifs/masalia/experiment_tifs/3h_destination_index_masalia.tif' # Jump Matrix
# DESTINATION_INDEX_TIFF = '../tifs/lower_ganga_basin/itr_3H_jump.tif' # Jump Matrix
# SR_DIRECTORY = '../tifs/masalia/'
# RAINFALL_FOLDER = '../test/rain_subset'
# RAINFALL_FOLDER = '../tifs/lgb/Trash/3hr_quantisation/merged_rainfall_3H_2023-07-01_2023-10-01/'
RAINFALL_FOLDER = '../tifs/masalia/masalia_3h_rain'
# OUTPUT_FILE = '../tifs/lgb/runoff_raster.tiff'
OUTPUT_FOLDER = '../tifs/masalia/runoffs_no_sr_negative_bounds_check/'

SOIL_TIFF = '../tifs/masalia/soil_lulc/soil.tif'
LULC_TIFF = '../tifs/masalia/soil_lulc/lulc.tif'
SLOPE_TIFF = '../tifs/masalia/masalia_dem/dem.tif'

# wrong dimensions of rainfall tiff
# HEIGHT, WIDTH = 22431, 33352
# reference tiff had:
# HEIGHT, WIDTH = 21761, 32356

logger = make_logger("logs/masalia_bounds_check.log", gpu_mem_usage=True)
# logger
handler = GeoTIFFHandler(REFERENCE_TIFF, logger)

def check_numerical_stability(arr: cp.ndarray, name: str) -> bool:
    """
    Checks for NaN, Inf, or -Inf values in a CuPy array.
    Returns True if stable, False if unstable.
    """
    is_stable = True

    # Check for NaN
    if cp.any(cp.isnan(arr)):
        logger.warn(f"    ❌ VALIDATION FAILED: '{name}' contains NaN values.")
        is_stable = False

    # Check for Inf or -Inf
    if cp.any(cp.isinf(arr)):
        logger.warn(f"    ❌ VALIDATION FAILED: '{name}' contains Inf or -Inf (divergence).")
        is_stable = False

    return is_stable

def check_physical_range(arr: cp.ndarray, name: str,
                           min_val: float = None, max_val: float = None) -> bool:
    """
    Checks if array values are within a specified physical range, ignoring NaN/Inf.
    Returns True if in range, False if out of range.
    """
    is_in_range = True

    # Create a mask for valid (non-NaN, non-Inf) numbers
    valid_mask = ~cp.isnan(arr) & ~cp.isinf(arr)

    # If there are no valid numbers, we can't check the range.
    if not cp.any(valid_mask):
        return True # Handled by check_numerical_stability

    if min_val is not None:
        # Check for values below min_val
        if cp.any(arr[valid_mask] < min_val):
            logger.warn(f"    ❌ VALIDATION FAILED: '{name}' has values < {min_val}.")
            is_in_range = False

    if max_val is not None:
        # Check for values above max_val
        if cp.any(arr[valid_mask] > max_val):
            logger.warn(f"    ❌ VALIDATION FAILED: '{name}' has values > {max_val}.")
            is_in_range = False

    return is_in_range

def static_all_methods(cls):
    for name, attr in cls.__dict__.items():
        if callable(attr):
            setattr(cls, name, staticmethod(attr))
    return cls

@static_all_methods
class LegacyCodes:

    def compute_cn2(soil: cp.ndarray, lulc: cp.ndarray) -> cp.ndarray:
        """
        Compute CN2 values from soil and lulc matrices using CuPy.

        Parameters:
            soil (cp.ndarray): Soil type matrix (values 0–4).
            lulc (cp.ndarray): LULC class matrix (values 0–7).

        Returns:
            cp.ndarray: CN2 values.
        """
        # Define lookup table [soil_type][lulc_class]
        LUT = cp.array([
            [ 0,  0,  0,  0,  0,  0,  0,  0],  # soil 0 → CN2 = 0 (as fallback)
            [ 0, 30, 39,  0, 64, 39, 82, 49],  # soil 1
            [ 0, 55, 61,  0, 75, 61, 88, 69],  # soil 2
            [ 0, 70, 74,  0, 82, 74, 91, 79],  # soil 3
            [ 0, 77, 80,  0, 85, 80, 93, 84],  # soil 4
        ], dtype=cp.int32)

        # Ensure valid bounds before indexing
        soil = cp.clip(soil.astype(cp.int32), 0, 4)
        lulc = cp.clip(lulc.astype(cp.int32), 0, 7)

        # Apply lookup
        CN2 = LUT[soil, lulc]

        del soil, lulc, LUT

        return CN2

    @staticmethod   # this should happen for other funcs also, idk why isn't happening
    def compute_cn1(CN2: cp.ndarray) -> cp.ndarray:
        """
        Compute CN1 from CN2 using the formula: CN1 = -75 * CN2 / (CN2 - 175)

        Parameters:
            CN2 (cp.ndarray): Curve Number 2 matrix (usually int or float)

        Returns:
            cp.ndarray: CN1 values as float32
        """
        CN2 = CN2.astype(cp.float32)
        denom = CN2 - 175
        # Prevent division by zero
        denom = cp.where(denom == 0, cp.finfo(cp.float32).eps, denom)
        CN1 = (-75 * CN2) / denom

        del denom

        return CN1

    def compute_cn3(CN2: cp.ndarray) -> cp.ndarray:
        """
        Compute CN3 from CN2 using the formula:
            CN3 = CN2 * (e ** (0.00673 * (100 - CN2)))

        Parameters:
            CN2 (cp.ndarray): CuPy array of Curve Number 2 values.

        Returns:
            cp.ndarray: CuPy array of CN3 values.
        """
        CN2 = CN2.astype(cp.float32)
        exponent = 0.00673 * (100.0 - CN2)
        CN3 = CN2 * cp.exp(exponent)

        del exponent, CN2

        return CN3

    def compute_part1(CN3: cp.ndarray, CN2: cp.ndarray) -> cp.ndarray:
        CN3 = CN3.astype(cp.float32)
        CN2 = CN2.astype(cp.float32)
        p1 = (CN3 - CN2) / 3.0

        del CN3, CN2

        return p1

    def compute_part2(slope: cp.ndarray) -> cp.ndarray:
        slope = slope.astype(cp.float32)
        p2 = 1.0 - 2.0 * cp.exp(-13.86 * slope)

        del slope

        return p2

    def compute_CN2a(part1: cp.ndarray, part2: cp.ndarray, CN2: cp.ndarray) -> cp.ndarray:
        # Ensure type consistency
        part1 = part1.astype(cp.float32)
        part2 = part2.astype(cp.float32)
        CN2 = CN2.astype(cp.float32)

        CN2a = part1 * part2 + CN2

        del part1, part2, CN2

        return CN2a

    def compute_CN1a(CN2a: cp.ndarray) -> cp.ndarray:
        CN2a = CN2a.astype(cp.float32)
        CN1a = 4.2 * CN2a / (10 - 0.058 * CN2a)

        del CN2a

        return CN1a

    def compute_CN3a(CN2a: cp.ndarray) -> cp.ndarray:
        CN2a = CN2a.astype(cp.float32)
        CN3a = 23 * CN2a / (10 + 0.13 * CN2a)

        del CN2a

        return CN3a

    def compute_sr(CN: cp.ndarray) -> cp.ndarray:
        CN = CN.astype(cp.float32)

        # Mask where CN is invalid (e.g., 0 or very small)
        mask = CN <= 10
        CN = cp.where(mask, 100.0, CN)  # default value or np.nan

        # Clip CN to range 30–100
        CN = cp.clip(CN, 30.0, 100.0)

        sr = (25400.0 / CN) - 254.0

        # # Optional: mask output back where CN was invalid
        # sr = cp.where(mask, -254.0, sr)

        del CN, mask

        return sr




    def compute_M(sr, p):
        """
        Compute M2 using CuPy, ensuring that if either sr or p is NaN, the result is NaN.
        """
        nan_mask = cp.isnan(sr) | cp.isnan(p)
        sqrt_term = cp.sqrt(cp.maximum(sr**2 + 4 * p * sr, 0.0))
        M2 = 0.5 * (-sr + sqrt_term)
        M2[nan_mask] = cp.nan  # Preserve NaN values
        return M2

    def compute_M_alt(sr, p):
        # Ensure float type, potentially float64 for precision
        sr = sr.astype(cp.float64)
        p = p.astype(cp.float64)

        # Preserve input NaNs
        nan_mask_input = cp.isnan(sr) | cp.isnan(p)

        # Calculate term inside sqrt
        term = sr**2 + 4 * p * sr

        # Allow sqrt to produce NaN for negative inputs (and suppress warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore sqrt domain error warning
            sqrt_term = cp.sqrt(term) # This will be NaN where term < 0

        M_result = 0.5 * (-sr + sqrt_term)

        # Ensure input NaNs propagate, and sqrt NaNs are kept
        M_result[nan_mask_input | cp.isnan(sqrt_term)] = cp.nan

        del sr, p, nan_mask_input, term, sqrt_term

        return M_result

    def sum_tif_images(images, start, end):
        """
        Sum multiple TIFF images using CuPy on the GPU.
        """
        return cp.sum(images[start:end], axis=0)

    def calculate_p_and_p5(file_paths):
        """
        Load images, convert to CuPy, and compute total precipitation sums.
        """
        images = [cp.asarray(load_tif_image(fp), dtype=cp.float32) for fp in file_paths]
        total_sum = cp.sum(images, axis=0)
        mid_sum = cp.sum(images[-2:], axis=0) if len(images) >= 2 else total_sum
        return total_sum, mid_sum

    def calculate_runoff(P, P5, M1, M2, M3, sr1, sr2, sr3):
        """
        Compute runoff using CuPy.
        """
        nan_mask = cp.isnan(P) | cp.isnan(P5) | cp.isnan(M1) | cp.isnan(M2) | cp.isnan(M3) | cp.isnan(sr1) | cp.isnan(sr2) | cp.isnan(sr3)
        runoff = cp.zeros_like(sr1)
        mask1 = (~nan_mask) & (P >= 0.2 * sr1) & (P5 >= 0) & (P5 <= 35)
        mask2 = (~nan_mask) & (P >= 0.2 * sr2) & (P5 > 35) & (P5 <= 52.5)
        mask3 = (~nan_mask) & (P >= 0.2 * sr3) & (P5 > 52.5)

        runoff[mask1] = ((P[mask1] - 0.2 * sr1[mask1]) * (P[mask1] - 0.2 * sr1[mask1] + M1[mask1])) / (P[mask1] + 0.2 * sr1[mask1] + sr1[mask1] + M1[mask1])
        runoff[mask2] = ((P[mask2] - 0.2 * sr2[mask2]) * (P[mask2] - 0.2 * sr2[mask2] + M2[mask2])) / (P[mask2] + 0.2 * sr2[mask2] + sr2[mask2] + M2[mask2])
        runoff[mask3] = ((P[mask3] - 0.2 * sr3[mask3]) * (P[mask3] - 0.2 * sr3[mask3] + M3[mask3])) / (P[mask3] + 0.2 * sr3[mask3] + sr3[mask3] + M3[mask3])

        runoff[nan_mask] = cp.nan  # Restore NaNs
        return runoff

    @staticmethod
    def calculate_runoff_cupy(P, P5, m1, m2, m3, sr1, sr2, sr3):
        """
        Calculates runoff using a CuPy implementation mirroring a GEE expression.

        Follows the logic:
        Q = f(P, P5, sr, m) based on AMC I, II, III where P5 determines AMC.
        Uses derived m1, m2, m3 and potential max retention sr1, sr2, sr3.
        Ensures runoff >= 0 and handles NaN inputs (NaN in any input -> NaN output).

        Args:
            P (cp.ndarray): Precipitation matrix.
            P5 (cp.ndarray): 5-day antecedent precipitation matrix.
            m1 (cp.ndarray): Derived moisture parameter for AMC I.
            m2 (cp.ndarray): Derived moisture parameter for AMC II.
            m3 (cp.ndarray): Derived moisture parameter for AMC III.
            sr1 (cp.ndarray): Potential maximum retention for AMC I (S derived from CN1).
            sr2 (cp.ndarray): Potential maximum retention for AMC II (S derived from CN2).
            sr3 (cp.ndarray): Potential maximum retention for AMC III (S derived from CN3).

        Returns:
            cp.ndarray: Calculated runoff matrix, with NaN where any input was NaN.
        """
        # Optional: Check if inputs are indeed CuPy arrays (if function might receive others)
        # P = cp.asarray(P) # etc. for all inputs

        # --- 0. Input Validation (Optional but Recommended) ---
        if not P.shape == P5.shape == m1.shape == m2.shape == m3.shape == \
                sr1.shape == sr2.shape == sr3.shape:
            raise ValueError("All input CuPy arrays must have the same shape.")

        # --- 1. Handle NaN Inputs: Create combined mask ---
        # If any input pixel is NaN, the output for that pixel will be NaN.

        # --- 2. Calculate Intermediate Terms (Initial Abstraction) ---
        Ia1 = 0.2 * sr1
        Ia2 = 0.2 * sr2
        Ia3 = 0.2 * sr3

        # --- 3. Calculate Potential Runoff Values (Q1, Q2, Q3) ---
        # Suppress potential division-by-zero or invalid value warnings as we handle them
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # # Calculate Denominators for the runoff formula
            den = P + Ia1 + sr1 + m1
            num = (P - Ia1) * (P - Ia1 + m1)
            Q1 = cp.where(den != 0, num / den, 0.0).astype(cp.float64)

            den = P + Ia2 + sr2 + m2
            num = (P - Ia2) * (P - Ia2 + m2)
            Q2 = cp.where(den != 0, num / den, 0.0).astype(cp.float64)

            den = P + Ia3 + sr3 + m3
            num = (P - Ia3) * (P - Ia3 + m3)
            Q3 = cp.where(den != 0, num / den, 0.0).astype(cp.float64)

            del den, num

        # --- 4. Define Conditions using Boolean Masks ---
        # Basic precipitation conditions: P must be >= Initial Abstraction (Ia)
        condP = (P >= Ia1)
        condAMC = (P5 >= 0) & (P5 <= 35)
        condQ_pos = (Q1 >= 0)
        cond1_full = condP & condAMC & condQ_pos

        condP = (P >= Ia2)
        condAMC = (P5 >= 0) & (P5 > 35)   # Corresponds to the second check in GEE ternary
        condQ_pos = (Q2 >= 0)
        cond2_full = condP & condAMC & condQ_pos

        condP = (P >= Ia3)
        condAMC = (P5 >= 0) & (P5 > 52.5)  # Corresponds to the third check in GEE ternary
        condQ_pos = (Q3 >= 0)
        cond3_full = condP & condAMC & condQ_pos

        del condP, condAMC, condQ_pos
        del Ia1, Ia2, Ia3

        # Antecedent Moisture Conditions based on P5 thresholds from GEE expression
        # Note: P5>=0 check is included in the GEE expression, so we replicate it.

        # Runoff non-negativity conditions (Q must be >= 0) from GEE expression

        # Combine all conditions for each case
        # These directly represent the full condition before the '?' in the GEE expression

        # --- 5. Apply Conditions using Nested cp.where (Mirrors GEE Ternary Logic) ---
        # This structure directly implements: cond1 ? Q1 : (cond2 ? Q2 : (cond3 ? Q3 : 0))
        final_runoff = cp.where(cond1_full, Q1,             # If Cond1 is true, use Q1
                        cp.where(cond2_full, Q2,         # Else, if Cond2 is true, use Q2
                            cp.where(cond3_full, Q3,     # Else, if Cond3 is true, use Q3
                                0.0)))                   # Else (all conditions false), use 0.0

        del cond1_full, cond2_full, cond3_full

        # --- 6. Apply NaN Mask ---
        # Ensure any pixel that had NaN in any input results in NaN output
        # Note: cp.where might already propagate NaNs correctly in many cases,
        # but applying the mask explicitly guarantees it.

        nan_mask = cp.isnan(P) | cp.isnan(P5) | cp.isnan(m1) | cp.isnan(m2) | cp.isnan(m3) | \
                cp.isnan(sr1) | cp.isnan(sr2) | cp.isnan(sr3)

        final_runoff[nan_mask] = cp.nan

        return final_runoff


    def runoff_total_volume(runoff):
        nan_mask = cp.isnan(runoff)
        runoff[~nan_mask] = runoff[~nan_mask] * 900
        return runoff

    # Define the kernel
    transfer_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void transfer_flow(const int* F, const float* V, float* V_out, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        int dest = F[i];  // Get destination index
        if (dest >= 0 && dest < size) {
            atomicAdd(&V_out[dest], V[i]);  // Transfer value to destination
        }
    }
    ''', 'transfer_flow')

    # Function to execute the kernel
    def transfer_flow(F, V):
        F_cp = cp.asarray(F, dtype=cp.int32)
        V_cp = cp.asarray(V, dtype=cp.float32)
        V_out = cp.zeros_like(V_cp)  # Initialize output matrix

        size = F_cp.size
        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block


        # Launch the kernel
        LegacyCodes.transfer_kernel((blocks_per_grid,), (threads_per_block,), (F_cp, V_cp, V_out, size))

        return V_out

def compute_sr_and_CNs(soil, lulc, slope):
    """
        Returns sr1, sr2 and sr3 cupy arrays.
    """

    check_numerical_stability(soil, "soil")
    check_numerical_stability(lulc, "lulc")
    check_numerical_stability(slope, "slope")
    check_physical_range(soil, "soil", min_val=0, max_val=4)
    check_physical_range(lulc, "lulc", min_val=0, max_val=7)
    check_physical_range(slope, "slope", min_val=0.0)

    logger.info("Calculating SR ...")

    CN2 = LegacyCodes.compute_cn2(soil, lulc)
    CN1 = LegacyCodes.compute_cn1(CN2)
    CN3 = LegacyCodes.compute_cn3(CN2)

    check_numerical_stability(CN2, "CN2")
    check_numerical_stability(CN1, "CN1")
    check_numerical_stability(CN3, "CN3")

    p1 = LegacyCodes.compute_part1(CN3, CN2)
    p2 = LegacyCodes.compute_part2(slope)

    CN2a = LegacyCodes.compute_CN2a(p1, p2, CN2)
    CN1a = LegacyCodes.compute_CN1a(CN2a)
    CN3a = LegacyCodes.compute_CN3a(CN2a)

    check_numerical_stability(CN2a, "CN2a")
    check_numerical_stability(CN1a, "CN1a")
    check_numerical_stability(CN3a, "CN3a")
    assert check_physical_range(CN2a, "CN2a", min_val=0.0, max_val=100.0)
    assert check_physical_range(CN1a, "CN1a", min_val=0.0, max_val=100.0)
    assert check_physical_range(CN3a, "CN3a", min_val=0.0, max_val=100.0)

    sr1 = LegacyCodes.compute_sr(CN1a)
    sr2 = LegacyCodes.compute_sr(CN2a)
    sr3 = LegacyCodes.compute_sr(CN3a)

    check_numerical_stability(sr1, "sr1")
    check_numerical_stability(sr2, "sr2")
    check_numerical_stability(sr3, "sr3")
    assert check_physical_range(sr1, "sr1", min_val=0.0)
    assert check_physical_range(sr2, "sr2", min_val=0.0)
    assert check_physical_range(sr3, "sr3", min_val=0.0)

    logger.info("Just completed calculating SR")

    return sr1, sr2, sr3

def test_raster(img, index, file):
    img_cp = cp.asarray(img)
    pos_inf_check = cp.sum(cp.isposinf(img_cp))
    neg_inf_check = cp.sum(cp.isneginf(img_cp))
    nan_check = cp.sum(cp.isnan(img_cp))

    logger.info(f"Positive infinity check sum for rainfall image: {pos_inf_check}")
    logger.info(f"Negative infinity check sum for rainfall image: {neg_inf_check}")
    # logger.info(f"Post-transfer positive infinity check sum: {cp.sum(positive_inf_check)}, first 5 values with pos inf: {cp.asarray(previous_Runoff)[positive_inf_check][:min(5, cp.sum(positive_inf_check))].get()}")
    # logger.info(f"Post-transfer negative infinity check sum: {cp.sum(negative_inf_check)}, first 5 values with neg inf: {cp.asarray(previous_Runoff)[negative_inf_check][:min(5, cp.sum(negative_inf_check))].get()}")
    # logger.info(f"Post-transfer nan check sum: {cp.sum(nan_check)}, first 5 values with nan: {cp.asarray(previous_Runoff)[nan_check][:min(5, cp.sum(nan_check))].get()}")

    logger.info(f"NaN check sum for rainfall image: {nan_check}")
    if pos_inf_check > 0 or neg_inf_check > 0:
        logger.warning(f"Infinity values found in the rainfall data at index {index},file {file}. They will be replaced with NaN.")
    elif nan_check > 0:
        logger.warning(f"NaN values found in the rainfall data at index {index},file {file}. They will be preserved as NaN.")


def runoff_simulation_algo(rainfall_folder:str):
    """
        rainfall folder should have files in "natural" sorted order. (see GNU sort's `sort -V`)
        Yield: list of rainfall and optionally (None for first 4) runoff rasters
    """


    files = natsorted(os.listdir(rainfall_folder))
    files = [file for file in files if file.endswith('.tif')]
    # files = files[:10]    # for testing

    images = []
    P_sum = None
    P5_sum = None
    previous_Runoff = None
    runoff_rasters = []
    destination_index = handler.load_with_padding(DESTINATION_INDEX_TIFF)
    # sr1 = cp.asarray(handler.load_with_padding(SR_DIRECTORY+'sr1.tif'))
    # sr2 = cp.asarray(handler.load_with_padding(SR_DIRECTORY+'sr2.tif'))
    # sr3 = cp.asarray(handler.load_with_padding(SR_DIRECTORY+'sr3.tif'))

    sr1, sr2, sr3 = compute_sr_and_CNs(
        cp.asarray(handler.load_with_padding(SOIL_TIFF)),
        cp.asarray(handler.load_with_padding(LULC_TIFF)),
        cp.asarray(handler.load_with_padding(SLOPE_TIFF))
    )

    handler.save_tiff(cp.asnumpy(sr1), os.path.join(OUTPUT_FOLDER, 'sr1.tif'))
    handler.save_tiff(cp.asnumpy(sr2), os.path.join(OUTPUT_FOLDER, 'sr2.tif'))
    handler.save_tiff(cp.asnumpy(sr3), os.path.join(OUTPUT_FOLDER, 'sr3.tif'))

    logger.info("Loaded sr's")

    for index, file in enumerate(tqdm(files)):
        logger.info(f"Processing file {index}: {file}")
        img_path = os.path.join(rainfall_folder, file)
        img = handler.load_with_padding(img_path)

        test_raster(img, index, file)
        np.nan_to_num(img, copy=False)


        # img = cp.asarray(img, dtype=cp.float32)
        images.append(img)

        if index == 4:
            P_sum = cp.sum(cp.stack([cp.asarray(i) for i in images[2:5]]), axis=0)
            P5_sum = cp.sum(cp.stack([cp.asarray(i) for i in images[:5]]), axis=0)
            logger.info(f"4. Initial sum creation")
        elif index >= 5:
            new_img = cp.asarray(images[-1])
            assert check_physical_range(new_img, "new_img", min_val=0.0)
            old_img = cp.asarray(images[0])
            assert check_physical_range(old_img, "old_img", min_val=0.0)
            P5_sum = P5_sum - old_img + new_img
            old_img = cp.asarray(images[-3])
            assert check_physical_range(old_img, "old_img (for P_sum)", min_val=0.0)
            P_sum = P_sum - old_img + new_img
            logger.info(f"4. Updated sums")
            del old_img, new_img
            old_img = images.pop(0)
            del old_img
            logger.info(f"5. Pop and delete oldest image")

        if index >= 4:
            if previous_Runoff is not None:
                P_sum += previous_Runoff
                P5_sum += previous_Runoff
                logger.info("6. Add previous runoff")

            check_numerical_stability(P_sum, "P_sum")
            check_numerical_stability(P5_sum, "P5_sum")
            assert check_physical_range(P_sum, "P_sum", min_val=0.0)
            assert check_physical_range(P5_sum, "P5_sum", min_val=0.0)

            m1_cp = LegacyCodes.compute_M(sr1, P5_sum)
            m2_cp = LegacyCodes.compute_M(sr2, P5_sum)
            m3_cp = LegacyCodes.compute_M(sr3, P5_sum)
            logger.info("7. Compute M1,M2,M3")

            # cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            # with cp.cuda.memory_hooks.DebugPrintHook():
            R = LegacyCodes.calculate_runoff_cupy(P_sum, P5_sum, m1_cp, m2_cp, m3_cp, sr1, sr2, sr3)
            del m1_cp, m2_cp, m3_cp
            logger.info("8. Calculate runoff")

            previous_Runoff = LegacyCodes.transfer_flow(destination_index, R)

            test_raster(previous_Runoff, index, file)

            # positive_inf_check = cp.isposinf(cp.asarray(previous_Runoff))
            # negative_inf_check = cp.isneginf(cp.asarray(previous_Runoff))
            # nan_check= cp.isnan(cp.asarray(previous_Runoff))
            # if cp.sum(positive_inf_check) > 0 or cp.sum(negative_inf_check) > 0:
            #     logger.warning(f"Infinity values found in the transferred runoff data at index {index} and rainfall file {file}. They will be replaced with NaN.")
            # elif cp.sum(nan_check) > 0:
            #     logger.warning(f"NaN values found in the transferred runoff data at index {index} and rainfall file {file}. They will be preserved as NaN.")

            logger.info("9. Transfer flow")

            handler.save_tiff(cp.asnumpy(R), os.path.join(OUTPUT_FOLDER, f'runoff_simulation_{index}.tif'))
            logger.info("10. write runoff simulation result")
            # runoff_rasters.append(R.get())

        #    yield (img, R)
        #else:
        #    yield (img, None)

    logger.info("Done runoff sim")
    return runoff_rasters

def rainfall_folder_check(rainfall_folder):
    files = natsorted(os.listdir(rainfall_folder))
    files = [file for file in files if file.endswith('.tif')]
    for index, file in enumerate(tqdm(files)):
        logger.info(f"Checking file {index}: {file}")
        img_path = os.path.join(rainfall_folder, file)
        img = handler.load_with_padding(img_path)
        img = cp.asarray(img, dtype=cp.float32)
        pos_inf_check = cp.isposinf(cp.asarray(img))
        neg_inf_check = cp.isneginf(cp.asarray(img))
        print(f"Positive infinity check sum: {cp.sum(pos_inf_check)}")
        print(f"Negative infinity check sum: {cp.sum(neg_inf_check)}")
        print(f"nan count is {cp.sum(cp.isnan(cp.asarray(img)))}")
        if cp.sum(pos_inf_check) > 0 or cp.sum(neg_inf_check) > 0:
            logger.warning(f"Infinity values found in the rainfall data at index {index},file {file}. They will be replaced with NaN.")

def check_static_layer():
    # destination_index = handler.load_with_padding(DESTINATION_INDEX_TIFF)
    # pos_inf_check = cp.isposinf(cp.asarray(destination_index))
    # neg_inf_check = cp.isneginf(cp.asarray(destination_index))
    # nan_check = cp.isnan(cp.asarray(destination_index))
    # logger.info(f"Positive infinity check sum for destination index: {cp.sum(pos_inf_check)}")
    # logger.info(f"Negative infinity check sum for destination index: {cp.sum(neg_inf_check)}")
    # logger.info(f"nan count for destination index is {cp.sum(nan_check)}")
    # if cp.sum(pos_inf_check) > 0 or cp.sum(neg_inf_check) > 0 or cp.sum(nan_check) > 0:
    #     logger.warning(f"Infinity values found in the destination index data. They will be replaced with NaN.")

    # sr_layers = {
    #     'sr1': cp.asarray(handler.load_with_padding(os.path.join(SR_DIRECTORY, 'sr1.tif'))),
    #     'sr2': cp.asarray(handler.load_with_padding(os.path.join(SR_DIRECTORY, 'sr2.tif'))),
    #     'sr3': cp.asarray(handler.load_with_padding(os.path.join(SR_DIRECTORY, 'sr3.tif'))),
    # }

    deps = {
        'soil': cp.asarray(handler.load_with_padding('../tifs/lgb/INPUTS/soil_clipped.tif')),
        'lulc': cp.asarray(handler.load_with_padding('../tifs/lgb/INPUTS/lulc_clipped.tif')),
        'slope': cp.asarray(handler.load_with_padding('../tifs/lgb/INPUTS/dem_clipped.tif')),
    }

    for name, layer in deps.items() :
        pos_inf_check = cp.isposinf(layer)
        neg_inf_check = cp.isneginf(layer)
        nan_check = cp.isnan(layer)

        logger.info(f"Positive infinity check sum for {name}: {cp.sum(pos_inf_check)}")
        logger.info(f"Negative infinity check sum for {name}: {cp.sum(neg_inf_check)}")
        logger.info(f"nan count for {name} is {cp.sum(nan_check)}")

        if cp.any(nan_check):
            logger.info(f"NaNs found in {name}. Saving NaN mask and locations.")

            # Save NaN mask raster
            nan_mask_filename = os.path.join(SR_DIRECTORY, f'{name}_nan.tif')
            nan_mask_numpy = nan_check.get().astype(np.uint8)
            handler.save_tiff(nan_mask_numpy, nan_mask_filename)
            logger.info(f"Saved NaN mask for {name} to {nan_mask_filename}")

            # Save NaN pixel locations to a text file
            nan_locations_filename = os.path.join(SR_DIRECTORY, f'{name}_nan_locations.txt')
            nan_coords = cp.argwhere(nan_check).get()
            with open(nan_locations_filename, 'w') as f:
                f.write(f"NaN locations for {name} (row, col):\n")
                np.savetxt(f, nan_coords, fmt='%d', delimiter=', ')
            logger.info(f"Saved NaN locations for {name} to {nan_locations_filename}")

        if cp.sum(pos_inf_check) > 0 or cp.sum(neg_inf_check) > 0:
            logger.warning(f"Infinity values found in the {name} data.")

if __name__ == "__main__":
    #mws_arr = handler.load_with_padding(MWS_TIFF)

    # creaye OUTPUT FOLDER if not exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    runoff_simulation_algo(RAINFALL_FOLDER)
    # rainfall_folder_check(RAINFALL_FOLDER)
    # check_static_layer()

    # rainfalltime_series = defaultdict(list)
    # runofftime_series = defaultdict(list)

    #for rainfall_raster, runoff_raster in runoff_simulation_algo(RAINFALL_FOLDER, sr1, sr2, sr3):
    #    # update_series(runofftime_series, rainfalltime_series, runoff_raster, rainfall_raster, mws_arr, logger)
    #    pass

    # make_all_plots(rainfalltime_series, runofftime_series, output_folder=OUTPUT_FOLDER+'plots/')

    # handler.save_multiband_tiff(OUTPUT_FILE, runoff_rasters)
