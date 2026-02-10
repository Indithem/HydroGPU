import cupy as cp
from utils import GeoTIFFHandler # Assuming this and load_tif_image are correctly defined elsewhere
from utils import load_tif_image # Assuming this and load_tif_image are correctly defined elsewhere
import numpy as np
import time

FLOW_DIR_TIF='../tifs/dem_experiments/doubleCheckResults/masalia_flow_dir_from_deplessCupy.tif'
DEM_TIF='../tifs/dem_experiments/deplessDemMasalia.tif'
OUTPUT_TIF='../tifs/dem_experiments/flow_acc_masalia_cupyFd.tif'

# Flow direction lookup table (host-side)
flow_offsets = {
    1: (-1, 1), 2: (-1, 0), 3: (-1, -1), # 1:NE, 2:N, 3:NW
    4: (0, -1), 5: (1, -1), 6: (1, 0),  # 4:W,  5:SW, 6:S
    7: (1, 1), 8: (0, 1)               # 7:SE, 8:E
}

# Create device-side lookup arrays
# These correspond to (dy, dx) or (row_offset, col_offset)
# For dir=1 (NE): dy=-1 (up), dx=1 (right)
# dirs_x maps to col_offset, dirs_y maps to row_offset
dirs_x = cp.array([flow_offsets[d][1] for d in range(1, 9)], dtype=cp.int8)
dirs_y = cp.array([flow_offsets[d][0] for d in range(1, 9)], dtype=cp.int8)

# Kernel 1: Compute in-degree
kernel_in_degree = cp.RawKernel(r'''
extern "C" __global__
void compute_in_degree(
    const unsigned char* flow_dir,
    int* in_deg, // Changed from uint8 in Python to int32, kernel signature is fine
    int width, int height,
    const signed char* dx,
    const signed char* dy
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    unsigned char dir = flow_dir[idx];
    // Values 1-8 are valid flow directions. Other values (e.g., 0, 255 for NoData) are skipped.
    if (dir < 1 || dir > 8) return;

    // Target cell coordinates (where current cell flows to)
    int tx = x + dx[dir - 1]; // dx for direction `dir`
    int ty = y + dy[dir - 1]; // dy for direction `dir`

    if (tx >= 0 && tx < width && ty >= 0 && ty < height) {
        int tidx = ty * width + tx;
        atomicAdd(&in_deg[tidx], 1); // Increment in-degree of the cell being flowed into
    }
}
''', 'compute_in_degree')

# Kernel 2: Initialize active frontier
kernel_init_active = cp.RawKernel(r'''
extern "C" __global__
void init_active(const int* in_deg, unsigned char* active, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    // Cells with an in-degree of 0 are the initial sources of flow
    active[i] = (in_deg[i] == 0) ? 1 : 0;
}
''', 'init_active')

# Kernel 3: Flow propagation
kernel_propagate = cp.RawKernel(r'''
extern "C" __global__
void flow_propagate(
    const unsigned char* flow_dir,
    int* acc,
    int* in_deg,
    const unsigned char* active,
    unsigned char* next_active,
    int width, int height,
    const signed char* dx,
    const signed char* dy
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    if (active[idx] == 0) return; // Process only currently active cells

    // Accumulation for the current cell (initially 1, grows as flow is received)
    // This value will be added to the downstream cell.
    int current_acc = acc[idx];

    unsigned char dir = flow_dir[idx];
    if (dir < 1 || dir > 8) return; // Skip if no valid flow direction

    int tx = x + dx[dir - 1];
    int ty = y + dy[dir - 1];

    if (tx >= 0 && tx < width && ty >= 0 && ty < height) {
        int tidx = ty * width + tx;

        // Add current cell's accumulated flow to the downstream target cell
        atomicAdd(&acc[tidx], current_acc);

        // Decrement in-degree of the target cell, as flow from current cell is now accounted for
        // atomicSub returns the OLD value (before subtraction)
        int prev_deg = atomicSub(&in_deg[tidx], 1);

        // If the target cell's in-degree becomes 0 (it was 1 before subtraction),
        // it becomes active in the next iteration.
        if (prev_deg == 1) {
            next_active[tidx] = 1;
        }
    }
}
''', 'flow_propagate')


def flow_accum_topo(flow_dir_gpu): # Renamed input for clarity
    H, W = flow_dir_gpu.shape
    N = H * W

    # Ensure flow_dir is uint8 as expected by kernels
    flow_dir_gpu = flow_dir_gpu.astype(cp.uint8)

    # ***** FIX 1: Change in_deg data type to int32 *****
    # uint8 is too small and can lead to overflow for in-degree counts.
    # CUDA kernels expect int*, which matches cp.int32.
    in_deg = cp.zeros((H, W), dtype=cp.int32)

    # Initialize accumulation: each cell contributes 1 to itself initially.
    acc = cp.ones((H, W), dtype=cp.int32) # Use cp.int64 for extremely large catchments if overflow is a concern

    active = cp.zeros((H, W), dtype=cp.uint8)
    next_active = cp.zeros((H, W), dtype=cp.uint8) # Buffer for next iteration's active cells

    block = (16, 16)
    grid = ((W + block[0] - 1) // block[0], (H + block[1] - 1) // block[1]) # Corrected grid calculation

    # Step 1: Compute in-degree for all cells
    # For each cell, find where it flows TO, and increment the IN-DEGREE of that TARGET cell.
    kernel_in_degree(grid, block, (flow_dir_gpu, in_deg, W, H, dirs_x, dirs_y))

    # Step 2: Initialize active frontier (cells with in-degree == 0)
    # These are the "source" cells (e.g., ridge tops, or cells whose upstream contributors are NoData).
    # Kernel operates on flattened arrays.
    kernel_init_active(((N + 255) // 256,), (256,), (in_deg.ravel(), active.ravel(), N))

    # Step 3: Propagate flow iteratively until no more cells are active
    iteration_count = 0
    while cp.any(active):
        iteration_count += 1
        next_active.fill(0) # Clear the next active set for the current iteration

        kernel_propagate(grid, block, (
            flow_dir_gpu, acc, in_deg, active, next_active, W, H, dirs_x, dirs_y
        ))
        active, next_active = next_active, active # Swap active sets for the next iteration
        # print(f"Iteration {iteration_count}, Active cells: {cp.sum(active).item()}") # Optional: for debugging

    # print(f"Total iterations: {iteration_count}") # Optional
    return acc


# --- Main script part ---
# FLOW_DIR_TIF = '../tifs/masalia/qgis/qgis_masalia_fdir.tif'
# DEM_TIF = '../tifs/masalia/masalia_dem/dem.tif' # Renamed for clarity
# OUTPUT_TIF = '../tifs/masalia/experiment_tifs/masalia_f_acc.tif' # Renamed for clarity
print("experimenting with dep less dem")
# FLOW_DIR_TIF = '../tifs/lower_ganga_basin/dem/lower_ganga_fd.tif'
# DEM_TIF = '../tifs/lower_ganga_basin/dem/lower_ganga_dem.tif'
# OUTPUT_TIF = '../tifs/lgb/flow_acc1.tif'
# FLOW_DIR_TIF = '../tifs/brahmani/dems/brahmani_depressionless_dem_dir.tif'
# DEM_TIF = '../tifs/brahmani/dems/brahmani_depressionless_dem.tif' # Renamed for clarity
# OUTPUT_TIF = '../tifs/brahmani/outputs/brahmani_flow_acc.tif'
# It's crucial to know the NoData value of your DEM.
# Assuming GeoTIFFHandler can provide this, or load_tif_image handles it.
# For this example, we'll assume you can get the NoData value.
# If load_tif_image sets NoData pixels to a specific value (e.g., 0 or NaN), use that.

tif_handler = GeoTIFFHandler(DEM_TIF) # Initialize with DEM to get its properties

print("Loading flow direction raster...",FLOW_DIR_TIF)
flow_direction_cpu = load_tif_image(FLOW_DIR_TIF)
# **POTENTIAL ISSUE 2: Flow Direction Convention**
# Verify that the values in qgis_masalia_fdir.tif match the 'flow_offsets' definition.
# E.g., if your TIF has '1' for East, but 'flow_offsets' defines '1' as NE, results will be wrong.
# You might need to remap 'flow_direction_cpu' values here if conventions differ.
flow_direction_gpu = cp.asarray(flow_direction_cpu)

print("Calculating flow accumulation...")
start_time = time.time()
flow_accumulation_gpu = flow_accum_topo(flow_direction_gpu)
end_time = time.time()
print(f"Flow accumulation calculation took {end_time - start_time:.2f} seconds.")

print("Loading DEM for masking...")
dem_cpu = load_tif_image(DEM_TIF)
dem_gpu = cp.asarray(dem_cpu)

# **POTENTIAL ISSUE 3: DEM NoData Value for Masking**
# Determine the actual NoData value used in your DEM_TIF.
# Replace 'dem_nodata_value_placeholder' with the correct NoData value from your DEM.
# If GeoTIFFHandler provides it: dem_nodata_value = tif_handler.nodata (or similar)
# If load_tif_image already converts NoData to 0, then '0' is correct here.
dem_nodata_value_placeholder = 0 # Replace with actual NoData value if not 0
                                 # For example, if NoData is -9999:
                                 # dem_nodata_value_placeholder = -9999

print(f"Masking flow accumulation with DEM (NoData value used for comparison: {dem_nodata_value_placeholder})...")
# Set flow accumulation to 0 (or a specific NoData value for the output) where DEM is NoData
# It's often better to set to an output-specific NoData value rather than 0 if 0 is a valid accumulation.
output_nodata_val = 0 # Or perhaps tif_handler.nodata if you want to preserve it
flow_accumulation_masked_gpu = cp.where(dem_gpu != dem_nodata_value_placeholder, flow_accumulation_gpu, output_nodata_val)

print(f"Saving output TIFF to {OUTPUT_TIF}...")
tif_handler.save_tiff(flow_accumulation_masked_gpu.get(), OUTPUT_TIF) # .get() transfers data from GPU to CPU
print("Processing complete.")
