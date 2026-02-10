import cupy as cp
import numpy as np
from flow_direction import load_tif_image
from make_tif import GeoTIFFHandler
import time



# --- Vectorized Stream Order Calculation ---
# D8 direction offsets (dr, dc)
d8_offsets = {
    1: (-1, 1),   # top-right
    2: (-1, 0),   # top-center
    3: (-1, -1),  # top-left
    4: (0, -1),   # center-left
    5: (1, -1),   # bottom-left
    6: (1, 0),    # bottom-center
    7: (1, 1),    # bottom-right
    8: (0, 1),    # center-right
}

# Reverse D8: direction that a neighbor must have to flow INTO current cell
reverse_d8_flow_into = {
    (-1, 1): 5,   # top-right -> bottom-left
    (-1, 0): 6,   # top-center -> bottom-center
    (-1, -1): 7,  # top-left -> bottom-right
    (0, -1): 8,   # left -> right
    (1, -1): 1,   # bottom-left -> top-right
    (1, 0): 2,    # bottom-center -> top-center
    (1, 1): 3,    # bottom-right -> top-left
    (0, 1): 4     # right -> left
}

def compute_stream_order(flow_dir, flow_accum, threshold):
    """

    Compute Strahler stream order using BFS-style propagation with in-degree tracking.
    """
    d8_offsets = {
        1: (-1, 1), 2: (-1, 0), 3: (-1, -1), 4: (0, -1),
        5: (1, -1), 6: (1, 0), 7: (1, 1), 8: (0, 1)
    }

    reverse_d8 = {
        (-1, 1): 5, (-1, 0): 6, (-1, -1): 7, (0, -1): 8,
        (1, -1): 1, (1, 0): 2, (1, 1): 3, (0, 1): 4
    }

    shape = flow_dir.shape


    stream_mask = (flow_accum >= threshold).astype(cp.int8)
    flow_dir_masked = flow_dir * stream_mask

    # Step 1: compute in-degree for each cell
    in_degree = cp.zeros(shape, dtype=cp.int8)
    for (dr, dc), code in reverse_d8.items():
        shifted_dir = cp.roll(cp.roll(flow_dir_masked, -dr, axis=0), -dc, axis=1)
        shifted_mask = cp.roll(cp.roll(stream_mask, -dr, axis=0), -dc, axis=1)
        in_degree += ((shifted_dir == code) & (shifted_mask > 0)).astype(cp.int8)

    # Step 2: initialize order and active source cells
    stream_order = cp.zeros(shape, dtype=cp.int32)
    reached = cp.zeros(shape, dtype=cp.int8)
    max_upstream = cp.zeros(shape, dtype=cp.int32)
    max_count = cp.zeros(shape, dtype=cp.int8)

    active = (in_degree == 0) & (flow_dir_masked > 0)
    stream_order = cp.where(active, 1, 0)

    # Step 3: iterative BFS propagation
    while active.any():
        # For each of the 8 D8 directions
        for (code, (dr, dc)) in d8_offsets.items():
            # Active cells that flow in this direction
            is_flowing = (flow_dir_masked == code) & active

            # Stream order from active cells
            flowing_order = cp.where(is_flowing, stream_order, 0)
            flowing_count = is_flowing.astype(cp.int8)

            # Propagate order and count downstream
            target_order = cp.roll(cp.roll(flowing_order, dr, axis=0), dc, axis=1)
            target_count = cp.roll(cp.roll(flowing_count, dr, axis=0), dc, axis=1)

            # Compute whether target_order is a new max
            is_new_max = target_order > max_upstream
            is_equal_max = target_order == max_upstream

            # Update max_upstream
            max_upstream = cp.maximum(max_upstream, target_order)

            # Reset or increment max_count
            max_count = cp.where(is_new_max, 1, cp.where(is_equal_max, max_count + 1, max_count))

            # Track how many upstreams have reached
            reached += target_count

        # Identify next active cells
        ready = (reached == in_degree) & (stream_mask > 0) & (stream_order == 0)
        # print(f"iteration ready : {cp.sum(ready)} max_count : {cp.unique(max_count)}")
        is_confluence = ready & (max_count >= 2)
        next_order = cp.where(is_confluence, max_upstream + 1, max_upstream)
        stream_order = cp.where(ready, next_order, stream_order)

        # Prepare for next iteration
        active = ready
        print(cp.sum(active))

    return stream_order

# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    # FLOW_DIR_TIF = '../tifs/masalia/masalia_dem/dem_dir.tif'
    # FLOW_ACC_TIF = '../tifs/masalia/qgis/masalia_flow_acc.tif'
    # DEM = '../tifs/masalia/masalia_dem/dem.tif'
    # OUTPUT_STREAM_ORDER_TIF = '../tifs/masalia/experiment_tifs/mws_masalia_debug.tif'
    FLOW_DIR_TIF='../tifs/dem_experiments/masalia_fd_without_clipping.tif'
    FLOW_ACC_TIF='../tifs/dem_experiments/flow_acc_masalia_cupyFd.tif'
    DEM='../tifs/dem_experiments/dem_depless_masalia_clipped.tif'
    OUTPUT_STREAM_ORDER_TIF='../tifs/dem_experiments/stream_order_masalia_strahler.tif'
    print("new strahler")
    # Define the threshold for stream initiation (number of upstream cells)
    STREAM_THRESHOLD = 1000 # Example threshold - adjust as needed

    print("Loading data...")
    try:
        handler = GeoTIFFHandler(FLOW_DIR_TIF) # Initialize handler to get profile
        flow_dir_np = load_tif_image(FLOW_DIR_TIF)
        flow_acc_np = load_tif_image(FLOW_ACC_TIF)
        dem = load_tif_image(DEM)

        # --- Transfer Data to GPU ---
        print("Transferring data to CuPy arrays on GPU...")
        flow_direction_gpu = cp.asarray(flow_dir_np, dtype=cp.int32)
        flow_accumulation_gpu = cp.asarray(flow_acc_np, dtype=cp.float32) # Accumulation often float

        print(f"Input shapes: FlowDir={flow_direction_gpu.shape}, FlowAcc={flow_accumulation_gpu.shape}")
        if flow_direction_gpu.shape != flow_accumulation_gpu.shape:
             raise ValueError("Flow direction and flow accumulation grids must have the same shape.")

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
        print("Please ensure the TIF files exist at the specified paths.")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading or transfer: {e}")
        exit()

    # --- Calculate Stream Order ---
    print(f"Starting stream order calculation with threshold: {STREAM_THRESHOLD}")
    start_time = time.time()
    stream_order_result_gpu = compute_stream_order(
        flow_direction_gpu,
        flow_accumulation_gpu,
        STREAM_THRESHOLD
    )
    cp.cuda.runtime.deviceSynchronize() # Wait for GPU computation to finish
    end_time = time.time()
    print(f"Stream order calculation finished in {end_time - start_time:.4f} seconds.")

    # --- Process and Save Results ---
    print("Transferring result back to CPU...")

    stream_order_result_np = cp.asnumpy(stream_order_result_gpu).astype(np.int32)
    stream_order_result_np = np.where(dem != 0, stream_order_result_np, 0)

    print(f"Maximum stream order found: {np.max(stream_order_result_np)}")
    print(f"Number of stream cells (order > 0): {np.sum(stream_order_result_np > 0)}")

    print(f"Saving stream order TIF to {OUTPUT_STREAM_ORDER_TIF}...")
    try:
        handler.save_tiff(stream_order_result_np, OUTPUT_STREAM_ORDER_TIF)
        print("Stream Order Processing Completed with CuPy (Vectorized)!")
    except Exception as e:
        print(f"Error saving output TIF: {e}")





    # FLOW_DIR_TIF = '../tifs/lower_ganga_basin/dem/lower_ganga_fd.tif'
    # FLOW_ACC_TIF = '../tifs/lower_ganga_basin/dem/full_itr_floacc.tif'
    # DEM = '../tifs/lower_ganga_basin/dem/lower_ganga_dem.tif'
    # OUTPUT_STREAM_ORDER_TIF = '../tifs/lower_ganga_basin/stream_order/lower_ganga_stream_order_cupy.tif'






# def find_confluence_pixels(flow_dir, target):
#     """
#     Find cells on the target grid that have more than one incoming neighbor
#     whose flow direction points to the current cell and are also active in `target`.

#     Args:
#         flow_dir (cp.ndarray): 2D array with D8 flow directions.
#         target (cp.ndarray): Binary or integer mask indicating active upstream cells.

#     Returns:
#         cp.ndarray: A mask where confluence cells (with >1 valid upstreams) hold their incoming count.
#     """
#     nrows, ncols = flow_dir.shape
#     incoming_count = cp.zeros_like(flow_dir, dtype=cp.int8)

#     for (dr, dc), required_dir in reverse_d8_flow_into.items():
#         # Shift both flow_dir and target to align upstream neighbors to current cell
#         shifted_dir = cp.roll(cp.roll(flow_dir, -dr, axis=0), -dc, axis=1)
#         shifted_target = cp.roll(cp.roll(target, -dr, axis=0), -dc, axis=1)

#         # Count only if neighbor points to current cell AND is active in target
#         is_valid_upstream = (shifted_dir == required_dir) & (shifted_target > 0)
#         incoming_count += is_valid_upstream.astype(cp.int8)

#     # Mark only cells inside the target that have >1 valid upstream neighbors
#     confluence_mask = cp.where((incoming_count > 1) & (target > 0), incoming_count, 0)

#     return confluence_mask.astype(cp.int8)


# FLOW_DIR_TIF = '../tifs/masalia/masalia_dem/dem_dir.tif'
# FLOW_ACC_TIF = '../tifs/masalia/qgis/masalia_flow_acc.tif'
# DEM = '../tifs/masalia/masalia_dem/dem.tif'
# OUTPUT_STREAM_ORDER_TIF = '../tifs/masalia/experiment_tifs/mws_masalia_debug.tif'
