# import tensorflow as tf
import rasterio
import os

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def load_tif_image(file_path):
    """Load a .tif image efficiently and return a NumPy array (float32)."""
    with rasterio.open(file_path) as src:
        image = src.read(1)  # Read only the first band
        print(f"Loaded Raster - Shape: {image.shape}, Dtype: {image.dtype}")
        return image  # Kept as NumPy array for easier slicing

# def pad_image(image, pad_size, pad_value=1000.0):
#     """Efficient padding using NumPy (before converting to TensorFlow)."""
#     return np.pad(image, pad_size, mode='constant', constant_values=pad_value)

# # Define kernels (already flipped) with shape: (3, 3, 1, 8)
# kernels_flipped = tf.constant([
#     [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
#     [[-0.876, 0, 0], [0, 0.876, 0], [0, 0, 0]],
#     [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
#     [[0, 0, -0.876], [0, 0.876, 0], [0, 0, 0]],
#     [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
#     [[0, 0, -0.876], [0, 0.876, 0], [0, 0, 0]],
#     [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
#     [[-0.876, 0, 0], [0, 0.876, 0], [0, 0, 0]],
# ], dtype=tf.float32)
# kernels_flipped = tf.reshape(kernels_flipped, (3, 3, 1, 8))

# @tf.function(jit_compile=True)  # Optimized execution on GPU
# def apply_convolution(image):
#     """Apply 2D convolution and return the max kernel index."""
#     convolutions = tf.nn.conv2d(image, kernels_flipped, strides=1, padding="VALID")
#     return tf.argmax(convolutions, axis=-1)

# def flow_direction(image_path, output_path='./outputs/fd_lower_ganga_basin.png', tile_size=2048, overlap=2):
#     """
#     Process a large image in smaller tiles to prevent OOM errors.
#     For each tile, if a pixel is NaN in the original data it will remain NaN
#     in the final flow direction (neighbor/kernel index) output.
#     """
#     image = load_tif_image(image_path)
#     height, width = image.shape
    
#     # Create output array as float32 so we can store NaN values.
#     output = np.full((height, width), np.nan, dtype=np.float32)

#     s_time = time.time()
#     for y in range(0, height, tile_size - overlap):
#         for x in range(0, width, tile_size - overlap):
#             # Define tile boundaries (handle edge cases)
#             y_end = min(y + tile_size, height)
#             x_end = min(x + tile_size, width)

#             # Extract tile from the original image.
#             tile = image[y:y_end, x:x_end]
#             # Record positions that are NaN so we can preserve them.
#             tile_nan_mask = np.isnan(tile)

#             # Pad the tile (only the inner part has real data; borders get pad_value)
#             tile_padded = pad_image(tile, pad_size=1)  # Pad by 1 pixel
#             # Expand dims: shape becomes (1, height+2, width+2, 1)
#             tile_padded = np.expand_dims(tile_padded, axis=(0, -1))
#             # Convert to Tensor
#             tile_tensor = tf.convert_to_tensor(tile_padded, dtype=tf.float32)

#             # Run convolution; result has shape (1, tile_height, tile_width, 1)
#             best_kernel_indices = apply_convolution(tile_tensor).numpy().squeeze()
#             # Cast to float32 so we can assign NaN values (best_kernel_indices is originally int32)
#             best_kernel_indices = best_kernel_indices.astype(np.float32)
            
#             # Here, we mimic the original cropping:
#             # Remove the first row and column from the convolution output so that overlapping edges
#             # are handled consistently.
#             result_tile = best_kernel_indices[1:, 1:]
#             # Crop the original nan mask in the same way.
#             tile_nan_mask_cropped = tile_nan_mask[1:, 1:]
#             # Set positions that were originally NaN to NaN in the result.
#             result_tile[tile_nan_mask_cropped] = np.nan

#             # Store result in the correct region of the output.
#             # Note: Since we cropped one row and one column, assign to [y:y_end-1, x:x_end-1]
#             output[y:y_end-1, x:x_end-1] = result_tile

#     e_time = time.time()
#     print("flow direction time :", e_time - s_time)

#     # Optionally, save the final result using create_plot (if desired)
#     # create_plot(output, output_path=output_path)
#     # print(f"Processing complete. Output saved to: {output_path}")
    
#     return output


# @tf.function(jit_compile=True)
# def compute_tile_steepest_neighbor_index(elev_tile):
#     """
#     Computes the steepest neighbor index for each cell in a small elevation tile.
#     The steepness is defined as (center elevation - neighbor elevation) divided by 
#     the distance between cells. The eight neighbors are considered in row-major order:
#         0: Top-Left,  1: Top,    2: Top-Right,
#         3: Left,                  4: Right,
#         5: Bottom-Left, 6: Bottom, 7: Bottom-Right
#     Distances: sqrt(2) (~1.414) for diagonals, 1.0 for cardinal directions.
    
#     If no neighbor is lower (i.e. maximum slope <= 0), the output is set to 0.
    
#     Args:
#         elev_tile (tf.Tensor): 2D tensor of shape (H, W) with dtype tf.float32.
        
#     Returns:
#         tf.Tensor: 2D tensor of shape (H, W) with dtype tf.int32, where each value is 
#                    the 1-indexed steepest neighbor (1 to 8) or 0 if no descending neighbor exists.
#     """
#     # Ensure input is float32.
#     elev_tile = tf.convert_to_tensor(elev_tile, dtype=tf.float32)
    
#     H = tf.shape(elev_tile)[0]
#     W = tf.shape(elev_tile)[1]
    
#     # Define 8-neighborhood offsets in row-major order.
#     offsets = tf.constant([
#         [-1, -1], [-1, 0], [-1, 1],   # Top-left, Top, Top-right
#         [ 0, -1],          [ 0, 1],   # Left,           Right
#         [ 1, -1], [ 1, 0], [ 1, 1]     # Bottom-left, Bottom, Bottom-right
#     ], dtype=tf.int32)
    
#     # Define distance factors.
#     distances = tf.constant([
#         1.414, 1.0, 1.414,
#         1.0,        1.0,
#         1.414, 1.0, 1.414
#     ], dtype=tf.float32)
    
#     # Create a grid of indices for the center cells.
#     row_idx, col_idx = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
#     base_indices = tf.stack([row_idx, col_idx], axis=-1)  # Shape: (H, W, 2)
    
#     # Expand base indices to shape (H, W, 8, 2) and add the offsets.
#     neighbor_indices = tf.expand_dims(base_indices, axis=2) + offsets  # (H, W, 8, 2)
    
#     # Clip indices so that they lie within the tile boundaries.
#     neighbor_indices = tf.clip_by_value(neighbor_indices, [0, 0], [H - 1, W - 1])
    
#     # Gather neighbor elevation values.
#     neighbor_elev = tf.gather_nd(elev_tile, neighbor_indices)  # Shape: (H, W, 8)
    
#     # Compute the slope for each neighbor.
#     center_elev = tf.expand_dims(elev_tile, axis=-1)  # Shape: (H, W, 1)
#     slopes = (center_elev - neighbor_elev) / distances  # Shape: (H, W, 8)
    
#     # For each cell, find the neighbor with the maximum slope.
#     max_slopes = tf.reduce_max(slopes, axis=-1)
#     best_neighbor = tf.argmax(slopes, axis=-1, output_type=tf.int32)
    
#     # If the maximum slope is not positive, set index to 0 (meaning no descending neighbor).
#     best_neighbor = tf.where(max_slopes > 0, best_neighbor + 1, tf.zeros_like(best_neighbor))
    
#     return best_neighbor

# def compute_steepest_neighbor_index_tiled(elevation_matrix, tile_size=512, overlap=1):
#     """
#     Computes the steepest neighbor index for each cell in a large elevation matrix by processing
#     it in tiles. Each cell in the output contains a value in {0, 1, ..., 8}, where 0 indicates
#     no neighbor with a descending slope, and 1-8 indicate the (1-indexed) neighbor in row-major order.
    
#     Args:
#         elevation_matrix (np.ndarray or tf.Tensor): A 2D array representing elevation values.
#         tile_size (int): The size of the tile to process (without padding).
    
#     Returns:
#         np.ndarray: A 2D array of type int32 with the same shape as elevation_matrix containing the steepest neighbor index.
#     """
#     # Ensure the elevation matrix is a NumPy array.
#     if not isinstance(elevation_matrix, np.ndarray):
#         elevation_matrix = elevation_matrix.numpy()
    
#     H, W = elevation_matrix.shape
#     # Initialize output array.
#     output = np.zeros((H, W), dtype=np.int32)
    
#     # Process in tiles.
#     for i in range(0, H, tile_size - overlap):
#         for j in range(0, W, tile_size-overlap):
#             # Determine tile boundaries.
#             # Add a 1-cell padding on each side if available for correct neighbor computation.
#             top = i if i == 0 else i - 1
#             bottom = min(i + tile_size, H) if (i + tile_size) == H else min(i + tile_size, H) + 1
#             left = j if j == 0 else j - 1
#             right = min(j + tile_size, W) if (j + tile_size) == W else min(j + tile_size, W) + 1
            
#             # Extract the tile with padding.
#             tile = elevation_matrix[top:bottom, left:right]
#             # Compute steepest neighbor indices on the tile.
#             tile_result = compute_tile_steepest_neighbor_index(tile).numpy()
            
#             # Determine the cropping indices to remove the extra padding.
#             crop_top = 0 if i == 0 else 1
#             crop_left = 0 if j == 0 else 1
#             crop_bottom = tile_result.shape[0] if (i + tile_size) >= H else tile_result.shape[0] - 1
#             crop_right = tile_result.shape[1] if (j + tile_size) >= W else tile_result.shape[1] - 1
#             tile_cropped = tile_result[crop_top:crop_bottom, crop_left:crop_right]
            
#             # Place the processed tile into the output array.
#             out_i_end = i + tile_cropped.shape[0]
#             out_j_end = j + tile_cropped.shape[1]
#             output[i:out_i_end, j:out_j_end] = tile_cropped
            
#     return output

# @tf.function
# def compute_flow_direction(D):
#     """
#     Compute flow direction for each cell based on the smallest neighbor.
    
#     Args:
#         D (tf.Tensor): Elevation matrix of shape (H, W)
        
#     Returns:
#         F (tf.Tensor): Flow direction matrix of shape (H, W) with values {1-8, 0}
#     """
#     # Get shape
#     H, W = tf.shape(D)[0], tf.shape(D)[1]

#     # Create shifted matrices to represent 8-neighbor elevations
#     D_padded = tf.pad(D, [[1, 1], [1, 1]], mode="CONSTANT", constant_values=tf.constant(1000, dtype=tf.float32))

#     neighbors = [
#         D_padded[:-2, 1:-1],  # Top (1)
#         D_padded[:-2, 2:],     # Top-right (2)
#         D_padded[1:-1, 2:],    # Right (3)
#         D_padded[2:, 2:],      # Bottom-right (4)
#         D_padded[2:, 1:-1],    # Bottom (5)
#         D_padded[2:, :-2],     # Bottom-left (6)
#         D_padded[1:-1, :-2],   # Left (7)
#         D_padded[:-2, :-2],    # Top-left (8)
#     ]

#     # Stack along a new dimension for easy comparison
#     neighbors_stack = tf.stack(neighbors, axis=-1)  # Shape: (H, W, 8)

#     # Find the smallest neighbor and its corresponding direction
#     min_values = tf.reduce_min(neighbors_stack, axis=-1)
#     min_indices = tf.cast(tf.argmin(neighbors_stack, axis=-1), tf.int32) + 1  # Adding 1 to match 1-8 direction indexing

#     # Create zeros tensor with same type as min_indices
#     zeros = tf.zeros_like(D, dtype=tf.int32)

#     # Apply condition: If smallest neighbor is less than current cell, assign direction else 0
#     F = tf.where(min_values < D, min_indices, zeros)

#     return F
# flow_dir_img=flow_direction('../tifs/brahmani/dems/brahmani_depressionless_dem.tif','../tifs/brahmani/dems/brahmani_flow_direction.tif')