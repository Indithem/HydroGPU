from . import GenericAlgorithm
import cupy as cp

class FlowDirection(GenericAlgorithm):
    def load_inputs(self):
        self.dem = self.tif_handler.load_with_padding(self.cfg.DEMFILE_PATH)

    def save_outputs(self):
        fd_np = self.fd.get()
        self.tif_handler.save_tiff(fd_np, self.cfg.FLOWDIRECTION_PATH)

    def main(self):

        # D8 offsets (row, col)
        flow_offsets = {
            1: (-1, 1),  # NE
            2: (-1, 0),  # N
            3: (-1, -1), # NW
            4: (0, -1),  # W
            5: (1, -1),  # SW
            6: (1, 0),   # S
            7: (1, 1),   # SE
            8: (0, 1)    # E
        }

        # Corresponding distance (D) for each direction
        direction_distance = {
            1: cp.sqrt(2),  # Diagonal
            2: 1.0,
            3: cp.sqrt(2),
            4: 1.0,
            5: cp.sqrt(2),
            6: 1.0,
            7: cp.sqrt(2),
            8: 1.0
        }

        def d8_flow_direction(dem: cp.ndarray) -> cp.ndarray:
            """
            Compute D8 flow direction raster for a depressionless DEM using CuPy.

            Parameters
            ----------
            dem : cp.ndarray
                2D CuPy array of elevation values (float32 or float64)

            Returns
            -------
            flow_dir : cp.ndarray
                2D CuPy array of flow direction codes (1–8)
            """

            nrows, ncols = dem.shape
            flow_dir = cp.zeros_like(dem, dtype=cp.uint8)

            # Pad DEM to handle edges
            dem_padded = cp.pad(dem, 1, mode='edge')

            # Store max slope and direction per pixel
            max_slope = cp.full_like(dem, -cp.inf, dtype=cp.float32)

            for dir_code, (dr, dc) in flow_offsets.items():
                D = direction_distance[dir_code]
                # Shifted DEM (neighbor elevations)
                neighbor = dem_padded[1 + dr : 1 + dr + nrows,
                                    1 + dc : 1 + dc + ncols]

                # Slope = (Z_center - Z_neighbor) / D
                slope = (dem - neighbor) / D

                # Update direction where slope is highest
                mask = slope > max_slope
                flow_dir = cp.where(mask, dir_code, flow_dir)
                max_slope = cp.maximum(max_slope, slope)

            return flow_dir

        dem_cp = cp.asarray(self.dem)
        self.fd = d8_flow_direction(dem_cp)
        return self.fd

class FlowAccumulation(GenericAlgorithm):
    def load_inputs(self):
        self.fd = self.tif_handler.load_with_padding(self.cfg.FLOWDIRECTION_PATH)

    def save_outputs(self):
        self.tif_handler.save_tiff(self.facc_gpu.get(), self.cfg.FLOWACCUMULATION_PATH)

    def main(self):

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

        def flow_accum_topo(flow_dir_gpu) -> cp.ndarray: # Renamed input for clarity
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

        fd_gpu = cp.asarray(self.fd)
        self.facc_gpu = flow_accum_topo(fd_gpu)
        return self.facc_gpu
