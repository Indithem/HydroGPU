from . import GenericAlgorithm, GeoTIFFHandler
import cupy as cp
import numpy as np

class Microwatersheds(GenericAlgorithm):
    def __init__(self, args, tif_handler: GeoTIFFHandler, threshold=1000) -> None:
        """
            adjust threshold as needed. masalia required smaller(10) threshold but lgb higher(10,000 etc)
        """
        super().__init__(args, tif_handler)
        tif_handler.logger.info(f"Using threshold {threshold}")
        self.STREAM_THRESHOLD = threshold

    def load_inputs(self):
        self.fd = self.tif_handler.load_with_padding(self.cfg.FLOWDIRECTION_PATH)
        self.facc = self.tif_handler.load_with_padding(self.cfg.FLOWACCUMULATION_PATH)

    def save_outputs(self):
        self.tif_handler.save_tiff(self.mws.get(), self.cfg.MICROWATERSHEDS_PATH)

    def main(self):
        # Define direction offsets mapping 1-8 to (row_offset, col_offset)
        # Matches the convention used in the flow accumulation code
        shift_offsets = {
            1: (-1, 1),   # Top-right
            2: (-1, 0),   # Top-middle
            3: (-1, -1),  # Top-left
            4: (0, -1),   # Left
            5: (1, -1),   # Bottom-left
            6: (1, 0),    # Bottom-middle
            7: (1, 1),    # Bottom-right
            8: (0, 1)     # Right
        }

        # Mapping from offset (dr, dc) to the direction value (1-8)
        offset_to_direction = {v: k for k, v in shift_offsets.items()}

        # Define inverse direction offsets mapping (dr, dc) to the direction value that flows *from* that neighbor
        # E.g., to find neighbors that flow *into* a cell from the top-right, we look for cells
        # whose flow direction is 5 (bottom-left) and are located at (r-1, c+1) relative to the current cell.
        inverse_shift_offsets = {
            (1, -1): 1,   # From Bottom-left (offset 1,-1) -> Flow Dir 1 (Top-right)
            (1, 0): 2,    # From Bottom-middle (offset 1,0) -> Flow Dir 2 (Top-middle)
            (1, 1): 3,    # From Bottom-right (offset 1,1) -> Flow Dir 3 (Top-left)
            (0, 1): 4,    # From Right (offset 0,1) -> Flow Dir 4 (Left)
            (-1, 1): 5,   # From Top-right (offset -1,1) -> Flow Dir 5 (Bottom-left)
            (-1, 0): 6,   # From Top-middle (offset -1,0) -> Flow Dir 6 (Bottom-middle)
            (-1, -1): 7,  # From Top-left (offset -1,-1) -> Flow Dir 7 (Bottom-right)
            (0, -1): 8    # From Left (offset 0,-1) -> Flow Dir 8 (Right)
        }


        def calculate_microwatersheds_by_convergence(flow_direction, stream_segment_lines):
            """
            Calculates micro-watershed delineation using CuPy,
            identifying outlets based on stream network convergence points.

            flow_direction: CuPy array (H, W) of D8 flow directions (1-8, 0 for no flow/boundary).
            stream_segment_lines: CuPy array (H, W) where 1 indicates a stream cell, 0 otherwise.

            Returns:
                CuPy array (H, W) containing micro-watershed basin IDs.
                Non-basin cells will have ID 0.
            """
            H, W = flow_direction.shape

            # Identify the stream network based on the input raster
            stream_mask = (stream_segment_lines == 1)
            stream_mask = cp.asarray(stream_mask, dtype='bool')

            # Initialize basin ID matrix (0 for unassigned/non-basin)
            basin_grid = cp.zeros_like(flow_direction, dtype=cp.int32)

            # Keep track of cells whose basin ID has been assigned
            assigned_mask = cp.zeros_like(flow_direction, dtype='bool')

            # 1. Identify potential outlet points based on convergence
            # These are cells on the stream network whose downstream neighbor (also on the stream network)
            # has more than one incoming neighbor on the stream network.

            # Calculate incoming neighbor count on the stream network for each cell
            incoming_neighbor_count = cp.zeros_like(flow_direction, dtype=cp.int32)

            # Iterate through all cells on the stream network
            stream_rows, stream_cols = cp.nonzero(stream_mask)

            # Only proceed if there are stream cells
            if stream_rows.size == 0:
                print("No stream cells found in the input stream segment lines. No basins delineated.")
                return basin_grid

            # For each cell on the stream network, find its downstream neighbor
            stream_directions = flow_direction[stream_rows, stream_cols]
            print(f"Processing {stream_rows.size} stream cells for incoming neighbor counts...")
            # Iterate through stream cells and increment the incoming count of their downstream neighbors
            for i in range(stream_rows.size):
                # print(f"  Processing stream cell {i+1}/{stream_rows.size}...")
                r, c = stream_rows[i], stream_cols[i]
                dir_val = stream_directions[i]

                # Skip if no flow direction or invalid direction
                if dir_val == 0 or int(dir_val) not in shift_offsets:
                    continue

                dr, dc = shift_offsets[int(dir_val)]
                nr, nc = r + dr, c + dc

                # Check if downstream neighbor is within bounds and is also on the stream network
                if nr >= 0 and nr < H and nc >= 0 and nc < W and stream_mask[nr, nc]:
                    incoming_neighbor_count[nr, nc] += 1
            print("Completed incoming neighbor count calculation.")
            # Identify cells on the stream network that are downstream neighbors receiving > 1 incoming neighbors
            convergence_points_mask = (incoming_neighbor_count > 1) & stream_mask

            # Doubt: Doesn't (incoming_neighbout_count > 1) imply stream_mask?
            # see: if condition, we do += 1 when stream_mask[nc, nr] only.

            # The outlets are the cells *upstream* of these convergence points, on the stream network.
            # We need to find cells on the stream network whose flow direction leads to a convergence point.
            potential_outlet_mask = cp.zeros_like(flow_direction, dtype='bool')

            # Iterate through all possible flow directions
            print("Identifying potential outlet cells based on convergence points...")
            for dir_val, (dr, dc) in shift_offsets.items():
                print(f"  Checking flow direction {dir_val}...")
                # Find cells in the grid with this flow direction
                cells_with_this_dir_mask = (flow_direction == dir_val)
                cells_with_this_dir_rows, cells_with_this_dir_cols = cp.nonzero(cells_with_this_dir_mask)

                if cells_with_this_dir_rows.size == 0:
                    continue

                # Calculate their downstream neighbors
                down_rows = cells_with_this_dir_rows + dr
                down_cols = cells_with_this_dir_cols + dc

                # Check bounds
                valid_down_mask = (down_rows >= 0) & (down_rows < H) & \
                                  (down_cols >= 0) & (down_cols < W)

                valid_up_rows = cells_with_this_dir_rows[valid_down_mask]
                valid_up_cols = cells_with_this_dir_cols[valid_down_mask]
                valid_down_rows = down_rows[valid_down_mask]
                valid_down_cols = down_cols[valid_down_mask]

                # Check if the downstream neighbor is a convergence point AND the upstream cell is on the stream network
                if valid_up_rows.size > 0:
                    is_convergence_point_downstream_mask = convergence_points_mask[valid_down_rows, valid_down_cols]
                    is_on_stream_upstream_mask = stream_mask[valid_up_rows, valid_up_cols]

                    # Cells that are on the stream network and flow into a convergence point
                    outlets_this_dir_mask = is_convergence_point_downstream_mask & is_on_stream_upstream_mask

                    potential_outlet_mask[valid_up_rows[outlets_this_dir_mask], valid_up_cols[outlets_this_dir_mask]] = True

            print("Potential outlet cells identified.")
            # Find coordinates of the actual outlet cells based on this convergence logic
            outlet_rows, outlet_cols = cp.nonzero(potential_outlet_mask)

            # If no outlets found by convergence, this might indicate a single main stem or issue with inputs
            if outlet_rows.size == 0:
                print("No convergence outlets found. Basins may not be delineated as expected.")
                # In some cases, the most downstream point on the main stem might be the only "outlet"
                # A fallback could be to find the cell on the stream network with the highest flow accumulation
                # if flow accumulation was available, but sticking to the convergence logic for now.
                # For this logic, if no convergence points, there are no basins defined by this method.
                return basin_grid


            # 2. Assign unique IDs to the identified outlet cells
            # Flattened index = row * W + col + 1 (add 1 so IDs are positive)
            outlet_flat_indices = outlet_rows * W + outlet_cols + 1
            basin_grid[outlet_rows, outlet_cols] = outlet_flat_indices
            assigned_mask[outlet_rows, outlet_cols] = True # Outlet cells are assigned their basin ID initially

            print(f"Identified {outlet_rows.size} convergence outlet cells and assigned initial basin IDs.")

            # 3. Iteratively Propagate Basin IDs Upstream
            # We propagate the basin ID of cells whose ID was just assigned.
            # In each iteration, we find unassigned cells that flow *into*
            # the cells assigned in the previous step and assign them the same basin ID.

            # Mask of cells whose basin ID was just assigned and needs to propagate upstream
            # Initially, this is just the specific outlet cells
            propagating_mask = potential_outlet_mask.copy() # Start propagation from the identified outlets

            iteration = 0
            # Loop while there are still cells whose basin ID was determined in the last step
            # and need to propagate their ID upstream.
            while cp.any(propagating_mask):
                iteration += 1
                # print(f"Propagation iteration {iteration}...") # Uncomment for detailed iteration progress

                # Create mask for cells whose basin ID gets determined in THIS iteration
                next_propagating_mask = cp.zeros_like(flow_direction, dtype='bool')

                # Find coordinates of cells in the current propagating_mask
                prop_rows, prop_cols = cp.nonzero(propagating_mask)

                if prop_rows.size == 0:
                     break

                # Iterate through all possible upstream neighbor offsets (dr_up, dc_up)
                for dr_up, dc_up in shift_offsets.values():
                    flow_dir_value = offset_to_direction[(dr_up, dc_up)]
                    potential_upstream_mask = (flow_direction == flow_dir_value)
                    potential_up_rows, potential_up_cols = cp.nonzero(potential_upstream_mask)

                    if potential_up_rows.size == 0:
                        continue

                    down_rows = potential_up_rows + dr_up
                    down_cols = potential_up_cols + dc_up

                    valid_down_mask = (down_rows >= 0) & (down_rows < H) & \
                                      (down_cols >= 0) & (down_cols < W)

                    valid_potential_up_rows = potential_up_rows[valid_down_mask]
                    valid_potential_up_cols = potential_up_cols[valid_down_mask]
                    valid_down_rows = down_rows[valid_down_mask]
                    valid_down_cols = down_cols[valid_down_mask]

                    downstream_is_propagating_mask = propagating_mask[valid_down_rows, valid_down_cols]

                    actual_up_rows = valid_potential_up_rows[downstream_is_propagating_mask]
                    actual_up_cols = valid_potential_up_cols[downstream_is_propagating_mask]
                    corresponding_down_rows = valid_down_rows[downstream_is_propagating_mask]
                    corresponding_down_cols = valid_down_cols[downstream_is_propagating_mask]


                    if actual_up_rows.size > 0:
                        # We only assign if the cell is not already assigned.
                        is_unassigned_mask = ~assigned_mask[actual_up_rows, actual_up_cols]

                        cells_to_assign_rows = actual_up_rows[is_unassigned_mask]
                        cells_to_assign_cols = actual_up_cols[is_unassigned_mask]
                        corresponding_down_rows_to_assign = corresponding_down_rows[is_unassigned_mask]
                        corresponding_down_cols_to_assign = corresponding_down_cols[is_unassigned_mask]


                        if cells_to_assign_rows.size > 0:
                            basin_ids_to_propagate = basin_grid[corresponding_down_rows_to_assign, corresponding_down_cols_to_assign]

                            basin_grid[cells_to_assign_rows, cells_to_assign_cols] = basin_ids_to_propagate

                            assigned_mask[cells_to_assign_rows, cells_to_assign_cols] = True

                            next_propagating_mask[cells_to_assign_rows, cells_to_assign_cols] = True

                propagating_mask = next_propagating_mask
                # print(f"  Iteration {iteration}: Assigned basin IDs to {cp.asnumpy(cp.sum(next_propagating_mask))} cells.")

            print("Micro-watershed delineation finished.")

            # The final basin_grid contains the delineated micro-watersheds.
            # Cells with value 0 did not drain to a stream cell identified by the threshold.
            return basin_grid


        fd = cp.asarray(self.fd)
        facc = cp.asarray(self.facc)
        facc_stream_cells = (facc >= self.STREAM_THRESHOLD).astype(cp.int32)
        self.mws = calculate_microwatersheds_by_convergence(fd, facc_stream_cells)
        return self.mws
