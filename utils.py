import rasterio
import os
import logging
import numpy.typing as nptypes
import cupy as cp
import numpy as np
from tqdm import tqdm
from typing import Any, List


def load_tif_image(file_path) -> nptypes.NDArray[Any]:
    """Load a .tif image efficiently and return a NumPy array (float32)."""
    with rasterio.open(file_path) as src:
        image = src.read(1)  # Read only the first band
        print(f"Loaded Raster - Shape: {image.shape}, Dtype: {image.dtype}")
        return image  # Kept as NumPy array for easier slicing

def make_logger(file_name, gpu_mem_usage=False, level=logging.INFO):
    # Create logger
    logger = logging.getLogger("mylog")
    # without level, nothing gets print. I guess level is set to WARN etc
    logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(file_name)
    fh.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    if gpu_mem_usage:
        class GPUMemUsageAdapter(logging.LoggerAdapter):
            def __init__(self, logger, extra) -> None:
                super().__init__(logger, extra)
                # pynvml.nvmlInit()
                # Get handle for the first GPU (device 0)
                # Loop pynvml.nvmlDeviceGetCount() for multi-GPU setups
                # self.GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.mempool = cp.get_default_memory_pool()
                # logger.info(f"pynvml initialized {self.GPU_HANDLE}")

            def process(self, msg, kwargs):
                # mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.GPU_HANDLE)
                used_bytes = self.mempool.used_bytes()
                # cp.cuda.runtime.memGetInfo() returns (free, total) bytes
                free_device, total_device = cp.cuda.runtime.memGetInfo()
                gpu_mem_str = f" | CuPy Used: {used_bytes / (1024**2):.0f}/{total_device / (1024**2):.0f} MiB"
                return msg + gpu_mem_str, kwargs

        return GPUMemUsageAdapter(logger, {})

    return logger


class GeoTIFFHandler:
    """
        Saves tiff file with correct crs and transforms.
    """

    def __init__(self, tiff_path: str, logger: logging.Logger):
        """
        Initialize by loading an existing GeoTIFF file and storing its properties.
        """
        with rasterio.open(tiff_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.width = src.width
            self.height = src.height
            self.dtype = src.dtypes[0]  # Get the data type of the first band
            self.count = src.count  # Number of bands
            self.window = src.window

            # Read original data (optional)
            # self.original_data = src.read(1)

        self.logger = logger
        logger.info(f"Loaded TIFF: {tiff_path}")
        logger.info(f"CRS: {self.crs}, Transform: {self.transform}, Size: {self.width}x{self.height}, Type: {self.dtype}, Window: {self.window}")

    def save_tiff(self, new_data, output_path: str):
            """
            Save a new TIFF file using the stored properties but with new data.

            :param new_data: 2D NumPy array containing new raster data.
            :param output_path: Path to save the new TIFF file.
            :param compression: Compression type for the TIFF file (default: "LZW").
            """
            if new_data.shape != (self.height, self.width):
                raise ValueError(f"Data shape {new_data.shape} does not match the stored shape ({self.height}, {self.width})")

            # output_dir = os.path.dirname(output_path)
            # if output_dir:
            #     os.makedirs(output_dir, exist_ok=True)

            gdal_options = {
                    'tiled': True,
                    'compress': 'ZSTD',      # <--- Use Zstandard
                    # 'ZSTD_LEVEL': 6,         # <--- Set to lowest level (1=fastest, 22=best)
                    'ZSTD_LEVEL': 1,         # <--- Set to lowest level (1=fastest, 22=best)
                    'NUM_THREADS': 10        # <--- Essential for speed. we have 12 cores.
                }

            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=self.height,
                width=self.width,
                count=self.count,
                dtype=new_data.dtype,
                crs=self.crs,
                transform=self.transform,
                # compress=compression
                **gdal_options
            ) as dst:
                dst.write(new_data, 1)  # Write new data to band 1

            self.logger.info(f"Saved new TIFF to {output_path}")

    def save_multiband_tiff(self, output_path:str, data_arrays: list[nptypes.NDArray[Any]], compression="LZW"):
        """
        Save multiple 2D NumPy arrays as bands in a single GeoTIFF.
        data_arrays: list or tuple of 2D numpy arrays with identical shape.
        """
        count = len(data_arrays)
        dtype = data_arrays[0].dtype

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=self.height,
            width=self.width,
            count=count,
            dtype=dtype,
            crs=self.crs,
            transform=self.transform,
            compress=compression
        ) as dst:
            for i, arr in tqdm(enumerate(data_arrays, start=1)):
                dst.write(arr, i)

        self.logger.info("Done writing to " + output_path)

    def load_with_padding(self, src_path, fill_value=0):
        with rasterio.open(src_path) as src:
            if src.crs != self.crs:
                raise ValueError("CRS mismatch; reproject first.")

            # Read source
            data = src.read(1)

            # Compute windows and offsets
            src_bounds = src.bounds
            # ref_bounds = self.bounds
            # ref_transform = self.transform
            # ref_res_x = ref_transform.a
            # ref_res_y = -ref_transform.e

            # Convert bounding boxes to pixel coordinates in the reference frame
            ref_window = self.window(*src_bounds)
            row_off = int(ref_window.row_off)
            col_off = int(ref_window.col_off)

            # Create full reference-size array and insert the source data
            padded = np.full((self.height, self.width), fill_value, dtype=data.dtype)

            # Compute slice positions within reference grid
            dest_row0 = max(0, row_off)
            dest_col0 = max(0, col_off)
            dest_row1 = min(self.height, dest_row0 + src.height)
            dest_col1 = min(self.width, dest_col0 + src.width)

            # Paste source data if overlapping
            src_row0 = max(0, -row_off)
            src_col0 = max(0, -col_off)
            src_row1 = src_row0 + (dest_row1 - dest_row0)
            src_col1 = src_col0 + (dest_col1 - dest_col0)

            padded[dest_row0:dest_row1, dest_col0:dest_col1] = data[src_row0:src_row1, src_col0:src_col1]

            self.logger.info("Loaded file " + src_path)
            return padded
