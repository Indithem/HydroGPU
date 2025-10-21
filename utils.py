import rasterio
import os
import logging

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def load_tif_image(file_path):
    """Load a .tif image efficiently and return a NumPy array (float32)."""
    with rasterio.open(file_path) as src:
        image = src.read(1)  # Read only the first band
        print(f"Loaded Raster - Shape: {image.shape}, Dtype: {image.dtype}")
        return image  # Kept as NumPy array for easier slicing

def make_logger(file_name, level=logging.INFO):
    # Create logger
    logger = logging.getLogger("mylog")
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

            # Read original data (optional)
            # self.original_data = src.read(1)

        self.logger = logger
        logger.info(f"Loaded TIFF: {tiff_path}")
        logger.info(f"CRS: {self.crs}, Transform: {self.transform}, Size: {self.width}x{self.height}, Type: {self.dtype}")

    def save_tiff(self, new_data, output_path: str, compression="LZW"):
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
                compress=compression
            ) as dst:
                dst.write(new_data, 1)  # Write new data to band 1

            self.logger.info(f"Saved new TIFF with compression='{compression}': {output_path}")
