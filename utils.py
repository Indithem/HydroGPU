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
