from compute_runoff_matrices import compute_sr_and_CNs
from utils import GeoTIFFHandler, make_logger
import cupy as cp

SOIL_TIFF = '../tifs/lgb/INPUTS/soil_clipped.tif'
LULC_TIFF = '../tifs/lgb/INPUTS/lulc_clipped.tif'
SLOPE_TIFF = '../tifs/lgb/INPUTS/dem_clipped.tif'
REFERENCE_TIFF = '../tifs/lgb/INPUTS/dem_clipped.tif'
SAVE_DIRECTORY = '../tifs/lgb/'

if __name__ == "__main__":
    logger = make_logger("logs/sr.log", gpu_mem_usage=True)
    handler = GeoTIFFHandler(REFERENCE_TIFF, logger)
    sr1, sr2, sr3 = compute_sr_and_CNs(
        cp.asarray(handler.load_with_padding(SOIL_TIFF)),
        cp.asarray(handler.load_with_padding(LULC_TIFF)),
        cp.asarray(handler.load_with_padding(SLOPE_TIFF)),
    )
    handler.save_tiff(sr1.get(), SAVE_DIRECTORY+'sr1.tif')
    handler.save_tiff(sr2.get(), SAVE_DIRECTORY+'sr2.tif')
    handler.save_tiff(sr3.get(), SAVE_DIRECTORY+'sr3.tif')
