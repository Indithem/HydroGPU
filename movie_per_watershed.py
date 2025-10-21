import cupy as cp
from utils import GeoTIFFHandler, make_logger, load_tif_image
import glob
from tqdm import tqdm
import os

logger = make_logger("logs/perms_vec.log")


def per_watershed_avg_runoff(watershed_raster, runoff_raster):
    runoff_raster = cp.asarray(runoff_raster)
    watershed_raster = cp.asarray(watershed_raster)

    ws_flat = watershed_raster.ravel()
    rf_flat = runoff_raster.ravel()

    unique_ws, inv = cp.unique(ws_flat, return_inverse=True)

    per_ws_sum = cp.bincount(inv, weights=rf_flat, minlength=unique_ws.size)
    per_ws_count = cp.bincount(inv, minlength=unique_ws.size)

    per_ws_mean = per_ws_sum / per_ws_count

    result = per_ws_mean[inv].reshape(watershed_raster.shape)

    # return result, (per_ws_mean.get(), unique_ws.get())
    return result

if __name__ == "__main__":
    MICROWATERSHED_RASTER = '../tifs/lgb/mwshed.tif'
    # RUNOFF_FOLDER = "../tifs/lgb/run_off_sim/"
    # MOVIE_RASTERS_FOLDER = '../tifs/lgb/mws_movie/'
    RUNOFF_FOLDER = "../test/rain_subset"
    MOVIE_RASTERS_FOLDER = '../tifs/lgb/movie_rainfall/'


    watershed_raster = load_tif_image(MICROWATERSHED_RASTER)
    handler = GeoTIFFHandler(MICROWATERSHED_RASTER, logger)
    runoff_files = sorted(glob.glob(f"{RUNOFF_FOLDER}/*.tif"))

    # testing
    logger.info(f"there are {len(runoff_files)} runoff files")

    for runoff_file in tqdm(runoff_files):
        runoff_raster = load_tif_image(runoff_file)
        movie_raster = per_watershed_avg_runoff(watershed_raster, runoff_raster)
        handler.save_tiff(movie_raster.get(), MOVIE_RASTERS_FOLDER+os.path.basename(runoff_file))
