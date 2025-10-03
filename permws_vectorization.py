import cupy as cp
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
import rasterio
from rasterio.features import shapes
import json
import glob
from utils import load_tif_image, make_logger
import tqdm
from pprint import pprint
import time
from collections import defaultdict

logger = make_logger("perms_vec.log")

def per_watershed_runoff(watershed_raster, runoff_raster):
    # Ensure the input arrays are on the same device
    # if cp.cuda.runtime.getDeviceCount() > 0:
    runoff_raster = cp.asarray(runoff_raster)
    watershed_raster = cp.asarray(watershed_raster)

    # Compute the total runoff for each watershed
    logger.info("transferred arrays to gpu")
    # unique_watersheds = cp.unique(watershed_raster)
    start_time=time.time()

    # Flatten arrays so we can use scatter-like operations
    ws_flat = watershed_raster.ravel()
    rf_flat = runoff_raster.ravel()

    # Map each pixel to a compact watershed index
    unique_ws, inv = cp.unique(ws_flat, return_inverse=True)
    logger.info("got unique watersheds")

    # Compute sum of runoff per watershed
    per_ws_sum = cp.bincount(inv, weights=rf_flat, minlength=unique_ws.size)

    # Map each pixel back to its watershed's total
    result = per_ws_sum[inv].reshape(watershed_raster.shape)

    print(f"took {time.time()-start_time}")
    # per_ws_rf_raster = cp.zeros_like(watershed_raster, dtype=runoff_raster.dtype)
    # logging.info("start compute")

    # for i, watershed in tqdm.tqdm(enumerate(unique_watersheds)):
    #     per_ws_rf_raster[watershed_raster == watershed] = cp.sum(runoff_raster[watershed_raster == watershed])

    return result, (per_ws_sum.get(), unique_ws.get())

def create_geojson(watershed_raster, runoff_dict, transform=None, crs="EPSG:4326"):
    """
    Compute per-watershed runoff and return as GeoJSON.

    Parameters:
    - watershed_raster: 2D array of watershed IDs
    - transform: affine transform for raster -> geo coordinates (optional)
    - crs: coordinate reference system string (optional)

    Returns:
    - geojson_dict: GeoJSON FeatureCollection with per-watershed runoff
    """
    # Transfer to GPU
    ws_raster = cp.asarray(watershed_raster)

    # Flatten arrays
    ws_flat = ws_raster.ravel()

    # Unique watersheds + indices
    unique_ws, inv = cp.unique(ws_flat, return_inverse=True)

    # Bring back to CPU
    unique_ws = unique_ws.get()

    # Convert raster to polygons using rasterio.features.shapes
    if transform is None:
        transform = rasterio.transform.from_origin(0, 0, 1, 1)  # dummy transform

    polygons = []
    for geom, val in shapes(watershed_raster.astype(np.int32), mask=None, transform=transform):
        polygons.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "watershed_id": int(val),
                "runoffs_each_day": runoff_dict[val]
            }
        })

    geojson_dict = {
        "type": "FeatureCollection",
        "features": polygons
    }

    return geojson_dict

def main(rainfall_folder, watershed_raster, json_filename):
    watershed_raster = load_tif_image(watershed_raster)
    rainfall_files = sorted(glob.glob(f"{rainfall_folder}/*.tif"))
    # testing
    #rainfall_files = rainfall_files[:2]
    logger.info(f"there are {len(rainfall_files)} rainfall files")

    time_series = defaultdict(list)

    for runoff_file in tqdm.tqdm(rainfall_files):
        runoff_raster = load_tif_image(runoff_file)
        _, (per_ws_sum, ws_ids) = per_watershed_runoff(watershed_raster, runoff_raster)
        for ws_id, runoff in tqdm.tqdm(zip(ws_ids, per_ws_sum)):
            time_series[int(ws_id)].append(float(runoff))

    logger.info('creating json')
    geojson = create_geojson(watershed_raster, time_series)
    with open(json_filename, 'w') as f:
        json.dump(geojson, f)


if __name__ == "__main__":
    # OUTPUT_FOLDER = "../tifs/lgb/rainfall_time_series_complete"
    RAINFALL_FOLDER = "../test/rain_subset"
    MICROWATERSHED_RASTER = '../tifs/lgb/mwshed.tif'
    GEO_JSON_FILE = '../tifs/lgb/mwshed_and_runoffs.geojson'
    main(RAINFALL_FOLDER, MICROWATERSHED_RASTER, GEO_JSON_FILE)
    logger.info("Done")
