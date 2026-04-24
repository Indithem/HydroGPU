import gc
import json
import os
import pathlib
import shutil
import threading
import time
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import config as cfg
import xarray as xr
import requests
import concurrent.futures

import zarr
from rasterio.enums import Resampling
from shapely.geometry import shape
from rasterio.transform import from_origin
from tqdm import tqdm
# import geedim as gd
import cupy as cp
import numpy as np
import pandas as pd
import cucim.skimage.transform as cimg

from downloads import GenericDownloader, ee, Logger

class DownloaderBase(GenericDownloader):
    def __init__(self):

        super().__init__()
        self.zarr_path = os.path.join(cfg.RAINFALL_FOLDER, "rainfall_archive.zarr")

    def init_zarr(self, dates, dummy_da):
        """
        Initializes a Zarr store with the full time extent to allow parallel region writes.
        """
        native_y, native_x = dummy_da.y.size, dummy_da.x.size

        # Create the skeleton Dataset
        # We use empty/zeros but with compute=False, so no data is actually written yet
        ds_skeleton = xr.Dataset(
            {"precipitation": (["time", "y", "x"],
                               np.zeros((len(dates), native_y, native_x), dtype='float32'))},
            coords={
                "time": dates,
                "y": dummy_da.y.values,
                "x": dummy_da.x.values
            }
        )

        # Metadata/CRS (Important for GeoZarr)
        ds_skeleton.rio.write_crs(dummy_da.rio.crs, inplace=True)
        ds_skeleton.rio.write_transform(dummy_da.rio.transform(), inplace=True)

        # Encoding: chunking by 1 day is standard for daily time-series
        encoding = {
            "precipitation": {"chunks": (1, native_y, native_x)},
            "time": {
                "units": "hours since 2017-07-01 00:00:00",
                "calendar": "proleptic_gregorian",
                "dtype": "int64"  # Ensuring integer storage for hours
            }
        }

        # Write ONLY metadata
        ds_skeleton.to_zarr(self.zarr_path, mode='w', encoding=encoding, compute=False,  zarr_format=2)
        self.logger.info(f"Initialized Zarr skeleton at {self.zarr_path} with {len(dates)} slots.")


    def save_geozarr(self, flat_data, timestamp, dummy_da, t):
        """
        Saves data as a 3D (Time, Y, X) chunked array with full spatial coordinates.
        """

        # 1. Reshape flat data to native grid dimensions
        # Using the shape from self.dummy_da (e.g., [Lat, Lon])
        native_y, native_x = dummy_da.y.size, dummy_da.x.size
        raster_data = flat_data.reshape(native_y, native_x)

        # 2. Create the DataArray with proper spatial coords
        da = xr.DataArray(
            raster_data[np.newaxis, ...],  # Shape: (1, Y, X)
            dims=("time", "y", "x"),
            coords={
                "time": [pd.to_datetime(timestamp, format='%Y%m%d_%H')],
                "y": dummy_da.y.values,
                "x": dummy_da.x.values
            },
            name="precipitation"
        )

        # 3. Add CRS and metadata
        da.rio.write_crs(dummy_da.rio.crs, inplace=True)
        da.rio.write_transform(dummy_da.rio.transform(), inplace=True)

        ds = da.to_dataset().drop_vars(["y", "x", "spatial_ref"])

        ds.to_zarr(self.zarr_path, region={"time": slice(t, t + 1)})

    def load_geozarr(self):
        """
        Loads the Zarr archive as a standard Xarray Dataset.
        """

        if not os.path.exists(self.zarr_path):
            self.logger.error(f"Zarr not found at {self.zarr_path}")
            return None

        # chunks={} opens it lazily using Dask
        ds = xr.open_zarr(self.zarr_path, consolidated=True, chunks={})
        return ds

class Download_to_database(DownloaderBase):
    def main(self):
        self.ingest_rainfall_to_zarr()

    def ingest_rainfall_to_zarr(self):
        """
        Part 1: Purely fetches data, sums it on GPU, and saves to GeoZarr.
        """
        self.logger.info("Starting rainfall ingestion to GeoZarr")
        region = self.load_region()
        buffered_region = region.buffer(12000).bounds()

        rainfall_collection = (
            ee.ImageCollection('JAXA/GPM_L3/GSMaP/v6/operational')
            .filterDate(cfg.ARG_START_DATE, cfg.ARG_END_DATE)
            .select('hourlyPrecipRate')
        )

        # rainfall_collection = (
        #     ee.ImageCollection("NASA/GPM_L3/IMERG_DAILY_V06")
        #     .filterDate(cfg.ARG_START_DATE, cfg.ARG_END_DATE)
        #     .select('total_accum')
        # )
        #
        # # 1. Define the time range
        # start_date = ee.Date(cfg.ARG_START_DATE)
        # end_date = ee.Date(cfg.ARG_END_DATE)
        #
        # # 2. Calculate the number of days between start and end
        # n_days = end_date.difference(start_date, 'days')
        #
        # def sum_daily(day_offset):
        #     # Calculate the start and end of each 24-hour window
        #     start = start_date.advance(ee.Number(day_offset), 'days')
        #     end = start.advance(1, 'days')
        #
        #     # Filter the collection for this specific day and sum
        #     daily_sum = (rainfall_collection
        #                  .filterDate(start, end)
        #                  .sum())  # Sums the 'hourlyPrecipRate'
        #
        #     # Return the image with its date metadata (important for further filtering)
        #     return daily_sum.set({
        #         'system:time_start': start.millis(),
        #         'date_string': start.format('YYYY-MM-DD')
        #     })
        #
        # # 3. Create a sequence of days and map the function
        # daily_collection = ee.ImageCollection(
        #     ee.List.sequence(0, n_days.subtract(1)).map(sum_daily)
        # )

        first_img = rainfall_collection.first()
        native_proj = first_img.projection()

        ds = xr.open_dataset(
            rainfall_collection,
            engine='ee',
            projection=native_proj,
            geometry=buffered_region,
            fast_time_slicing=True,
        )

        da = ds['hourlyPrecipRate'].rename({'lat': 'y', 'lon': 'x'}).transpose("time", "y", "x")
        total_pixels = da.y.size * da.x.size
        dummy_da = da.isel(time=0)

        N = len(da.time)
        K = 24  # Hours per day

        dates = pd.date_range(start=cfg.ARG_START_DATE, end=cfg.ARG_END_DATE, freq='D')
        self.init_zarr(dates, dummy_da)

        ASK_BUFF = 50
        WORKERS = 20

        def process_slice(t, ticket_num):
            da_slice = da.isel(time=slice(t, min(t + K, N)))

            # Extract raw data and sum on GPU
            raw_data = da_slice.values
            gpu_sum = cp.zeros(total_pixels, dtype=cp.float32)

            for i in range(raw_data.shape[0]):
                gpu_sum += cp.asarray(raw_data[i]).ravel()

            timestamp = da_slice.time[0].dt.strftime('%Y%m%d_%H').item()

            # Save the flat sum to the database
            # Assuming self.save_geozarr handles time-indexing inside the Zarr
            self.save_geozarr(gpu_sum.get(), timestamp, dummy_da, ticket_num)

        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor,
            tqdm(range(N), desc="Downloading Rainfall") as pbar
        ):
            futures = []

            while pbar.n < pbar.total:
                if len(futures) == 0:
                    gc.collect()
                    t = pbar.n
                    assert t % K == 0
                    for i in range(ASK_BUFF):
                        current_hour = t+i*K
                        day_index = current_hour // K
                        if current_hour >= N:
                            break
                        futures.append(executor.submit(
                            process_slice,
                            current_hour,
                            day_index
                        ))
                futures.pop().result()
                pbar.update(K)

        zarr.consolidate_metadata(self.zarr_path)
        self.logger.info("Ingestion complete.")

class Load_from_database(DownloaderBase):

    def __init__(self):
        super().__init__()


        region = self.load_region()
        buffered_region = region.buffer(12000).bounds()

        clipping_geom = shape(region.getInfo())

        rainfall_collection = (
            ee.ImageCollection('JAXA/GPM_L3/GSMaP/v6/operational')
            .filterDate(cfg.ARG_START_DATE, cfg.ARG_END_DATE)
            .select('hourlyPrecipRate')
        )

        first_img = rainfall_collection.first()
        native_proj = first_img.projection()
        ds = xr.open_dataset(
            rainfall_collection,
            engine='ee',
            # chunks={'time': 48},
            projection=native_proj,  # <--- CRITICAL: Matches original GEE alignment
            geometry=buffered_region,  # <--- Ensures the extent covers your region
            # scale=0.1             # Remove this; projection already contains the 0.1 scale
            fast_time_slicing=True,
        )

        # Prepare the whole DataArray once
        da = ds['hourlyPrecipRate'].rename({'lat': 'y', 'lon': 'x'})
        da = da.transpose("time", "y", "x")

        # Convert GEE resolution (30m) to decimal degrees for EPSG:4326
        # 111,319.49m is the approximate length of 1 degree at the equator
        target_scale_meters = cfg.GEE_SCALE  # 30
        deg_per_meter = 1 / 111319.49
        target_res_degrees = target_scale_meters * deg_per_meter

        # 1. Calculate dimensions
        y_size, x_size = da.y.size, da.x.size
        total_pixels = y_size * x_size

        # 2. Create the Index Map
        # The 'sink_index' is the very last position (index = total_pixels)
        source_indices = np.arange(total_pixels).reshape(y_size, x_size).astype(np.int32)

        self.dummy_da = da.isel(time=0)
        dummy_da = self.dummy_da.copy(data=source_indices)
        dummy_da.rio.write_crs("EPSG:4326", inplace=True)

        # CRITICAL: Set nodata to the sink index
        # This ensures clipped areas are filled with 'total_pixels'
        dummy_da.rio.write_nodata(total_pixels, inplace=True)

        # 4. Reproject and Clip
        # Areas outside clipping_geom will now contain the value: total_pixels
        mapped_indices_da = dummy_da.rio.reproject(
            dst_crs="EPSG:4326",
            resolution=target_res_degrees,
            resampling=Resampling.nearest
        ).rio.clip([clipping_geom], crs="EPSG:4326", drop=True, all_touched=True)

        # 5. Transfer LUT to GPU
        self.GPU_LUT = cp.asarray(mapped_indices_da.values)

        # Move the LUT to GPU
        # 4. Extract static metadata and move LUT to GPU
        self.STATIC_METADATA = {
            "bounds": mapped_indices_da.rio.bounds(),
            "transform": mapped_indices_da.rio.transform(),
            "crs": mapped_indices_da.rio.crs,
            "shape": mapped_indices_da.shape,
            "original_size": total_pixels
        }

    def main(self):
        self.logger.info("Starting rainfall ingestion to GeoZarr")

        yield from self.stream_reprojected_rainfall()

    def stream_reprojected_rainfall(self):
        ds = self.load_geozarr()
        
        for i in tqdm(range(len(ds.time))):
            # Extract 2D slice and flatten for the GPU LUT
            hourly_slice = ds.precipitation.isel(time=i)
            flat_cpu = hourly_slice.values.ravel() 
            
            # Move to GPU
            gpu_src = cp.asarray(flat_cpu)
            
            # Append sink pixel for nodata and apply LUT
            projected_buffer = cp.concatenate([gpu_src, cp.array([0], dtype=gpu_src.dtype)])
            gpu_final = projected_buffer[self.GPU_LUT]

            yield {
                "timestamp": hourly_slice.time.dt.strftime('%Y%m%d_%H').item(),
                "data": gpu_final.get(),
                "bounds": self.STATIC_METADATA["bounds"],
                "transform": self.STATIC_METADATA["transform"],
                "crs": self.STATIC_METADATA["crs"]
            }