import gc
import json
import os
import pathlib
import shutil
import time
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

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
        self.zarr_lock = Lock()
        self.zarr_path = os.path.join(self.cfg.RAINFALL_FOLDER, "rainfall_archive.zarr")

    def save_geozarr(self, flat_data, timestamp, dummy_da):
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

        ds = da.to_dataset()

        # To ensure no race conditions during writes to zarr, a lock is used
        # We can potentially increase performance here
        # you can initialize an empty Zarr store and have threads write to specific "regions" without appending
        with self.zarr_lock:
            if not os.path.exists(self.zarr_path):
                encoding = {
                    "precipitation": {"chunks": (1, native_y, native_x)},
                    "time": {
                        "units": "hours since 2017-07-01 00:00:00",
                        "calendar": "proleptic_gregorian",
                        "dtype": "int64"  # Ensuring integer storage for hours
                    }
                }
                ds.to_zarr(self.zarr_path, mode='w', encoding=encoding, zarr_format=2)
            else:
                ds.to_zarr(self.zarr_path, mode='a', append_dim='time', zarr_format=2)

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
    


class Xarr(DownloaderBase):
    """
    Ignore this class. This used to help when "streaming" rainfall data.
    Currently, in one instance we download data and then in another we calculate runoff data.
    """

    def main(self):
        self.logger.info("Starting rainfall download")

        # 1. Convert dict/GeoJSON to EE Geometry
        region = self.load_region()

        # 2. Add the buffer (12km is safe for 11km pixels)
        # This ensures you don't get "cut off" edges
        buffered_region = region.buffer(12000).bounds()

        clipping_geom = shape(region.getInfo())

        rainfall_collection = (
            ee.ImageCollection('JAXA/GPM_L3/GSMaP/v6/operational')
            # .filterBounds(region)
            .filterDate(self.cfg.ARG_START_DATE, self.cfg.ARG_END_DATE)
            .select('hourlyPrecipRate')
        )

        first_img = rainfall_collection.first()
        native_proj = first_img.projection()

        # 2. Open the dataset using that EXACT projection
        ds = xr.open_dataset(
            rainfall_collection,
            engine='ee',
            # chunks={'time': 48},
            projection=native_proj,  # <--- CRITICAL: Matches original GEE alignment
            geometry=buffered_region,  # <--- Ensures the extent covers your region
            # scale=0.1             # Remove this; projection already contains the 0.1 scale
            fast_time_slicing = True,
        )

        # Expected for a 1-degree box at 0.1 scale: {'time': X, 'lat': 10, 'lon': 10}
        # print(f"New Dimensions: {ds.sizes}")
        # return

        # Prepare the whole DataArray once
        da = ds['hourlyPrecipRate'].rename({'lat': 'y', 'lon': 'x'})
        da = da.transpose("time", "y", "x")

        # Convert GEE resolution (30m) to decimal degrees for EPSG:4326
        # 111,319.49m is the approximate length of 1 degree at the equator
        target_scale_meters = self.cfg.GEE_SCALE  # 30
        deg_per_meter = 1 / 111319.49
        target_res_degrees = target_scale_meters * deg_per_meter

        # 1. Calculate dimensions
        y_size, x_size = da.y.size, da.x.size
        total_pixels = y_size * x_size

        # 2. Create the Index Map
        # The 'sink_index' is the very last position (index = total_pixels)
        source_indices = np.arange(total_pixels).reshape(y_size, x_size).astype(np.int32)

        self.dummy_da = da.isel(time=0)
        pathlib.Path(self.cfg.RAINFALL_FOLDER).mkdir(parents=True, exist_ok=True)


        # 3. Prepare rioxarray with a specific nodata pointer
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
        GPU_LUT = cp.asarray(mapped_indices_da.values)

        # Move the LUT to GPU
        # 4. Extract static metadata and move LUT to GPU
        STATIC_METADATA = {
            "bounds": mapped_indices_da.rio.bounds(),
            "transform": mapped_indices_da.rio.transform(),
            "crs": mapped_indices_da.rio.crs,
            "shape": mapped_indices_da.shape,
            "original_size": total_pixels
        }

        # self.empty_folder(self.cfg.RAINFALL_FOLDER)

        self.logger.info("ready to download rainfall")
        def process_hour(i, da, clipping_geom, target_res):
            """
            Worker function to process a single time slice.
            """
            # 1. Slice and prepare metadata
            hourly_slice = da.isel(time=i)

            # Ensure standard spatial names for rioxarray
            hourly_slice.rio.write_crs("EPSG:4326", inplace=True)
            hourly_slice.rio.write_transform(inplace=True)

            # 2. Upscale/Reproject to 30m (Nearest Neighbor for 'copying' blocks)
            hourly_slice_30m = hourly_slice.rio.reproject(
                dst_crs="EPSG:4326",
                resolution=target_res,
                resampling=Resampling.nearest
            )

            # 3. Clip to the specific watershed geometry
            final_raster = hourly_slice_30m.rio.clip(
                [clipping_geom],
                crs="EPSG:4326",
                drop=True,
                all_touched=True
            )

            # 4. Generate filename and save
            # timestamp = final_raster.time.dt.strftime('%Y%m%d_%H').item()
            # filename = f"{self.cfg.RAINFALL_FOLDER}/rainfall_{timestamp}.tif"
            #
            # # This line triggers the actual computation/download
            # final_raster.rio.to_raster(filename, dtype="float32")


            return {
                "timestamp": final_raster.time.dt.strftime('%Y%m%d_%H').item(),
                "data": final_raster.values.squeeze(),
                "bounds": final_raster.rio.bounds(),
                "transform": final_raster.rio.transform(),
                "crs": final_raster.rio.crs
            }

        def process_hour_gpu(i, da, gpu_lut, static_meta):
            """
            Lightning fast reprojection using GPU Look-Up Table.
            """
            # 1. Get the raw CPU slice
            # We use .values to avoid xarray overhead; ensure it matches the LUT shape
            hourly_slice_cpu = da.isel(time=i).values

            # 2. Transfer to GPU
            gpu_src = cp.asarray(hourly_slice_cpu)

            # 3. Apply LUT (Fancy Indexing)
            # .ravel() treats the source as a 1D array so the LUT can pick indices directly
            gpu_final = gpu_src.ravel()[gpu_lut]

            # 4. Return results (Bringing data back to CPU for the dictionary)
            return {
                "timestamp": da.isel(time=i).time.dt.strftime('%Y%m%d_%H').item(),
                "data": gpu_final,
                "bounds": static_meta["bounds"],
                "transform": static_meta["transform"],
                "crs": static_meta["crs"]
            }

        def process_hour_gpu_sum(da, gpu_lut, static_meta):
            """
            Computes the sum of all reprojected slices entirely on GPU.
            """
            # Pre-fetch the data to minimize xarray overhead in the loop
            # If the dataset fits in RAM, da.values is faster than repeated .isel()
            raw_data = da.values
            gpu_sum = cp.zeros(static_meta['original_size'], dtype=cp.float32)

            for i in range(raw_data.shape[0]):
                # gpu_src_flat = cp.concatenate([
                #     cp.asarray(raw_data[i]).ravel(),
                #     cp.array([0], dtype=raw_data[i].dtype)  # at the last index(=total_pixels) LUT looks up this value
                # ])

                # 2. Apply LUT and add to total (In-place)
                # We use .ravel() to treat the source as 1D for the index lookup
                # gpu_sum += gpu_src_flat[gpu_lut]
                gpu_sum += cp.asarray(raw_data[i]).ravel()

            time_stamp = da.isel(time=0).time.dt.strftime('%Y%m%d_%H').item()
            self.save_geozarr(gpu_sum, time_stamp)

            projected = cp.concatenate([
                gpu_sum,
                cp.array([0], dtype=raw_data[0].dtype)  # at the last index(=total_pixels) LUT looks up this value
            ])
            projected = projected[gpu_lut]

            # self.tif_loader.save_tiff(gpu_sum.get(),
            #     f"{self.cfg.RAINFALL_FOLDER}/rainfall_{da.time[0].dt.strftime('%Y%m%d_%H').item()}.tif)")

            return {
                "timestamp": time_stamp,
                "data": projected.get(),  # Bring back to CPU for output
                "bounds": static_meta["bounds"],
                "transform": static_meta["transform"],
                "crs": static_meta["crs"]
            }

        K = 24
        def process_K(i, da, clipping_geom, target_res):
            """
            Worker function to process a single time slice.
            """
            # 1. Slice and prepare metadata
            hourly_slice = da.isel(time=range(i, i+K))

            # Ensure standard spatial names for rioxarray
            hourly_slice.rio.write_crs("EPSG:4326", inplace=True)
            hourly_slice.rio.write_transform(inplace=True)

            # 2. Upscale/Reproject to 30m (Nearest Neighbor for 'copying' blocks)
            hourly_slice_30m = hourly_slice.rio.reproject(
                dst_crs="EPSG:4326",
                resolution=target_res,
                resampling=Resampling.bilinear
            )

            # 3. Clip to the specific watershed geometry
            final_raster = hourly_slice_30m.rio.clip(
                [clipping_geom],
                crs="EPSG:4326",
                drop=True,
                all_touched=True
            )

            # Sum across the time dimension to produce a daily total
            daily_sum = final_raster.sum(dim="time", skipna=True)

            # Ensure appropriate dtype and shape
            daily_sum = daily_sum.astype("float32").squeeze()

            # Build a day-level timestamp (use the first hour of the block)
            timestamp = hourly_slice.time[0].dt.strftime("%Y%m%d_%H").item()

            return {
                "timestamp": timestamp,
                "data": daily_sum.values,
                "bounds": daily_sum.rio.bounds(),
                "transform": daily_sum.rio.transform(),
                "crs": daily_sum.rio.crs
            }

        WOKRERS = 20
        ASK_BUFF = 16
        N = len(da.time)
        # times = da.time.values
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=WOKRERS) as executor,
            tqdm(range(N), desc="Exporting Rainfall") as pbar
        ):
            futures = deque()

            while pbar.n < N:
                if len(futures)==0:
                    gc.collect()
                    t = pbar.n
                    # da_big_slice = da.isel(time=range(t, min(t+ASK_BUFF*K, N)))
                    for i in range(ASK_BUFF):
                        if t+i*K >= N:
                            break
                        # start_local = i * K
                        # end_local = min((i + 1) * K, da_big_slice.time.size)
                        # it = slice(start_local, end_local)
                        # da_slice = da_big_slice.isel(time=it)

                        da_slice = da.isel(time=slice(t+i*K, min(t+(i+1)*K, N)))

                        futures.append(
                            executor.submit(
                                process_hour_gpu_sum,
                                da_slice,
                                GPU_LUT,
                                STATIC_METADATA,
                            )
                        )

                    # it = range(t, min(t+ASK_BUFF*K, N))
                    # da_slice = da.isel(time=it)
                    # for i in it:
                    #     futures.append(
                    #         executor.submit(
                    #             process_hour,
                    #             i-t,
                    #             da_slice,
                    #             clipping_geom,
                    #             target_res_degrees,
                    #         )
                    #     )

                res = futures.popleft().result()
                pbar.update(K)
                yield res

                # res = futures.popleft().result()
                # res["data"] = cp.asarray(res["data"])
                # pbar.update(1)
                # for _ in range(K-1):
                #     if len(futures)==0:
                #         # happens when date difference is not fortnightly
                #         assert pbar.n == N
                #         break
                #     res2 = futures.popleft().result()
                #     res["data"] += cp.asarray(res2["data"])
                #     pbar.update(1)
                # res["data"] = res["data"].get()
                # yield res

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
            .filterDate(self.cfg.ARG_START_DATE, self.cfg.ARG_END_DATE)
            .select('hourlyPrecipRate')
        )

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

        ASK_BUFF = 50
        WORKERS = 20

        def process_slice(t):
            da_slice = da.isel(time=slice(t, min(t + K, N)))

            # Extract raw data and sum on GPU
            raw_data = da_slice.values
            gpu_sum = cp.zeros(total_pixels, dtype=cp.float32)

            for i in range(raw_data.shape[0]):
                gpu_sum += cp.asarray(raw_data[i]).ravel()

            timestamp = da_slice.time[0].dt.strftime('%Y%m%d_%H').item()

            # Save the flat sum to the database
            # Assuming self.save_geozarr handles time-indexing inside the Zarr
            self.save_geozarr(gpu_sum.get(), timestamp, dummy_da)

        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor,
            tqdm(range(N), desc="Downloading Rainfall") as pbar
        ):
            futures = []

            while pbar.n < pbar.total:
                if len(futures) == 0:
                    gc.collect()
                    t = pbar.n
                    for i in range(ASK_BUFF):
                        if (t+i*K) >= N:
                            break
                        futures.append(executor.submit(
                            process_slice,
                            t+i*K
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
            .filterDate(self.cfg.ARG_START_DATE, self.cfg.ARG_END_DATE)
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
        target_scale_meters = self.cfg.GEE_SCALE  # 30
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