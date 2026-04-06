import gc
import json
import os
import shutil
import time
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import xarray as xr
import requests
import concurrent.futures
from rasterio.enums import Resampling
from shapely.geometry import shape
from rasterio.transform import from_origin
from tqdm import tqdm
# import geedim as gd
import cupy as cp
import numpy as np
import cucim.skimage.transform as cimg

from downloads import GenericDownloader, ee, Logger

class Downloader(GenericDownloader):
    added_args = False

    def main(self):
        # with open(self.cfg.BOUNDARY_GEOJSON_PATH) as f:
        #     geojson = json.load(f)
        #     region = ee.Geometry.Polygon(geojson['geometry']['coordinates'])

        region = self.load_region()
        rainfall_collection = (
            ee.ImageCollection('JAXA/GPM_L3/GSMaP/v6/operational')
            .filterBounds(region)
            .filterDate(self.args.start, self.args.end)
            .select('hourlyPrecipRate')
        )

        self.logger.info(f'Total Images in Collection: {rainfall_collection.size().getInfo()}')

        images = rainfall_collection.toList(rainfall_collection.size())
        n = images.size().getInfo()

        tasks: list[ee.batch.Task] = []

        folder_name = self.cfg.GOOGLEDRIVE_RAINFALL_FOLDER
        query = f"title = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"

        if not self.args.skip_gee:
            for folder in self.drive.ListFile({'q': query}).GetList():
                self.drive.CreateFile({'id': folder['id']}).Trash()

            for i in tqdm(range(n)):
                img = ee.Image(images.get(i))
                date = ee.Date(img.get('system:time_start')).format('YYYYMMdd_HH')
                filename_pref = 'rainfall_' + date.getInfo()
                task = ee.batch.Export.image.toDrive(
                    image=img.clip(region),
                    description='rainfall_' + date.getInfo(),
                    folder=self.cfg.GOOGLEDRIVE_RAINFALL_FOLDER,
                    fileNamePrefix=filename_pref,
                    region=region,
                    scale=self.cfg.GEE_SCALE,  # GSMaP native ~10 km
                    # crs='EPSG:4326',
                    # maxPixels=1e13
                )
                task.start()
                tasks += [task]

            for task in tqdm(tasks):
                while task.active():
                    time.sleep(30)

        self.empty_folder(self.cfg.RAINFALL_FOLDER)

        folders = self.drive.ListFile({'q': query}).GetList()

        assert len(folders) == 1

        folder_id = folders[0]['id']
        query = f"'{folder_id}' in parents and trashed=false"
        files_in_folder = self.drive.ListFile({'q': query}).GetList()
        file_ids = [f['id'] for f in files_in_folder]

        with ThreadPoolExecutor() as executor:
            list(executor.map(self.download_gdrive_file, file_ids, [self.cfg.RAINFALL_FOLDER]*len(file_ids)))

    def parse_args(parser: ArgumentParser):
        if not Downloader.added_args:
            parser.add_argument('--start', help="in YYYY-MM-DD format (inclusive)", default='2023-07-01', type=ee.Date)
            parser.add_argument('--end', help="in YYYY-MM-DD format (exclusive)", default='2023-07-03', type=ee.Date)
            parser.add_argument('--skip-gee', help="assume rasters are already fetched to Google Drive", default=False,
                                action='store_true')
            Downloader.added_args = True

class Downloader2(Downloader):
    """
    afaik, Bypasses google drive and directly downloads to folder

    Doesn't work if file size >50MB
    """
    def main(self):
        region = self.load_region()
        rainfall_collection = (
            ee.ImageCollection('JAXA/GPM_L3/GSMaP/v6/operational')
            .filterBounds(region)
            .filterDate(self.args.start, self.args.end)
            .select('hourlyPrecipRate')
        )

        # FIX: Map properties to a Feature properly
        features = rainfall_collection.map(lambda img: ee.Feature(None, {
            'id': img.id(),
            'date': img.date().format('YYYYMMdd_HH')
        }))

        # 2. FIX: Get the actual list of property dictionaries
        # We use .list() on the collection then getInfo to bring the data to Python
        img_infos = [f['properties'] for f in features.getInfo()['features']]

        self.logger.info(f"Sample Feature: {features.first().getInfo()}")
        self.logger.info(f'Total Images in Collection: {rainfall_collection.size().getInfo()}')
        self.logger.info(f'Total Images to process: {len(img_infos)}')

        def download_worker(info):
            # info is {'date': '20230702_22', 'id': '20230702_2200'}
            leaf_id = info['id']
            date_str = info['date']
            filename = f"rainfall_{date_str}.tif"

            # FIX: Prepend the parent collection path
            parent_path = 'JAXA/GPM_L3/GSMaP/v6/operational'
            full_asset_id = f"{parent_path}/{leaf_id}"

            try:
                # Now GEE will find the asset in the public catalog
                img = ee.Image(full_asset_id).clip(region)

                url = img.getDownloadURL({
                    'scale': 30,
                    'region': region,
                    'format': 'GEO_TIFF',
                    'crs': 'EPSG:4326'
                })

                res = requests.get(url)
                if res.status_code == 200:
                    path = f"{self.cfg.RAINFALL_FOLDER}/{filename}"
                    with open(path, 'wb') as f:
                        f.write(res.content)
                    return True
            except Exception as e:
                self.logger.error(f"Error downloading {filename}: {e}")
                return False

        # FIX 2: Use ThreadPoolExecutor for parallel downloads
        # This saturates your bandwidth and bypasses the slow Drive export queue
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(tqdm(executor.map(download_worker, img_infos), total=len(img_infos)))

class Xarr(Downloader):
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
            .filterDate(self.args.start, self.args.end)
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

        target_scale_meters = self.cfg.GEE_SCALE  # 30
        deg_per_meter = 1 / 111319.49
        target_res_degrees = target_scale_meters * deg_per_meter

        # 1. Calculate dimensions
        y_size, x_size = da.y.size, da.x.size
        total_pixels = y_size * x_size

        # 2. Create the Index Map
        # The 'sink_index' is the very last position (index = total_pixels)
        source_indices = np.arange(total_pixels).reshape(y_size, x_size).astype(np.int32)

        # 3. Prepare rioxarray with a specific nodata pointer
        dummy_da = da.isel(time=0).copy(data=source_indices)
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
            "shape": mapped_indices_da.shape
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
            # Initialize the accumulator on the GPU with the shape of your LUT
            # This matches the shape of the clipped watershed
            gpu_sum = cp.zeros(gpu_lut.shape, dtype=cp.float32)

            # Pre-fetch the data to minimize xarray overhead in the loop
            # If the dataset fits in RAM, da.values is faster than repeated .isel()
            raw_data = da.values

            for i in range(raw_data.shape[0]):
                gpu_src_flat = cp.concatenate([
                    cp.asarray(raw_data[i]).ravel(),
                    cp.array([0], dtype=raw_data[i].dtype)
                ])

                # 2. Apply LUT and add to total (In-place)
                # We use .ravel() to treat the source as 1D for the index lookup
                gpu_sum += gpu_src_flat[gpu_lut]

            # self.tif_loader.save_tiff(gpu_sum.get(),
            #     f"{self.cfg.RAINFALL_FOLDER}/rainfall_{da.time[0].dt.strftime('%Y%m%d_%H').item()}.tif)")

            return {
                "timestamp": da.isel(time=0).time.dt.strftime('%Y%m%d_%H').item(),
                "data": gpu_sum.get(),  # Bring back to CPU for output
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


# class Geedim(Downloader):
#     def main(self):
#         self.logger.info("Starting rainfall download")
#
#         # 1. Convert dict/GeoJSON to EE Geometry
#         region = self.load_region()
#
#         # 2. Add the buffer (12km is safe for 11km pixels)
#         # This ensures you don't get "cut off" edges
#         buffered_region = region.buffer(12000).bounds()
#
#         clipping_geom = shape(region.getInfo())
#
#         rainfall_collection = (
#             ee.ImageCollection('JAXA/GPM_L3/GSMaP/v6/operational')
#             # .filterBounds(region)
#             .filterDate(self.args.start, self.args.end)
#             .select('hourlyPrecipRate')
#         )
#
#         first_img = rainfall_collection.first()
#         native_proj = first_img.projection()
#
#         gd.Initialize(
#             project=self.cfg.GEE_PROJECT_NAME,
#             opt_url='https://earthengine-highvolume.googleapis.com'
#         )
#
#         prep_coll = rainfall_collection.gd.prepareForExport(
#             region=buffered_region,
#             scale=0.1,
#             crs=native_proj.crs().getInfo(),
#             dtype='float32'  # Match GSMaP data type
#         )
#
#         prep_coll.gd.toGeoTIFF('download_folder', split='images')