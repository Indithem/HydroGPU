from argparse import ArgumentParser

from downloads import GenericDownloader, ee
import json
import requests
from downloads.rainfall import DownloaderBase as RainfallDownloader

class DynamicWorld(GenericDownloader):
    """
    Currently, the LULC is mode of lulc's found in past 2 months from end date. Static
    """
    def main(self):
        # with open(self.cfg.BOUNDARY_GEOJSON_PATH) as f:
        #     geojson = json.load(f)
        #     # region = ee.Feature(geojson['geometry'])
        #     region = ee.Geometry.Polygon(geojson['geometry']['coordinates'])

        region = self.load_region()

        end_date = ee.Date(self.cfg.ARG_END_DATE)
        start_date = ee.Date(self.cfg.ARG_START_DATE)

        # start = end_date.advance(-2, 'month')
        start = start_date
        end = end_date

        # --- Dynamic World ---
        dw_col = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                  .filterDate(start, end)
                  .filterBounds(region)
                  .select('label'))

        # Convert collection → single image (most frequent class per pixel)
        dw_image = dw_col.reduce(ee.Reducer.mode()).rename('lulc')

        # --- Clip & scale like your elevation example ---
        dw_clip = dw_image.clipToBoundsAndScale(
            geometry=region,
            scale=self.cfg.GEE_SCALE  # e.g., 10 for native DW resolution
        )

        # --- Get download URL ---
        url = dw_clip.getDownloadURL({
            'format': 'GEO_TIFF'
        })

        self.logger.info('Download URL:' + url)

        response = requests.get(url)

        with open(self.cfg.LULC_PATH, 'wb') as f:
            f.write(response.content)

# I had teseted different sources. For now, lulc is taken from DynamicWorldv1
Downloader = DynamicWorld

class Corestack(GenericDownloader):
    def main(self):
        dataset = ee.Image('projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/pan_india_lulc_v3_2024_2025')
        band = dataset.select(0)

        # with open(self.cfg.BOUNDARY_GEOJSON_PATH) as f:
        #     geojson = json.load(f)
        #     # region = ee.Feature(geojson['geometry'])
        #     region = ee.Geometry.Polygon(geojson['geometry']['coordinates'])

        region = self.load_region()

        elevation_clip = band.clipToBoundsAndScale(
            geometry=region,
            scale=self.cfg.GEE_SCALE
        )

        url = elevation_clip.getDownloadURL({
            'format': 'GEO_TIFF'
        })

        self.logger.info('Download URL:' + url)

        response = requests.get(url)

        with open(self.cfg.LULC_PATH, 'wb') as f:
            f.write(response.content)
