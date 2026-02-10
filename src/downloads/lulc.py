from argparse import ArgumentParser

from downloads import GenericDownloader, ee
import json
import requests
from downloads.rainfall import Downloader as RainfallDownloader

class DynamicWorld(GenericDownloader):
    def main(self):
        with open(self.cfg.BOUNDARY_GEOJSON_PATH) as f:
            geojson = json.load(f)
            # region = ee.Feature(geojson['geometry'])
            region = ee.Geometry.Polygon(geojson['geometry']['coordinates'])

        start = self.args.end.advance(-30, 'day')
        end = self.args.end

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

    def parse_args(parser: ArgumentParser):
        return RainfallDownloader.parse_args(parser)

Downloader = DynamicWorld

class Corestack(GenericDownloader):
    def main(self):
        dataset = ee.Image('projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/pan_india_lulc_v3_2024_2025')
        band = dataset.select(0)

        with open(self.cfg.BOUNDARY_GEOJSON_PATH) as f:
            geojson = json.load(f)
            # region = ee.Feature(geojson['geometry'])
            region = ee.Geometry.Polygon(geojson['geometry']['coordinates'])

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
