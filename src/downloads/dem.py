"""
Currently, this calculates Slope (gradient) on GEE Servers.
DEM is not downloaded.
"""
import xarray
import config as cfg
from downloads import GenericDownloader, ee
import config
import geemap

class Downloader(GenericDownloader):
    def __init__(self):
        """
        Override parent class's init. As GeoTiff handler needs one tif file for CRS reference
        We download DEM as this reference
        """
        pass

    def main(self):
        dataset = ee.Image('USGS/SRTMGL1_003')
        elevation = dataset.select('elevation')

        region = self.load_region()

        # 1. Calculate slope in degrees
        slope_deg = ee.Terrain.slope(elevation)

        # 2. Convert to Gradient: tan(slope * pi / 180)
        slope_gradient = slope_deg.multiply(3.141592).divide(180).tan()

        elevation_clip = slope_gradient.clipToBoundsAndScale(
            geometry=region,
            scale=cfg.GEE_SCALE
        )

        geemap.download_ee_image(
            elevation_clip,
            filename=cfg.DEMFILE_PATH,
            scale=cfg.GEE_SCALE,  # Adjust scale as needed
            region=region,
        )
