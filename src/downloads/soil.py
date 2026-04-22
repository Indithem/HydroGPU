from argparse import ArgumentParser
import config as cfg
import xarray
from downloads import GenericDownloader, ee
import geemap

class Downloader(GenericDownloader):
    def main(self):
        region = self.load_region()

        hsg_image = ee.Image('projects/ee-dharmisha-siddharth/assets/HYSOGs250m')

        hsg_clip = hsg_image.clipToBoundsAndScale(
            geometry=region,
            scale=cfg.GEE_SCALE
        )

        geemap.download_ee_image(
            hsg_clip,
            filename=cfg.SOIL_PATH,
            scale=cfg.GEE_SCALE,  # Adjust scale as needed
            region=region,
        )