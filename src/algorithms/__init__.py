from typing import Dict
import config as cfg

from utils import GeoTIFFHandler


class GenericAlgorithm:
    def __init__(self, args, tif_handler: GeoTIFFHandler, *oth, config=cfg, **kwargs) -> None:
        self.cfg = config
        self.args = args
        self.tif_handler = tif_handler
        self.logger = tif_handler.logger

    def load_inputs(self):
        pass

    def main(self):
        pass

    def save_outputs(self):
        pass

    def run(self):
        self.logger.info("Loading inputs")
        self.load_inputs()
        self.logger.info("Starting algo")
        self.main()
        self.logger.info("Saving outputs")
        self.save_outputs()
