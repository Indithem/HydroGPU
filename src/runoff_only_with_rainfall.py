from argparse import ArgumentParser

from algorithms import timeseries
from downloads import rainfall
import config as cfg
from utils import GeoTIFFHandler, make_logger

if __name__=="__main__":
    parser = ArgumentParser()
    rainfall.Downloader.parse_args(parser)

    args = parser.parse_args()
    logger = make_logger("runoff_only_with_rainfall.log")

    rainfall.Downloader(args, logger).main()

    tif_handler = GeoTIFFHandler(cfg.DEMFILE_PATH, logger)

    timeseries.TimeSeries(args, tif_handler).run()
