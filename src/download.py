from argparse import ArgumentParser

from downloads import rainfall, dem, lulc, soil
from utils import make_logger

if __name__=="__main__":
    downloaders = [
        # soil,
        # lulc,
        # rainfall
    ]
    downloaders = [d.Downloader for d in downloaders] + [rainfall.Geedim]

    parser = ArgumentParser()
    for downloader in downloaders:
        downloader.parse_args(parser)
    logger = make_logger("download.log")
    args = parser.parse_args()

    for downloader in downloaders:
        downloader(args, logger).main()