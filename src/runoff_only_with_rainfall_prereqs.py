from argparse import ArgumentParser
from downloads import dem, lulc, soil
from utils import make_logger

if __name__=="__main__":
    downloaders = [
        dem,
        soil,
        lulc,
    ]
    downloaders = [d.Downloader for d in downloaders]

    parser = ArgumentParser()
    for downloader in downloaders:
        downloader.parse_args(parser)

    logger = make_logger("runoff_only_with_rainfall_prereqs.log")
    args = parser.parse_args()

    for downloader in downloaders:
        downloader(args, logger).main()