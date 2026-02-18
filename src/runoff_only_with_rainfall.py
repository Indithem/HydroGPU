from argparse import ArgumentParser
from pathlib import Path

from downloads import dem, lulc, soil, mws
from algorithms import timeseries
from downloads import rainfall
import config as cfg
from utils import GeoTIFFHandler, make_logger

parser = ArgumentParser()
logger = make_logger("runoff_only_with_rainfall.log")

def modify_cfg(args):
    config = cfg
    config.BOUNDARY_GEOJSON_PATH = args.boundary
    if args.t:
        path_obj = Path(config.BOUNDARY_GEOJSON_PATH)
        new_path = path_obj.with_name(f"{path_obj.stem}_timeseries{path_obj.suffix}")
        config.TIMESERIES_VECTOR = new_path

    return config

def prereq():
    downloaders = [
        dem,
        soil,
        lulc,
    ]
    downloaders = [d.Downloader for d in downloaders]

    for downloader in downloaders:
        downloader.parse_args(parser)

    args = parser.parse_args()
    config = modify_cfg(args)

    for downloader in downloaders:
        downloader(args, logger, config).main()

if __name__=="__main__":
    parser.add_argument('-p', "--pre-req", action='store_true', help="also do pre-req stuff")
    parser.add_argument('-b', '--boundary', help="overwrite BOUNDARY_GEOJSON_PATH", default=cfg.BOUNDARY_GEOJSON_PATH)
    parser.add_argument('-t', help="Dump timeseries next to boundary file", action='store_true')
    parser.add_argument('--skip-rainfall', help="assume rainfall data is downloaded", action='store_true')
    rainfall.Downloader.parse_args(parser)

    if parser.parse_args().pre_req:
        prereq()

    args = parser.parse_args()
    config = modify_cfg(args)

    if not args.skip_rainfall:
        rainfall.Xarr(args, logger, config).main()

    tif_handler = GeoTIFFHandler(cfg.DEMFILE_PATH, logger)

    timeseries.TimeSeries(args, tif_handler, config=config).run()
