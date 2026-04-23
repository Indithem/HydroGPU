import shutil
from argparse import ArgumentParser
from pathlib import Path
import ee
from downloads import dem, lulc, soil
from algorithms import timeseries
from downloads import rainfall
import config as cfg
from utils import GeoTIFFHandler, make_logger
import utils

parser = ArgumentParser()
logger = make_logger("runoff_only_with_rainfall.log")

def modify_cfg(args):
    cfg.BOUNDARY_GEOJSON_PATH = args.boundary
    cfg.MICROWATERSHEDS_PATH = args.boundary
    if args.t:
        path_obj = Path(cfg.BOUNDARY_GEOJSON_PATH)
        new_path = path_obj.with_name(f"{path_obj.stem}_timeseries{path_obj.suffix}")
        cfg.TIMESERIES_VECTOR = new_path
    cfg.ARG_START_DATE = args.start
    cfg.ARG_END_DATE = args.end

def prereq():
    downloaders = [
        lulc.Downloader,
        soil.Downloader,
        rainfall.Download_to_database
    ]

    args = parser.parse_args()
    modify_cfg(args)

    for downloader in downloaders:
        downloader().main()

if __name__=="__main__":
    logger.info("Starting up")

    parser.add_argument('-p', "--pre-req", action='store_true', help="also do pre-req stuff")
    parser.add_argument('-b', '--boundary', help="use another boundary file", default=cfg.BOUNDARY_GEOJSON_PATH)
    parser.add_argument('-t', help="Dump timeseries next to boundary file", action='store_true')
    parser.add_argument('--start', help="in YYYY-MM-DD format (inclusive)", default=cfg.ARG_START_DATE)
    parser.add_argument('--end', help="in YYYY-MM-DD format (exclusive)", default=cfg.ARG_END_DATE)

    args = parser.parse_args()
    modify_cfg(args)

    if args.pre_req:
        shutil.rmtree(cfg.RAINFALL_FOLDER, ignore_errors=True)
        dem.Downloader().main()
        utils.tif_handler = GeoTIFFHandler(cfg.DEMFILE_PATH, logger)
        prereq()
    else:
        utils.tif_handler = GeoTIFFHandler(cfg.DEMFILE_PATH, logger)

    shutil.rmtree(cfg.RUNOFFS_FOLDER, ignore_errors=True)
    timeseries.TimeSeries().run()
    logger.info("Done")
