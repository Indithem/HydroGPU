"""
An example of how to run different algorithms,
a generic runner, a scratch pad to test.
"""

from algorithms import flows, microwatershed, runoff, timeseries
import config as cfg
from utils import GeoTIFFHandler, make_logger

if __name__=="__main__":
    logger = make_logger("run.log")
    tif_handler = GeoTIFFHandler(cfg.DEMFILE_PATH, logger)
    args = None

    # flows.FlowDirection(args, tif_handler).run()
    # flows.FlowAccumulation(args, tif_handler).run()
    # microwatershed.Microwatersheds(args, tif_handler, 10).run()

    # for _ in runoff.Runoff(args, tif_handler).main():
    #     pass

    timeseries.TimeSeries(args, tif_handler).run()
