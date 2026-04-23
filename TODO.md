- [ ] Clip mws raster using AoI and a global raster.
- [x] Google earth engine doesn't properly clip.
- [ ] Migrate stream order from legacy code.
- [ ] Jump Martix.

- [ ] Currently, slope calculation in GEE engines, shift to local
- [x] Shift getDownloadUrl to xee in downloads/dem,lulc,soil.
- [x] Get per month summed rainfall in GEE only.

Current Implementation:
Completed in 45s

Summing up in GEE:
Lots of warnings such as 
```
WARNING:urllib3.connectionpool:Connection pool is full, discarding connection: earthengine.googleapis.com. Connection pool size: 10
WARNING:root:fast_time_slicing is enabled but ImageCollection images don't have IDs. Reverting to default behavior.
```
Completed in 1min27s

rainfall_collection = (
            ee.ImageCollection("NASA/GPM_L3/IMERG_DAILY_V06")
            .filterDate(cfg.ARG_START_DATE, cfg.ARG_END_DATE)
            .select('total_accum')
        )
took 30s
 
- [ ] Cleanup unused config and functions and classes.
There are some in utils and */__init__.py