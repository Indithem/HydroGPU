# What this does
Produces runoff and rainfall timeseries over region for each microwatershed
in a region of interest.

## Inputs
- Geojson polygon boundary of region of interest. (an example is provided in `tifs/region.geojson`)
- Geojson vector file of Microwatershed Boundary.
- Time period of interest, as start date and end date.

# Usage - First time initialization
## Authenticate to Google Earth Engine
Inorder to download rainfall data from Google Earth Engine, you need to authenticate your account. You can do this by running the following command in your terminal:
```sh
uv run src/authenticate.py
```
## Check configuration values
Check if the configuration values in `config/config.toml` are set correctly.

## One time inputs downloads
We need DEM file as input for this project. These files are fetched from GEE 
*but may have to be downloaded interactively, in case of errors*.

```sh
uv run src/runoff_only_with_rainfall_prereqs.py
```
Save it somewhere, probably as `tifs/dem.tiff`, using config from previous step.

If you want to test with other rasters you can skip this step, but ensure your
rasters are pointed by config.

# Usage
Run from the root folder, the folder where this README.md is located, the following:
```sh
uv run src/runoff_only_with_rainfall.py
```

```
usage: runoff_only_with_rainfall.py [-h] [--start START] [--end END] [--skip-gee]

options:
  -h, --help     show this help message and exit
  --start START  in YYYY-MM-DD format (inclusive)
  --end END      in YYYY-MM-DD format (exclusive)
  --skip-gee     assume rasters are already fetched to Google Drive
```

# Dependencies
- [uv](https://docs.astral.sh/uv/)
- Cuda. Ensure the version of cupy being used is compatable with your cuda version. For proper version, see [here](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi).
I had cuda13, so I installed `cupy-cuda13x`. To change this dependency, do:
```sh
uv remove cupy-cuda13x
uv add cupy-cudaYYx
```
where `YY` is your appropriate cuda version. Or, if you cannot figure out which version, 
as last resort,`uv add cupy` works, but needs to compile cuda from scratch.
