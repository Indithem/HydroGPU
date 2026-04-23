# What this does
Produces runoff and rainfall timeseries over region for each microwatershed
in a region of interest.

## Inputs
- Geojson vector file of Microwatershed Boundary which is 
also the polygon boundary of region of interest. 
(an example is provided in `tifs/delhi.geojson`)
- Time period of interest, as start date and end date.

# Usage - First time initialization
## Authenticate to Google Earth Engine
Inorder to download rainfall data from Google Earth Engine, you need to authenticate your account. You can do this by running the following command in your terminal:
```sh
uv run src/authenticate.py
```
## Check configuration values
Check if the configuration values in `config/config.toml` are set correctly.

# Usage
Run from the root folder, the folder where this README.md is located, the following:
```sh
uv run src/runoff.py
```

```
usage: runoff.py [-h] [-p] [-b BOUNDARY] [-t] [--start START] [--end END]

options:
  -h, --help            show this help message and exit
  -p, --pre-req         also do pre-req stuff
  -b, --boundary BOUNDARY
                        use another boundary file
  -t                    Dump timeseries next to boundary file
  --start START         in YYYY-MM-DD format (inclusive)
  --end END             in YYYY-MM-DD format (exclusive)
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
