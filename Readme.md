# HydroGPU

This repository contains usable and runnable source code for [GPU-Accelerated Hydrology Algorithms for On-Prem Computation](https://dl.acm.org/doi/10.1145/3759536.3763805)

This is a continuation of [RahulkumarRV/HydroGPU](https://github.com/RahulkumarRV/HydroGPU).


# Dependencies Installation / Environment setup

Please refer to 
[doc/HydroGPU Project documentation.pdf](./doc/HydroGPU%20Project%20documentation.pdf).
This is the documentation on environment setup from previous project. 
Essentailly, have conda installed and create environment using the following commands:

```sh
conda create -n my_rapids_env -c rapidsai -c conda-forge -c defaults python=3.10 cudatoolkit=11.8
conda activate my_rapids_env
```

and then install all dependent python packages as documented above:
```sh
conda install rasterio matplotlib numpy pandas geopandas gdal earthengine-ap 
```

We also depend on other python packages such as natsort, install them using pip:
```sh
pip install natsort
```

We have also uploaded out [pip freeze](./doc/pip_freeze.txt), incase we did not specify any dependencies anywhere.

# Usage

This flow chart should give a general idea of what we are trying to achieve:
![Flow chart of workflow](./doc/workflow.svg)

The intended, or our use, of this repository is the following. This entire repository is root of
some folder, lets call `src`. At the same level as `src`, we also have `tifs`. 
We put all .tif raster files in `tifs` folder while invoking these python scripts from `src` folder.
Which is why the files are hardcoded with "../tifs/" prefix.

For each file, you can see the input and output tif file locations are hardcoded.
At each stage, we produce new .tif files from other .tif files as inputs. 

File names such as `compute_legacy_*.py` correspond to the source code files which were developed
in [previous project](https://github.com/RahulkumarRV/HydroGPU). We didnot edit these files.
We only used the outputs from these files in our project, and developed newer algorithms.
!todo: We will edit these files to be better aligned with rest of other algorithms.






