import cupy as cp
from cupyx.scipy.ndimage import binary_dilation
from utils import load_tif_image, make_logger
import numpy as np
import json

logger = make_logger("logs/source_mws.log")

# load rasters as arrays (via rasterio or GDAL)
ws = cp.asarray(load_tif_image("../tifs/lgb/mwshed.tif"))
flow = cp.asarray(load_tif_image("../tifs/lgb/flow_dir.tif"))

logger.info("loaded arrays")

# offsets for D8 directions (ESRI convention)
offsets = {
    1: (0, 1),   # E
    2: (1, 1),   # SE
    4: (1, 0),   # S
    8: (1, -1),  # SW
    16: (0, -1), # W
    32: (-1, -1),# NW
    64: (-1, 0), # N
    128:(-1, 1)  # NE
}

H, W = ws.shape
mark = cp.ones_like(ws, dtype=bool)

for code, (dy, dx) in offsets.items():
    mask = flow == code
    y, x = cp.nonzero(mask)
    ny, nx = y + dy, x + dx
    valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    mark[ny[valid], nx[valid]] = False

logger.info("done masking")

# boundary = where mark==True
# dilation: identify border pixels of each watershed

borders = binary_dilation(ws) != ws
logger.info("found borders")

boundary_ids = ws[borders & mark]
interior_ids = ws[borders & ~mark]
logger.info("done border masks")


# IDs with no false boundaries
valid_ids = cp.unique(boundary_ids)
invalid_ids = cp.unique(interior_ids)
final_ids = cp.setdiff1d(valid_ids, invalid_ids)

logger.info("done, saving files")

# transfer to host and store
# np.save("final_ids.npy", cp.asnumpy(final_ids))
# cp.asnumpy(final_ids).tofile("final_ids.txt", sep="\n", format="%d")

# Got 0 final_ids
# debugging why:


ids = cp.unique(ws)
ids = ids[ids != 0]  # skip background if any
ids = ids.astype(cp.int32)


borders_mask = binary_dilation(ws) != ws
border_ws = ws[borders_mask]
border_mark = mark[borders_mask]

border_ws = border_ws.astype(cp.int32)
border_mark = border_mark.astype(cp.int32)  # weights must also be int/float


# total border count per watershed
totals = cp.bincount(border_ws)
# matched border count (where mark=True)
matched = cp.bincount(border_ws, weights=border_mark.astype(cp.int32))

# avoid divide-by-zero
pct = cp.where(totals > 0, matched / totals * 100, 0)

# map from watershed id -> percent
percentages = dict(zip(cp.asnumpy(ids), cp.asnumpy(pct[ids])))

# convert keys and values to host types
percentages_cpu = {int(k): float(v) for k, v in percentages.items()}

percentages_cpu = sorted(percentages_cpu.items(), key=lambda x: x[1], reverse=True)

with open("percentages.json", "w") as f:
    json.dump(percentages_cpu, f)
