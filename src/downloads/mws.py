from downloads import GenericDownloader, ee
import json
import geopandas as gpd

class Clip(GenericDownloader):
    def main(self):
        # 1. Load the small boundary first
        mask = gpd.read_file(self.cfg.BOUNDARY_GEOJSON_PATH)

        # 2. Get the bounding box of the mask
        bbox = tuple(mask.total_bounds.tolist())  # returns [minx, miny, maxx, maxy]

        # 3. Read ONLY the features within that box from the 5GB file
        # This uses the spatial index of the file (if available) or streams efficiently
        gdf = gpd.read_file(self.cfg.PAN_INDIA_MWS, bbox=bbox)

        # 2. Ensure they use the same CRS (Coordinate Reference System)
        if gdf.crs != mask.crs:
            mask = mask.to_crs(gdf.crs)

        # 3. Clip the data
        clipped_gdf = gpd.clip(gdf, mask)
        return clipped_gdf