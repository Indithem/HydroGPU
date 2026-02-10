from argparse import ArgumentParser

from downloads import GenericDownloader, ee
import json
import requests
from downloads.rainfall import Downloader as RainfallDownloader

# '''
# Function definition to combine L1, L2, and L3 outputs of a given region
# '''
# def combine_LULC_images(L1_assetpath, L2_assetpath, L3_assetpath, roi_boundary):
#   L1_output = ee.Image(L1_assetpath)
#   L2_output = ee.Image(L2_assetpath)
#   L3_output = ee.Image(L3_assetpath) #.select(['class','confidence_L3']).rename(['Predicted_L3_Label','confidence_L3'])
#
#   greenery_mask_ = L1_output.select('Predicted_L1_Label').expression("b('Predicted_L1_Label') == 1").clip(roi_boundary)
#   crop_mask = L2_output.select('Predicted_L2_Label').expression("b('Predicted_L2_Label') == 5").And(L1_output.select('Predicted_L1_Label').expression("b('Predicted_L1_Label') == 1")).clip(roi_boundary)
#   final_classification_output = L1_output.select('Predicted_L1_Label').where(greenery_mask_, L2_output.select('Predicted_L2_Label')).where(crop_mask, L3_output.select('Predicted_L3_Label')).rename('class').clip(roi_boundary)
#   final_confidence_output = L1_output.select('confidence_L1').where(greenery_mask_, L2_output.select('confidence_L2')).where(crop_mask, L3_output.select('confidence_L3')).rename('confidence').clip(roi_boundary)
#
#   final_output = final_classification_output.addBands(final_confidence_output)
#   return final_output


class Downloader(GenericDownloader):
    def main(self):
        with open(self.cfg.BOUNDARY_GEOJSON_PATH) as f:
            geojson = json.load(f)
            # region = ee.Feature(geojson['geometry'])
            region = ee.Geometry.Polygon(geojson['geometry']['coordinates'])

        # Load hydrologic soil group dataset
        hsg_image = ee.Image('projects/sat-io/open-datasets/HiHydroSoilv2_0/Hydrologic_Soil_Group_250m')

        # Clip to your region and scale
        hsg_clip = hsg_image.clipToBoundsAndScale(
            geometry=region,
            scale=self.cfg.GEE_SCALE  # likely 250m
        )

        # Get download URL (GeoTIFF)
        url = hsg_clip.getDownloadURL({
            'format': 'GEO_TIFF'
        })

        self.logger.info('Download URL:' + url)

        response = requests.get(url)

        with open(self.cfg.SOIL_PATH, 'wb') as f:
            f.write(response.content)