from downloads import GenericDownloader, ee
import json

class Downloader(GenericDownloader):
    def main(self):
        dataset = ee.Image('projects/corestack-datasets/assets/datasets/India_mws_uid_area_gt_500')
        elevation = dataset.select('elevation')

        with open(self.cfg.BOUNDARY_GEOJSON_PATH) as f:
            geojson = json.load(f)
            # region = ee.Feature(geojson['geometry'])
            region = ee.Geometry.Polygon(geojson['geometry']['coordinates'])

        elevation_clip = elevation.clipToBoundsAndScale(
            geometry=region,
            scale=self.cfg.GEE_SCALE
        )

        url = elevation_clip.getDownloadURL({
            'format': 'GEO_TIFF'
        })

        self.logger.info('Download URL:' + url)

        response = requests.get(url)

        with open(self.cfg.DEMFILE_PATH, 'wb') as f:
            f.write(response.content)