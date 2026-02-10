import json
import os
import shutil
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from downloads import GenericDownloader, ee, Logger

class Downloader(GenericDownloader):
    def main(self):
        with open(self.cfg.BOUNDARY_GEOJSON_PATH) as f:
            geojson = json.load(f)
            region = ee.Geometry.Polygon(geojson['geometry']['coordinates'])

        rainfall_collection = (
            ee.ImageCollection('JAXA/GPM_L3/GSMaP/v6/operational')
            .filterBounds(region)
            .filterDate(self.args.start, self.args.end)
            .select('hourlyPrecipRate')
        )

        self.logger.info(f'Total Images in Collection: {rainfall_collection.size().getInfo()}')

        images = rainfall_collection.toList(rainfall_collection.size())
        n = images.size().getInfo()

        tasks: list[ee.batch.Task] = []
        file_names: list[str] = []

        if not self.args.skip_gee:
            for i in tqdm(range(n)):
                img = ee.Image(images.get(i))
                date = ee.Date(img.get('system:time_start')).format('YYYYMMdd_HH')
                filename_pref = 'rainfall_' + date.getInfo()
                task = ee.batch.Export.image.toDrive(
                    image=img.clip(region),
                    description='rainfall_' + date.getInfo(),
                    folder=self.cfg.GOOGLEDRIVE_RAINFALL_FOLDER,
                    fileNamePrefix=filename_pref,
                    region=region,
                    scale=self.cfg.GEE_SCALE,  # GSMaP native ~10 km
                    # crs='EPSG:4326',
                    # maxPixels=1e13
                )
                task.start()
                tasks += [task]
                file_names += [filename_pref]

            for task in tqdm(tasks):
                while task.active():
                    time.sleep(30)

        shutil.rmtree(self.cfg.RAINFALL_FOLDER)
        os.mkdir(self.cfg.RAINFALL_FOLDER)

        folder_name = self.cfg.GOOGLEDRIVE_RAINFALL_FOLDER
        query = f"title = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"
        folders = self.drive.ListFile({'q': query}).GetList()

        assert len(folders) == 1

        folder_id = folders[0]['id']
        query = f"'{folder_id}' in parents and trashed=false"
        files_in_folder = self.drive.ListFile({'q': query}).GetList()
        file_ids = [f['id'] for f in files_in_folder]

        with ThreadPoolExecutor() as executor:
            list(executor.map(self.download_gdrive_file, file_ids, [self.cfg.RAINFALL_FOLDER]*len(file_ids)))

    def parse_args(parser: ArgumentParser):
        parser.add_argument('--start', help="in YYYY-MM-DD format (inclusive)", default='2023-07-01', type=ee.Date)
        parser.add_argument('--end', help="in YYYY-MM-DD format (exclusive)", default='2023-07-03', type=ee.Date)
        parser.add_argument('--skip-gee', help="assume rasters are already fetched to Google Drive", default=False,
                            action='store_true')
