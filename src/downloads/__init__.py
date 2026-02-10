from logging import Logger
from argparse import ArgumentParser
from dataclasses import dataclass
import ee
import config as cfg
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

ee.Initialize(project=cfg.GEE_PROJECT_NAME)

class GenericDownloader:
    # Singleton pattern
    # _instance = None
    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super().__new__(cls, *args, **kwargs)
    #     return cls._instance

    @dataclass
    class InitializationData:
        gauth: GoogleAuth = None
        drive: GoogleDrive = None
    _init_structs = None

    def __init__(self, args, logger: Logger):
        self.cfg = cfg
        self.args = args
        self.logger = logger

        if GenericDownloader._init_structs is None:
            GenericDownloader._init_structs = GenericDownloader.InitializationData()
            GenericDownloader._init_structs.gauth = GoogleAuth(settings_file="src/pydrive_settings.yaml")
            GenericDownloader._init_structs.drive = GoogleDrive(GenericDownloader._init_structs.gauth)

        self.gauth = GenericDownloader._init_structs.gauth
        self.drive = GenericDownloader._init_structs.drive

    def download_gdrive_file(self, file_id, path):
        file_obj = self.drive.CreateFile({'id': file_id})
        # Fetching title first to name the local file
        file_obj.FetchMetadata()
        self.logger.info(f"Downloading {file_obj['title']}...")
        file_obj.GetContentFile(path + file_obj['title'])
        return f"Finished {file_obj['title']}"

    @staticmethod
    def parse_args(parser: ArgumentParser):
        pass

    def main(self):
        pass