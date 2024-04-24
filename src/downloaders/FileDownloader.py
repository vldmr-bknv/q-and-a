import wget
import logging

logger = logging.getLogger('FILE DOWNLOADER')
logger.setLevel(logging.DEBUG)
log_channel = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_channel.setFormatter(formatter)
logger.addHandler(log_channel)


class FileDownloader:

    def __init__(self, path):
        self.path = path


    def __call__(self, source):
        try:
            print("PDF downloading: ")
            filename = wget.download(source,
                          bar=wget.bar_thermometer,
                          out=self.path)
            return filename
        except Exception as e:
            logger.error(f"{source} downloading: {e}")
            return None