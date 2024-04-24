import wget
import logging

logger = logging.getLogger('FILE DOWNLOADER')
logger.setLevel(logging.DEBUG)
log_channel = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
log_channel.setFormatter(formatter)
logger.addHandler(log_channel)


class FileDownloader:
    '''A class used to download files.'''

    def __init__(self, path):
        '''
        Parameters
        ----------

        path : str
            file saving path
        '''
        self.path = path


    def __call__(self, source):
        '''Downloads the file from source, saves it to /data
        
        Parameters
        ----------
        
        source : str
            file url'''
        try:
            print("PDF downloading: ")
            filename = wget.download(source,
                          bar=wget.bar_adaptive,
                          out=self.path)
            print("\n")
            return filename
        except Exception as e:
            logger.error(f"{source} downloading: {e}")
            return None