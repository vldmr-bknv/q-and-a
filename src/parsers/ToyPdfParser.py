from langchain_community.document_loaders import PyPDFLoader
import nltk

import logging
logger = logging.getLogger('PDF PARSER')
logger.setLevel(logging.DEBUG)
log_channel = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
log_channel.setFormatter(formatter)
logger.addHandler(log_channel)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ToyPdfParser:
    '''A class used to cast a PDF to a string type and tokenize it.'''

    def __init__(self, filepath):
        '''
        Parameters
        ----------

        filepath : str
            PDF-file local path
        '''
        self.filepath = filepath


    def get_sentences(self):
        '''Tokenizes text into sentences.'''

        try:
            loader = PyPDFLoader(self.filepath)
        except Exception as e:
            logger.error(f"PDF load: {e}")
            return None
        
        try:
            pages = loader.load_and_split()
            assert len(pages) != 0, logger.error(f"PDF content not found!")
        except:
            return None

        str_pages = list(map(lambda x: x.page_content.replace("\n", " "), pages))

        sent_tokens = []
        for page in str_pages:
            sent_tokens += nltk.sent_tokenize(page)            
        
        return sent_tokens