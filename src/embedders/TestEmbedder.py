from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from numpy import dot
from numpy.linalg import norm
from numpy import array

import logging
logger = logging.getLogger('EMBEDDER')
logger.setLevel(logging.DEBUG)
log_channel = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
log_channel.setFormatter(formatter)
logger.addHandler(log_channel)

class TestEmbedder:
    '''
    A class used to embeddings.

    Attributes
    ----------
    model_name : str
    embeddings : Embeddings
    device : str
        computing device (cpu, cuda)
    '''

    model_name = "sentence-transformers/all-MiniLM-L6-v2";
    embeddings = None
    
    def __init__(self, device):
        '''
        Parameters
        ----------

        device : str
            computing device (cpu, cuda)
        '''
        self.device = device
        self._embedder_init()


    def _embedder_init(self):
        '''Initializes the HuggingFaceEmbeddings.'''
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.model_name, 
                                               model_kwargs={'device': self.device})
            self.embeddings = embeddings
        except Exception as e:
            logger.error(f"Initialization: {e}")


    def are_similar(self, documents, threshold):
        '''Determines the cosine similarity of two documents.
        
        Two documets are similar if cosine similarity > threshold
        
        Parameters
        ----------
        
        documents : list
            list of strings
        threshold : float
        '''
        embeddings = self.embeddings.embed_documents(documents)
        doc1 = array(embeddings[0])
        doc2 = array(embeddings[1])
        cos_sim = dot(doc1, doc2) / (norm(doc1)*norm(doc2))
        return cos_sim > threshold


    def __call__(self):
        return self.embeddings