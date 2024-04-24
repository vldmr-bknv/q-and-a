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

    model_name = "sentence-transformers/all-MiniLM-L6-v2";
    embeddings = None
    
    def __init__(self, device):
        self.device = device
        self._embedder_init()


    def _embedder_init(self):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.model_name, 
                                           model_kwargs={'device': self.device})
            self.embeddings = embeddings
        except Exception as e:
            logger.error(f"Initialization: {e}")


    def are_similar(self, documents, threshold):
        embeddings = self.embeddings.embed_documents(documents)
        doc1 = array(embeddings[0])
        doc2 = array(embeddings[1])
        cos_sim = dot(doc1, doc2) / (norm(doc1)*norm(doc2))
        return cos_sim > threshold


    def __call__(self):
        return self.embeddings