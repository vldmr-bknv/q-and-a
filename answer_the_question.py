import argparse

from src.downloaders.FileDownloader import FileDownloader
from src.parsers.ToyPdfParser import ToyPdfParser
from src.embedders.TestEmbedder import TestEmbedder

def file_download():
    downloader = FileDownloader()
    downloader.hello()


def get_sentences():
    parser = ToyPdfParser()
    parser.hello()


def get_embeddings():
    embedder = TestEmbedder()
    embedder.hello()


def main():
    arg_parser = argparse.ArgumentParser(description="The program answers the question about the document.",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument("-s", "--source", 
                            type=str, 
                            help="Link to document.", 
                            required=True)
    
    arg_parser.add_argument("-q", "--question", 
                            type=str, 
                            help="Question about the context of the document.",
                            required=True)
    arg_parser.add_argument("-d", "--device",
                            type=str, 
                            help="Computing device.", 
                            choices=['cpu', 'gpu'])
    args = vars(arg_parser.parse_args())
    
    source = args['source']
    question = args['question']
    device = args['device']

    # File (model and documet downloading)
    file_download()

    # Document parsing
    get_sentences()

    # Getting vectors
    get_embeddings()


if __name__=='__main__':
    main()