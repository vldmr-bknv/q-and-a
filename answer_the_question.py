import argparse
import os

from src.downloaders.FileDownloader import FileDownloader
from src.parsers.ToyPdfParser import ToyPdfParser
from src.embedders.TestEmbedder import TestEmbedder
from langchain_community.vectorstores import FAISS

LLM_NAME = "gpt4all-falcon-newbpe-q4_0.gguf"

def file_download(source, path):
    downloader = FileDownloader(path)
    return downloader(source)


def get_sentences(filepath):
    parser = ToyPdfParser(filepath)
    return parser.get_sentences()


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
    if device == 'gpu': device = 'cuda'

    if LLM_NAME not in os.listdir('.'):
        print("LLM not found!")
    

    pdf_filename = file_download(source, "data")
    sentences = get_sentences(f"{pdf_filename}")
    embedder = TestEmbedder(device)
    embeddings = embedder()
    vector = FAISS.from_texts(sentences, embeddings)
    print(vector)
    



if __name__=='__main__':
    main()