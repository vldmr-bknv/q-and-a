import argparse
import os
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.downloaders.FileDownloader import FileDownloader
from src.parsers.ToyPdfParser import ToyPdfParser
from src.embedders.TestEmbedder import TestEmbedder

import logging
logger = logging.getLogger('MAIN')
logger.setLevel(logging.DEBUG)
log_channel = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
log_channel.setFormatter(formatter)
logger.addHandler(log_channel)


LLM_NAME = "gpt4all-falcon-newbpe-q4_0.gguf"
NO_DATA = ["there is no information provided in the given context",
           "is not specified", "no data", 
           "it is impossible to answer this question based only on the given context"]


def get_chain(llm, prompt):
    ''' Return chain containing llm, prompt and context.'''
    return create_stuff_documents_chain(llm, prompt)


def get_prompt():
    ''' Returns a prompt that optimizes the llm's inference.'''
    return PromptTemplate.from_template("""Please, answer the following question based only on the 
                                        provided context and nothing more: {context} 
                                        
                                        Question: {input}
                                        """)


def main():
    arg_parser = argparse.ArgumentParser(description="The program answers the question about the document.",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument("-s", "--source", type=str, help="Link to document.") 
    arg_parser.add_argument("-q", "--question", type=str, help="Question about the context of the document.")
    arg_parser.add_argument("-d", "--device", type=str, help="Computing device.", choices=['cpu', 'gpu'])
    args = vars(arg_parser.parse_args())
    
    source = args['source']
    question = args['question']
    device = args['device']
    if device == 'gpu': device = 'cuda'

    if LLM_NAME not in os.listdir('.'):
        logger.error("LLM not found!")
        return
    
    try:
        llm = GPT4All(model=f"./{LLM_NAME}", device=device)
    except Exception as e:
        logger.error(f"LLM initialize: {e}")
        return
    
    # Downloading a file
    downloader = FileDownloader("data")
    pdf_filename = downloader(source)
    
    # Converting PDF-file to list of sentence tokens
    parser = ToyPdfParser(f"{pdf_filename}")
    sentences = parser.get_sentences()
    
    # Getting embeddings. Saving them to FAISS
    embedder = TestEmbedder(device)
    embeddings = embedder()
    vector = FAISS.from_texts(sentences, embeddings)

    # Getting prompt and chain
    prompt = get_prompt()
    chain = get_chain(llm, prompt)

    print(f"Question: {question}\n")
    
    print("--------- Answer ----------")
    print('Hmm, please wait, I need to think...')
    
    # Getting an answer to a question. 
    answer = chain.invoke({"input": question,
                           "context": vector.similarity_search(question)})
    
    # According to the task, if the chain does not return a response according 
    # to the context, it is required to return the string: "No data to answer the question".
    # However, the LLM may return the following:
    # - there is no information provided in the given context about...
    # - it is impossible to answer this question based only on the given context...
    # - is not specified, etc.
    # 
    # To determine the characteristic of the model's response, we find the cosine similarity 
    # between the model's response and possible responses when the model did not
    # find an answer in the context.
    # 
    # This is a temporary, not reliable solution.
    if embedder.are_similar([" ".join(NO_DATA), answer], 0.2):
        print("No data to answer the question.")
    else:
        print(answer.strip())
    

if __name__=='__main__':
    main()