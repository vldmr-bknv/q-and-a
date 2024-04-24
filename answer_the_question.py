import argparse
import os
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.downloaders.FileDownloader import FileDownloader
from src.parsers.ToyPdfParser import ToyPdfParser
from src.embedders.TestEmbedder import TestEmbedder


LLM_NAME = "gpt4all-falcon-newbpe-q4_0.gguf"
NO_DATA = ["there is no information provided in the given context",
           "is not specified", "no data", 
           "it is impossible to answer this question based only on the given context"]

def file_download(source, path):
    downloader = FileDownloader(path)
    return downloader(source)


def get_sentences(filepath):
    parser = ToyPdfParser(filepath)
    return parser.get_sentences()


def get_chain(llm, prompt):
    return create_stuff_documents_chain(llm, prompt)


def get_prompt():
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
        print("LLM not found!")
        return
    
    try:
        llm = GPT4All(model="./" + LLM_NAME, device=device)
    except Exception as e:
        print(e)
        return
    
    

    pdf_filename = file_download(source, "data")
    sentences = get_sentences(f"{pdf_filename}")
    embedder = TestEmbedder(device)
    embeddings = embedder()
    vector = FAISS.from_texts(sentences, embeddings)

    prompt = get_prompt()
    chain = get_chain(llm, prompt)

    print(f"Question: {question}\n")
    
    print("--------- Answer ----------")
    print('Hmm, please wait, I need to think...')
    answer = chain.invoke({"input": question,
                           "context": vector.similarity_search(question)})
    
    if embedder.are_similar([" ".join(NO_DATA), answer], 0.2):
        print(answer)
        print("No data to answer the question.")
    else:
        print(answer.strip())
    

if __name__=='__main__':
    main()