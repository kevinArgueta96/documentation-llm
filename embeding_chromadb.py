from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from dotenv import load_dotenv


## NEW UPLOAD PINECONE LIBRARIES
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

import os

def ingest_docs():
    # load PDF file using loaders from langchain
    loader = PyPDFLoader("pdf-files/codigo-de-trabajo.pdf")
    document = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    docs = text_splitter.split_documents(document)

    # Embed and store the docs from pdf files
    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = 'db'

    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)

    vectordb.persist()
    print("****Loading to vectorstore done ***")

def callDBfuncition():
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)

    query = "Puedes decirme cuantas horas laborales se tienen que trabajar, extiendete en la respuesta"
    response = qa.run(query)
    print(response)

if __name__ == "__main__":
    #ingest_docs()
    callDBfuncition()