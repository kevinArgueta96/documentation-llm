from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from consts import INDEX_NAME
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone


## NEW UPLOAD PINECONE LIBRARIES
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

from PyPDF2 import PdfReader

import os

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)

# load PDF file
pdf_file = open("pdf-files/codigo-de-trabajo.pdf", "rb")
pdf_reader = PdfReader(pdf_file)

text = ""
# Use the .pages attribute to get a list of page objects
for page in pdf_reader.pages:
    page_text = page.extract_text()
    if page_text:  # Check if there is text to add
        text += page_text

pdf_file.close()

loader = TextLoader(text, encoding='utf-8')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len
)
docs = text_splitter.split_documents(documents)

chunks = text_splitter.split_text(text)

#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
embeddings = OpenAIEmbeddings()

PineconeLangChain.from_documents(documents, embeddings, index_name=INDEX_NAME)

#print("PART OF CHUNK:" + chunks[7])
#print(len(chunks))


#TEST EMBEDDIGNS
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
#embeddings = OpenAIEmbeddings()

#knowledge_base = FAISS.from_texts(chunks, embeddings)

question = "¿Cuáles son las obligaciones específicas que debe cumplir el agente reclutador o la empresa, según el artículo 34, para garantizar la protección y bienestar de los trabajadores guatemaltecos que son contratados para trabajar fuera del país?"
#docs = knowledge_base.similarity_search(question, 3)

#PineconeLangChain.from_documents(text, embeddings, index_name=INDEX_NAME)

"""
prompt = ChatPromptTemplate.from_template(
    question +
    "\n\n{context}"
)

print("START TO PROCESS THE INFORMATION WITH THE LLM")
#model = ChatOpenAI(model="gpt-3.5-turbo-0125", verbose = True, temperature=0)
model = Ollama(model="mistral", verbose=True, temperature=0)

chain = create_stuff_documents_chain(model, prompt)

new_prompt = chain.invoke({"context": docs})

print(new_prompt)
"""
