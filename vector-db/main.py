import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain import VectorDBQA, OpenAI

from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")
PINECONE_API_KEY = config("PINECONE_API_KEY")

'''
Works only with pinecone 2.2.4. so pip install pinecone-client==2.2.4
TBD: Modify for pinecone-client 3.0 
'''
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="gcp-starter",
)

if __name__ == "__main__":
    print("Hello VectorDB")

    loader = TextLoader(
        "mediumblogs/mediumblog1.txt"
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index_name = 'medium-blogs-embeddings-index'
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff", vectorstore=docsearch, return_source_documents=True
    )
    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)