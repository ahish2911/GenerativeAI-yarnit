import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.document_loaders import UnstructuredURLLoader

# Global variables to store URL and its embeddings
stored_url = None
stored_db = None

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

def ingest_url(url):
    global stored_db, stored_url
    # Load data from the URL
    loaders = UnstructuredURLLoader(urls=[url])
    data = loaders.load()

    # Split text from the given URL into chunks
    text_splitter = CharacterTextSplitter(
        separator='\n', chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)

    # Create embeddings and vector store
    stored_db = Chroma.from_documents(docs, OpenAIEmbeddings())
    stored_url = url

def main():
    st.title('Question Answer chatbot')

    global stored_url, stored_db

    url_input = st.text_input('Enter URL:', stored_url if stored_url else '')
    question_input = st.text_input('Enter Question:', '')

    if st.button('Get Answer'):
        if not url_input:
            st.warning('Please enter a URL.')
        elif not question_input:
            st.warning('Please enter a question.')
        elif url_input != stored_url:
            ingest_url(url_input)

        if stored_db is not None:
            # Initialize OpenAI language model
            llm = OpenAI()

            # Create retrieval QA chain
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, retriever=stored_db.as_retriever())

            # Get the answer
            answer = chain({"question": question_input}, return_only_outputs=True)
            if "answer" in answer:
                st.write('Answer:', answer["answer"])
        else:
            st.warning("Please enter a valid URL.")

if __name__ == '__main__':
    main()
