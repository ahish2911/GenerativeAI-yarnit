from flask import Flask, render_template, request
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.document_loaders import UnstructuredURLLoader

app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answer', methods=['POST'])
def answer():
    urls = request.form['url']

    question = request.form['question']
    loaders = UnstructuredURLLoader(urls=[urls])
    data = loaders.load()

    # Split text from the given URL into chunks
    text_splitter = CharacterTextSplitter(
        separator='\n', chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)

    # Create embeddings and vector store
    db = Chroma.from_documents(docs, OpenAIEmbeddings())

     ##WE CAN DUMP THIS VECTORS TO OUR LOCAL DEVICE OR HARDDISK , SO THAT WE DONT HAVE TO USE OPENAI EMBEDDINGS AS IT HAS SOME COST.

    

    # Initialize OpenAI language model
    llm = OpenAI()

    # Create retrieval QA chain
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, retriever=db.as_retriever())

    # Get the answer
    answer = chain({"question": question}, return_only_outputs=True)

    return render_template('answer.html', answer=answer)


if __name__ == '__main__':
    app.run(debug=True)
