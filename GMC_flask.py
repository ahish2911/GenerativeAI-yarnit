from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Flask app
app = Flask(__name__)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries"),
        ("user", "generate a post for a  {topic}\n and its Format should be like that of {format}")
    ]
)

# OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

@app.route('/')
def home():
    return render_template('new_index.html')

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    topic = data.get('topic')
    format_ = data.get('format')

    if not topic or not format_:
        return jsonify({'error': 'Topic and format are required'}), 400

    # Invoke the chain with topic and format parameters
    output = chain.invoke({'topic': topic, 'format': format_})
    
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True)
