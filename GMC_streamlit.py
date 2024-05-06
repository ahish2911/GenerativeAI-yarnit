from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
os.environ["OPENAI_API_KEY"] = "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"
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Topic: {topic}\nFormat: {format}")
    ]
)
st.title('Marketing Content Generator')
input_topic = st.text_input("Enter the topic")
input_format = st.text_input("Enter the desired format (e.g., LinkedIn post, Email, etc.)")


# openAI LLm 
llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_topic:
    # Invoke the chain with topic and format parameters
    output = chain.invoke({'topic': input_topic, 'format': input_format})
    st.write(output)