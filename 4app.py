import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from PyPDF2 import PdfReader
import os

# Streamlit Configuration
st.set_page_config(page_title="PDF GPT Chatbot", layout="centered")
st.title("📄 GPT Chatbot with PDF Data")

# OpenAI Configuration
GPT_MODEL = "gpt-4o-mini"  # Use your specified model
client = OpenAI(model=GPT_MODEL, temperature=0)

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load the static dataset from the PDF
def load_static_data():
    pdf_path = "Pellet_mill.pdf"  # Ensure this file exists in the same directory or adjust the path
    return extract_text_from_pdf(pdf_path)

# Generate a response from the model
def generate_response(query_text):
    # Load and preprocess data
    st.info("Loading and processing PDF data...")
    document_text = load_static_data()

    # Split documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(document_text)

    # Prepare documents for QA chain
    input_documents = [Document(page_content=chunk) for chunk in chunks]

    # Create QA chain
    qa_chain = load_qa_chain(llm=client, chain_type="stuff")

    # Process each chunk
    st.info("Processing chunks to generate response...")
    aggregated_response = ""
    for i, doc in enumerate(input_documents):
        try:
            response = qa_chain.run({"input_documents": [doc], "question": query_text})
            aggregated_response += f"Chunk {i+1} Response:\n{response}\n\n"
        except Exception as e:
            aggregated_response += f"Chunk {i+1} Error: {str(e)}\n\n"

    return aggregated_response

# Streamlit UI for Query Input
query_text = st.text_input("Enter your question:", placeholder="Ask a specific question about the PDF.")

# Button to trigger processing
if st.button("Submit") and query_text:
    with st.spinner("Processing your query..."):
        try:
            result = generate_response(query_text)
            st.success("Response generated successfully!")
            st.text_area("Response:", value=result, height=400)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
