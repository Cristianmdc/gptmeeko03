import streamlit as st
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pysqlite3
import sys
import os
from PyPDF2 import PdfReader

# Replace sqlite3 module with pysqlite3 for Chroma compatibility
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Access the OpenAI API key from Streamlit secrets
api_key = st.secrets["YOUR_OPENAI_API_KEY"]
YOUR_OPENAI_API_KEY = st.secrets["YOUR_OPENAI_API_KEY"]

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load the static dataset from the PDF
def load_static_data():
    pdf_path = "Pellet_mill.pdf"  # Ensure this matches your repository's filename
    return extract_text_from_pdf(pdf_path)

def generate_response(query_text):
    if not YOUR_OPENAI_API_KEY:
        raise ValueError("OpenAI API Key is not set. Please set it in the environment variables.")

    # Load and preprocess data
    document_text = load_static_data()

    # Split documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    texts = text_splitter.split_text(document_text)

    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=YOUR_OPENAI_API_KEY)

    # Initialize Chroma vector store with a persistence directory
    db = Chroma.from_texts(
        texts,
        embeddings,
        persist_directory=".chroma_data"  # Directory for persistence
    )

    # Create retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": min(3, len(texts))})  # Adjust number of results

    # Ensure prompt length is within model limits
    retrieved_docs = retriever.invoke(query_text)
    combined_text = "\n".join([doc.page_content for doc in retrieved_docs])

    if len(combined_text) + len(query_text) > 4000:
        # Reduce size of retrieved context dynamically
        max_fit_length = 4000 - len(query_text) - 100  # Allowing buffer
        truncated_text = combined_text[:max_fit_length]
        combined_text = truncated_text

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=YOUR_OPENAI_API_KEY, temperature=0),
        chain_type="stuff",
        retriever=retriever
    )
    return qa.run(query_text)

# Streamlit page title and description
st.set_page_config(page_title='GPT Chatbot with PDF Data')
st.title('📄 GPT Chatbot: PDF Data')

# User input for query
query_text = st.text_input('Enter your question:', placeholder='Ask a specific question about the document.')

# Generate response when input is provided
if st.button("Submit") and query_text:
    with st.spinner('Processing your request...'):
        try:
            response = generate_response(query_text)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
