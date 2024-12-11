import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader

# Streamlit Page Configuration
st.set_page_config(page_title="PDF GPT Chatbot", layout="centered")
st.title("📄 GPT Chatbot with PDF Data")

# Access the OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Ensure this is set in Streamlit secrets
GPT_MODEL = "gpt-4"

# Validate API Key
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please add it to Streamlit secrets.")
    st.stop()

# Initialize ChatOpenAI Client
client = ChatOpenAI(model=GPT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load the static dataset from the PDF
def load_static_data():
    pdf_path = "data/Pellet_mill.pdf"  # Updated path for the PDF file
    return extract_text_from_pdf(pdf_path)

# Function to prepare FAISS vector store
def prepare_vector_store(document_text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(document_text)

    # Convert chunks into Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store

# Generate a response from the model
def generate_response(query_text):
    st.info("Loading and processing PDF data...")
    document_text = load_static_data()

    # Prepare vector store
    vector_store = prepare_vector_store(document_text)

    # Initialize RetrievalQA chain
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=client, retriever=retriever)

    # Get the answer from the chain
    st.info("Retrieving answer...")
    try:
        response = qa_chain.run(query_text)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI for Query Input
query_text = st.text_input("Enter your question:", placeholder="Ask a specific question about the PDF.")

# Process the query and display the response
if st.button("Submit") and query_text:
    with st.spinner("Processing your query..."):
        try:
            result = generate_response(query_text)
            st.text_area("Response:", value=result, height=400)
        except Exception as e:
            st.error(f"An error occurred: {e}")
