import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from PyPDF2 import PdfReader

# Streamlit Page Configuration
st.set_page_config(page_title="PDF GPT Chatbot", layout="centered")
st.title("📄 GPT Chatbot with PDF Data")

# Access the OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Ensure this is set in Streamlit secrets
GPT_MODEL = "gpt-4"  # Replace with "gpt-4o-mini" if available and valid

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

# Generate a response from the model
def generate_response(query_text):
    st.info("Loading and processing PDF data...")
    document_text = load_static_data()

    # Split the document into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(document_text)

    # Prepare documents for QA chain
    input_documents = [Document(page_content=chunk) for chunk in chunks]

    # Create QA chain
    st.info("Processing chunks to generate response...")
    aggregated_response = ""
    for i, doc in enumerate(input_documents):
        try:
            # Use ChatOpenAI and chain properly
            qa_chain = load_qa_chain(client, chain_type="stuff")
            response = qa_chain.invoke({"input_documents": [doc], "question": query_text})
            aggregated_response += f"Chunk {i+1} Response:\n{response['output']}\n\n"
        except Exception as e:
            aggregated_response += f"Chunk {i+1} Error: {str(e)}\n\n"

    return aggregated_response

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
