import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def get_pdf_text(pdf_docs):
    text = ""  # Initialize the text variable
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000 , chunk_overlap = 1000 )
    chunks = text_splitter.split_text(text)
    return chunks
def get_vector_store(text_chunks):
   embedding_model_path = "models/embedding-001"

    # Get the Google API key from the environment variable
   google_api_key = os.getenv("GOOGLE_API_KEY")

   if not google_api_key:
        raise ValueError("Google API key is missing. Please set the GOOGLE_API_KEY environment variable.")

    # Initialize GoogleGenerativeAIEmbeddings with the specified model path and API key
   embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_path, google_api_key=google_api_key)

   vector_store = FAISS.from_texts(text_chunks,embedding =embeddings)
   vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context provide a general answer and in the start of genral answer only  write "based on my own knowledge", if the answer is from 
    the provided context dont write any thing in the strat of the answer
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])
  
          
# Main function for Streamlit app
def main():
    st.title("PDF Text Analysis and Question Answering")
    
    # File uploader for PDFs
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload one or more PDF files", accept_multiple_files=True, type="pdf")
    
    # Display uploaded file names
    if uploaded_files:
        st.sidebar.write("Uploaded PDFs:")
        for file in uploaded_files:
            st.sidebar.write(file.name)
    
    # User input for question
    user_question = st.text_input("Enter your question:")
    
    # Button to process PDFs and generate response
    if st.button("Process and Answer"):
        if uploaded_files:
            # Extract text from PDFs
            pdf_text = get_pdf_text(uploaded_files)
            # Split text into chunks
            text_chunks = get_text_chunks(pdf_text)
            # Create vector store
            get_vector_store(text_chunks)
            # Generate response to user question
            response = user_input(user_question)
            # Display response
            st.write("Reply:", response)
        else:
            st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
