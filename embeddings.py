#embeddings
import os
import glob
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import faiss
from groq import Groq

# Embeddings model name
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Read the PDF file
def get_pdf_text(pdf_paths):
    text = ""
    files_read = len(pdf_paths)
    print(f"Number of files read: {files_read}")
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Creating a function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    total_chunks = len(chunks)
    print(f"Total number of chunks: {total_chunks}")
    return chunks

def main():
     # Provide PDF file paths beforehand
    pdf_folder = "F:/RightsMate/dataset/"
    pdf_paths = glob.glob(pdf_folder + "*.pdf")

    # Extract text from PDFs
    raw_texts = []
    for pdf_path in pdf_paths:
        raw_texts.append(get_pdf_text([pdf_path]))

    # Concatenate all raw texts into a single text
    all_text = "\n".join(raw_texts)

    # Split text into chunks
    text_chunks = get_text_chunks(all_text)
    
    # Count the number of embeddings created
    num_embeddings = len(text_chunks)
    print(f"Number of embeddings created: {num_embeddings}")
    
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_store")
    
if __name__ == "__main__":
    main()