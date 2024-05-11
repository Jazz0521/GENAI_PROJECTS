import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai

#from langchain.vectorstores import FAISS 
from langchain_community.vectorstores import FAISS#for vector embeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain # helps to do chats 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv # to load all env variables 

load_dotenv() # by this will be able to see the env variables 

os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) #cuz i need to configiure my api key from google gemini - so basically configuring the api key wrt whatever google api keyi have loaded in the .env file





#func to read the uploaded pdfs, go through each page and extract the text 
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs: 
        pdf_reader= PdfReader(pdf) #this pdf_reader is in form of a list 
        for page in pdf_reader.pages:
            text+= page.extract_text() #from this pg extract all details
    return  text


#func to divide the text into smaller text chunks 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#func to convert the text chunks into vectors 
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) #take all text chunks and embed according to the embedding i have initialized 
    vector_store.save_local("faiss_index")#this vector_store i am saving in local , faiss_index is the folder thats created and inside it are the vectors in some unreadable format  


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) #chain type as stuff cuz i also need to do internal text summarization

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True) #doing similarity search 
    
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Ask Questions from multiple PDF and get answer")
    st.header("Ask Questions from multiple PDF and get answer💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__": #to run streaclit apps do streamlit run app.py
    main()
