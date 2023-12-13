import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss  #faiss is used to store the vectors data locally in the own pc rther than uploading in cloud 
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#pip install InstructorEmbedding sentence_transformers 
# above instructor embedings are used to run embedings for free rather than getting charged by using openai apis 





def get_pdf_text(pdf_docs):
    text = " "
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):      # we will be using langchain to convert into chunks 
     text_splitter = TokenTextSplitter(
         chunk_size = 70, #the size of one para is chunk   
         chunk_overlap = 0,  #it will start next chunk few characters before 
     ) 
     chunks = text_splitter.split_text(raw_text)
     st.write(chunks)
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name ="hkunlp/instructor-xl")


    vectorstore = faiss.from_texts(texts = text_chunks , embedding = embeddings)
    return vectorstore
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()     ##changed
    memory = ConversationBufferMemory(memory_key='chat_history' , return_messages= True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain



def main():
    load_dotenv()
    st.set_page_config(page_title="chat with mutiple pdfs",page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("chat with mutiple pdfs :books:")
    st.text_input("Ask a question ?")

    with st.sidebar:
        st.subheader("documents")
        pdf_docs = st.file_uploader("upload your pdfs and click on process" ,accept_multiple_files= True)
        if st.button("process"): 
            with st.spinner("Processing..."):   #it shows processing while its processing 
                # get pdf text 
                raw_text = get_pdf_text (pdf_docs)                 


                #get the text chunks
                text_chunks = get_text_chunks(raw_text) #by providing raw text aas parmeter to that function it will convert into chunks 
                st.write(text_chunks)
                  
                #create vector store database
                vectorstore = get_vectorstore(text_chunks)


                # conversation chain 
                st.session_state.conversation = get_conversation_chain(vectorstore)
                #stream lit stopd the reintialing variable using session state
                 



if __name__=='__main__':
    main() 