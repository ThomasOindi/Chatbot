
import streamlit as st
import pickle
import time
import faiss
import base64


from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from llama_index.core import SimpleDirectoryReader
from langchain_core.documents import Document  # Import LangChain's Document class

from dotenv import load_dotenv
import os
OPENAI_API_KEY: str= os.getenv("OPENAI_API_KEY")


load_dotenv('.env')
# 🔹 Streamlit App Config
st.set_page_config(page_title="ISUZU IntelliChat R1")

def set_background(background_image_path, logo_image_path):
    """Sets a full-page background image and a small logo in the top left."""
    with open(background_image_path, "rb") as bg_file:
        encoded_bg = base64.b64encode(bg_file.read()).decode()
    with open(logo_image_path, "rb") as logo_file:
        encoded_logo = base64.b64encode(logo_file.read()).decode()
    background_style = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{encoded_bg}") no-repeat center center fixed;
        background-size: cover;
    }}
    .bg-left {{
        position: fixed;
        top: 80px;
        left: 70px;
        width: 150px;
        height: auto;
        z-index: 1;
    }}
    </style>
    <img src="data:image/png;base64,{encoded_logo}" class="bg-left">
    """
    st.markdown(background_style, unsafe_allow_html=True)

set_background(
    r"C:\Users\Thomas.Okiwi\OneDrive - Techno Brain Group\Documents\Data Science Projects\Generative AI\Car.jpg",
    r"C:\Users\Thomas.Okiwi\OneDrive - Techno Brain Group\Documents\Data Science Projects\Generative AI\Logo.png"
)



# 🔹 Streamlit App Title
st.markdown(
    """
    <h1 style="color: black; margin-top: 0;">ISUZU IntelliChat 1.0</h1>


    <h7 style="color: white; text-align: left;"> 💬 Ask me any question!!!</h3>
    """,
    unsafe_allow_html=True
)

# 🔹 Load and Process Documents
@st.cache_resource(show_spinner="📂 Loading and Indexing Documents...")
def load_documents():
    folder_path = "Data"
    llama_documents = SimpleDirectoryReader(folder_path).load_data()
    langchain_documents = [
        Document(page_content=doc.text, metadata={"source": doc.doc_id})
        for doc in llama_documents
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(langchain_documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

vector_store = load_documents()
retriever = vector_store.as_retriever()

# Define LLM and QA Chain with Improved Prompt
def custom_prompt(context, question):
    return f"""
    You are an AI assistant trained on ISUZU truck documents. Answer the question accurately based on the retrieved documents.
    
    Context: {context}
    Question: {question}
    
    If you don't know the answer, say "I don't know" instead of making up information.
    """

llm = OpenAI(temperature=0.4, max_tokens=500)
qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

# 🔹 Query Input
query = st.text_input("Enter your question:", "")

if query:
    with st.spinner("🔍 Searching for answers..."):
        response = qa_chain({"question": query})

    # Format the response as a list if it contains multiple lines
    answer_lines = response["answer"].split("\n")
    formatted_answer = "".join(f"<li>{line.strip()}</li>" for line in answer_lines if line.strip())

    # Display Response
    st.subheader("Answer")
    st.markdown(
        f"""
        <div style="color: white; background-color: black; padding: 10px; border-radius: 10px;">
            <ul>{formatted_answer}</ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 🔹 Feedback Section
    st.subheader("📝 Feedback")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, this answer is correct"):
            st.success("Thanks for your feedback! 😊")

    with col2:
        if st.button("❌ No, this answer is incorrect"):
            st.session_state.show_feedback = True

    # Display feedback text area only if "No" was clicked
    if st.session_state.get("show_feedback", False):
        feedback_text = st.text_area("💬 Please explain what went wrong:")
        if st.button("Submit Feedback"):
            if feedback_text.strip():
                st.success("Thank you! Your feedback will help us improve. 🚀")
                st.session_state.show_feedback = False  # Hide feedback after submission
            else:
                st.warning("⚠️ Please provide some details before submitting.")
