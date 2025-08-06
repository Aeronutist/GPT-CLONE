# app.py

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
import torch
import os

st.set_page_config(page_title="Claude/Gemini Clone", layout="wide")
st.title("💬 Claude/Gemini Clone with LangChain")

# Set HF token securely
HF_TOKEN = st.secrets.get("HF_TOKEN")

# SOLUTION 1: Use a smaller, more efficient model
@st.cache_resource
def load_llm_lightweight():
    """Load a lightweight model that works better on Streamlit Cloud"""
    model_id = "microsoft/DialoGPT-small"  # Much smaller model
    # Alternative options:
    # model_id = "distilgpt2"
    # model_id = "gpt2"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,  # Use float32 instead of float16 for better compatibility
            device_map="cpu",  # Force CPU usage
            token=HF_TOKEN,
            low_cpu_mem_usage=True
        )
        
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=256,  # Reduced token limit
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# SOLUTION 2: Use Hugging Face Inference API (Recommended)
@st.cache_resource
def load_llm_api():
    """Use Hugging Face Inference API - no local model loading required"""
    from langchain_community.llms import HuggingFaceHub
    
    try:
        return HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=HF_TOKEN,
            model_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 512,
                "do_sample": True,
            }
        )
    except Exception as e:
        st.error(f"Error setting up API: {str(e)}")
        return None

# SOLUTION 3: Fallback to a pipeline-only approach
@st.cache_resource
def load_llm_pipeline():
    """Use pipeline directly without loading model explicitly"""
    try:
        pipe = pipeline(
            "text-generation",
            model="distilgpt2",  # Small, fast model
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            device=0 if torch.cuda.is_available() else -1
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error creating pipeline: {str(e)}")
        return None

# Load FAISS with error handling
@st.cache_resource
def load_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        # Create a dummy vectorstore if file doesn't exist
        import numpy as np
        from langchain.docstore.document import Document
        
        # Create some dummy documents for testing
        docs = [Document(page_content="This is a test document.", metadata={})]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)

# Model selection in sidebar
st.sidebar.title("Model Configuration")
model_option = st.sidebar.selectbox(
    "Choose Model Loading Method:",
    ["Hugging Face API (Recommended)", "Lightweight Local Model", "Pipeline Only"]
)

# Load the selected model
if model_option == "Hugging Face API (Recommended)":
    llm = load_llm_api()
elif model_option == "Lightweight Local Model":
    llm = load_llm_lightweight()
else:
    llm = load_llm_pipeline()

if llm is None:
    st.error("Failed to load language model. Please check your configuration and try again.")
    st.stop()

# Load vectorstore
try:
    retriever = load_vectorstore().as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
except Exception as e:
    st.error(f"Error setting up QA chain: {str(e)}")
    st.stop()

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display current model info
st.sidebar.info(f"Current model: {model_option}")
if HF_TOKEN:
    st.sidebar.success("✅ HF Token configured")
else:
    st.sidebar.error("❌ HF Token missing")

# Chat input
question = st.chat_input("Ask anything about the document...")

if question:
    with st.spinner("Thinking..."):
        try:
            result = qa_chain({"question": question})
            answer = result["answer"]
            
            # Clean up the answer if it contains repetitive text
            if len(answer.split()) > 100:
                sentences = answer.split('.')
                answer = '. '.join(sentences[:3]) + '.'
            
            st.session_state.chat_history.append((question, answer))
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.chat_history.append((question, "Sorry, I encountered an error processing your question."))

# Display chat history
for q, a in st.session_state.chat_history:
    st.chat_message("user").markdown(f"**You:** {q}")
    st.chat_message("assistant").markdown(f"**AI:** {a}")

# Clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
