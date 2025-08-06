# app.py

import streamlit as st
import torch
import os
import tempfile
from typing import Optional, List, Tuple
import PyPDF2
import io

# Check if we're in a resource-constrained environment
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_CLOUD") or "streamlit" in os.getcwd().lower()

st.set_page_config(page_title="📄 Document Q&A Assistant", layout="wide")
st.title("📄 Document Q&A Assistant - Ask Questions About Your Documents")

# Set HF token securely
HF_TOKEN = st.secrets.get("HF_TOKEN") if hasattr(st, 'secrets') else os.getenv("HF_TOKEN")

# Document processing functions
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    """Extract text from uploaded text file"""
    try:
        # Handle different encodings
        try:
            text = txt_file.read().decode('utf-8')
        except UnicodeDecodeError:
            txt_file.seek(0)
            text = txt_file.read().decode('latin-1')
        return text
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > start + chunk_size // 2:
                chunk = text[start:break_point + 1]
                end = break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

# SOLUTION 1: Use Hugging Face Inference API
@st.cache_resource
def load_hf_inference_llm():
    """Use Hugging Face Inference API with proper client"""
    try:
        from huggingface_hub import InferenceClient
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        
        class HuggingFaceInferenceLLM(LLM):
            client: InferenceClient = None
            model_id: str = "microsoft/DialoGPT-medium"  # More reliable model
            max_new_tokens: int = 300
            temperature: float = 0.7
            
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.client = InferenceClient(model=self.model_id, token=HF_TOKEN)
            
            @property
            def _llm_type(self) -> str:
                return "huggingface_inference"
            
            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs
            ) -> str:
                try:
                    # Format prompt for better responses
                    formatted_prompt = f"""Based on the provided context, please answer the following question clearly and concisely.

Context: {prompt.split('Context:')[-1].split('Question:')[0].strip() if 'Context:' in prompt else ''}

Question: {prompt.split('Question:')[-1].strip() if 'Question:' in prompt else prompt}

Answer:"""
                    
                    response = self.client.text_generation(
                        prompt=formatted_prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        stop_sequences=stop or ["Question:", "Context:"],
                        return_full_text=False
                    )
                    
                    # Clean up response
                    response = response.strip()
                    if response.startswith("Answer:"):
                        response = response[7:].strip()
                    
                    return response
                except Exception as e:
                    return f"I apologize, but I encountered an error processing your question. Please try rephrasing it."
        
        return HuggingFaceInferenceLLM()
    except Exception as e:
        st.error(f"Error setting up HF Inference: {str(e)}")
        return None

# SOLUTION 2: Use a more reliable local model
@st.cache_resource
def load_local_llm():
    """Load a small local model that works reliably"""
    try:
        from transformers import pipeline
        from langchain_community.llms import HuggingFacePipeline
        
        # Use FLAN-T5 which is good for Q&A tasks
        model_name = "google/flan-t5-small"
        
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            device=-1,  # Force CPU
            torch_dtype=torch.float32
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading local model: {str(e)}")
        return None

# SOLUTION 3: Enhanced Mock LLM for testing
class EnhancedMockLLM:
    """Enhanced mock LLM that can work with document context"""
    
    def __init__(self):
        self.context = ""
    
    def _call(self, prompt: str, **kwargs) -> str:
        return self.__call__(prompt)
    
    def __call__(self, inputs):
        if isinstance(inputs, dict) and "question" in inputs:
            question = inputs["question"]
        elif isinstance(inputs, str):
            # Extract question from formatted prompt
            if "Question:" in inputs:
                question = inputs.split("Question:")[-1].strip()
            else:
                question = inputs
        else:
            question = str(inputs)
        
        # Extract context if present
        context = ""
        if isinstance(inputs, str) and "Context:" in inputs:
            context = inputs.split("Context:")[1].split("Question:")[0].strip()
        
        # Simple keyword-based responses using context
        question_lower = question.lower()
        
        if context:
            # Try to find relevant information in context
            context_lower = context.lower()
            words = question_lower.split()
            relevant_sentences = []
            
            for sentence in context.split('.'):
                if any(word in sentence.lower() for word in words if len(word) > 3):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                return f"Based on the document: {'. '.join(relevant_sentences[:2])}."
        
        # Default responses
        responses = {
            "what": f"Based on the document, regarding your question about '{question}', I can see relevant information in the provided context.",
            "how": f"The document explains how this works. Here's what I found relevant to your question.",
            "why": f"According to the document, the reason appears to be related to the context provided.",
            "when": f"The document mentions timing related to your question.",
            "where": f"The location or place mentioned in the document seems relevant to your question.",
            "who": f"The document identifies relevant people or entities related to your question."
        }
        
        for key, response in responses.items():
            if key in question_lower:
                return response
        
        return f"I found information in the document related to your question: '{question}'. This is a mock response for testing purposes."

# Load embeddings and create vectorstore from uploaded documents
@st.cache_resource
def load_embeddings():
    """Load embeddings model"""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

def create_vectorstore_from_text(text_chunks, embeddings):
    """Create FAISS vectorstore from text chunks"""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain.docstore.document import Document
        
        if not text_chunks:
            return None
        
        # Create documents from chunks
        documents = [
            Document(page_content=chunk, metadata={"chunk_id": i})
            for i, chunk in enumerate(text_chunks)
        ]
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

# Enhanced QA Chain for document-based Q&A
class DocumentQAChain:
    def __init__(self, llm, retriever=None):
        self.llm = llm
        self.retriever = retriever
        self.chat_history = []
    
    def __call__(self, inputs):
        question = inputs["question"]
        
        # Get relevant context from documents
        context = ""
        if self.retriever:
            try:
                docs = self.retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs[:3]])  # Top 3 relevant chunks
            except Exception as e:
                context = "Error retrieving context from documents."
        
        if not context.strip():
            return {"answer": "I don't have any document context to answer your question. Please upload a document first.", "source_documents": []}
        
        # Format prompt for better responses
        if hasattr(self.llm, '_call'):
            prompt = f"""Context from the uploaded document:
{context}

Question: {question}

Please provide a clear and accurate answer based only on the information in the document context above."""
            
            try:
                answer = self.llm._call(prompt)
            except:
                answer = self.llm(prompt)
        else:
            # For mock LLM
            prompt = f"Context: {context}\nQuestion: {question}"
            answer = self.llm(prompt)
        
        # Clean up answer
        if len(answer) > 800:
            sentences = answer.split('.')
            answer = '. '.join(sentences[:4]) + '.'
        
        return {"answer": answer, "source_documents": [], "context_used": context[:200] + "..."}

# Sidebar for file upload and configuration
st.sidebar.title("📁 Upload Documents")

# File upload
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or Text files",
    type=['pdf', 'txt'],
    accept_multiple_files=True,
    help="Upload one or more PDF or text files to ask questions about their content"
)

# Process uploaded files
processed_text = ""
if uploaded_files:
    all_text = []
    
    for uploaded_file in uploaded_files:
        st.sidebar.write(f"📄 Processing: {uploaded_file.name}")
        
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = extract_text_from_txt(uploaded_file)
        else:
            st.sidebar.error(f"Unsupported file type: {uploaded_file.type}")
            continue
        
        if text:
            all_text.append(f"\n\n--- From {uploaded_file.name} ---\n\n{text}")
            st.sidebar.success(f"✅ {uploaded_file.name} processed")
        else:
            st.sidebar.error(f"❌ Failed to process {uploaded_file.name}")
    
    processed_text = "\n".join(all_text)

# Model selection
st.sidebar.title("🤖 Model Configuration")

model_options = {
    "Hugging Face API (Recommended)": "hf_api",
    "Local FLAN-T5 Model": "local",
    "Mock LLM (Testing)": "mock"
}

selected_model = st.sidebar.selectbox(
    "Choose AI Model:",
    list(model_options.keys()),
    help="HF API requires token but gives better results. Local models work without tokens."
)

model_type = model_options[selected_model]

# Load selected model
@st.cache_resource
def get_llm_model(model_type):
    if model_type == "hf_api" and HF_TOKEN:
        return load_hf_inference_llm()
    elif model_type == "local":
        return load_local_llm()
    else:
        return EnhancedMockLLM()

llm = get_llm_model(model_type)

# Create vectorstore from uploaded documents
vectorstore = None
retriever = None

if processed_text:
    with st.spinner("🔍 Processing documents and creating search index..."):
        embeddings = load_embeddings()
        if embeddings:
            text_chunks = chunk_text(processed_text, chunk_size=800, overlap=100)
            vectorstore = create_vectorstore_from_text(text_chunks, embeddings)
            if vectorstore:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.sidebar.success(f"✅ Created search index with {len(text_chunks)} chunks")

# Create QA chain
qa_chain = DocumentQAChain(llm, retriever)

# Display status
st.sidebar.title("📊 Status")
if HF_TOKEN:
    st.sidebar.success("✅ HF Token configured")
else:
    st.sidebar.warning("⚠️ No HF Token - using local/mock models")

st.sidebar.info(f"🤖 Model: {selected_model}")

if uploaded_files:
    st.sidebar.success(f"📚 Documents: {len(uploaded_files)} loaded")
    if vectorstore:
        st.sidebar.success("🔍 Search index: Ready")
    else:
        st.sidebar.error("❌ Search index: Failed")
else:
    st.sidebar.warning("📁 No documents uploaded")

# Main interface
if not uploaded_files:
    st.warning("👆 Please upload PDF or text files using the sidebar to start asking questions!")
    st.markdown("""
    ### How to use this app:
    
    1. **Upload documents** - Use the sidebar to upload PDF or text files
    2. **Choose AI model** - Select from available options in the sidebar
    3. **Ask questions** - Type questions about your uploaded documents
    4. **Get answers** - The AI will search your documents and provide relevant answers
    
    ### Supported file types:
    - 📄 PDF files
    - 📝 Text files (.txt)
    
    ### Tips for better results:
    - Ask specific questions about the content
    - Use keywords that appear in your documents
    - Keep questions clear and focused
    """)
else:
    # Show document preview
    with st.expander("📖 Document Preview", expanded=False):
        preview_text = processed_text[:1500] + "..." if len(processed_text) > 1500 else processed_text
        st.text_area("Document Content Preview", preview_text, height=200)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "context" in message:
            with st.expander("📄 Source context used"):
                st.text(message["context"])

# Chat input
if prompt := st.chat_input("Ask me anything about your uploaded documents!", disabled=not uploaded_files):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Searching documents and generating answer..."):
            try:
                result = qa_chain({"question": prompt})
                response = result["answer"]
                
                st.markdown(response)
                
                # Store message with context info
                message_data = {"role": "assistant", "content": response}
                if "context_used" in result:
                    message_data["context"] = result["context_used"]
                    with st.expander("📄 Source context used"):
                        st.text(result["context_used"])
                
                st.session_state.messages.append(message_data)
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Help section
with st.sidebar.expander("ℹ️ Help & Info"):
    st.write("""
    **How it works:**
    1. Upload PDF/text files
    2. Files are split into searchable chunks
    3. AI searches relevant chunks for your question
    4. Provides answers based on document content
    
    **Model Options:**
    - **HF API**: Best quality, needs HF token
    - **Local**: Good quality, works offline
    - **Mock**: For testing without API keys
    
    **Setup:**
    - Add HF_TOKEN to Streamlit secrets for best results
    - Models will fallback automatically if tokens missing
    
    **Tips:**
    - Be specific in your questions
    - Ask about content that's actually in your documents
    - Try rephrasing if you don't get good answers
    """)

# Requirements info
st.sidebar.markdown("---")
st.sidebar.markdown("**Required packages:**")
st.sidebar.code("""
streamlit
torch
transformers
huggingface_hub
sentence-transformers
langchain
langchain-community
faiss-cpu
PyPDF2
numpy
""")
